#include "unity_device_config.h"

#include <IUnityGraphics.h>
#include <IUnityInterface.h>

#include <Windows.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_set>

#include <luisa/backends/ext/native_resource_ext.hpp>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/luisa-compute.h>

#include "base/film.h"
#include "base/interaction.h"
#include "base/renderer.h"
#include "base/scene.h"
#include "base/texture.h"
#include "cameras/pinhole.h"
#include "environments/distant.h"
#include "filters/box.h"
#include "integrators/path.h"
#include "lights/diffuse.h"
#include "samplers/independent.h"
#include "scene/scene_builder.h"
#include "scene/scene_spec_builder.h"
#include "shapes/inline_mesh.h"
#include "spectrum/hero.h"
#include "surfaces/diffuse.h"
#include "textures/constant.h"
#include "textures/scale.h"

namespace yutrel::unity {

using namespace luisa;
using namespace luisa::compute;

constexpr uint32_t abi_version = 6u;
constexpr int clear_event_id = 0;
constexpr int path_trace_event_id = 1;
constexpr size_t external_instance_capacity = 4096u;

struct ClearEventData {
    uint32_t abi_version;
    uint32_t struct_size;
    ID3D12Resource *output;
};

enum SceneMeshOperation : uint32_t {
    mesh_add_or_replace = 1u,
    mesh_transform = 2u,
    mesh_remove = 3u,
};

enum class ExternalTextureEncoding : uint32_t {
    linear_srgb = 0u,
    srgb = 1u,
};

struct SceneSubMeshData {
    uint32_t index_offset;
    uint32_t index_count;
    float emissive_color[3];
    float emissive_luminance_nits;
    uint32_t double_sided;
    ExternalTextureEncoding texture_encoding;
    const float *emissive_pixels;
    uint32_t texture_width;
    uint32_t texture_height;
    float uv_scale[2];
    float uv_offset[2];
};

struct SceneMeshDelta {
    uint64_t mesh_id;
    uint32_t operation;
    uint32_t reserved;
    const float *positions;
    const float *normals;
    const float *uvs;
    const uint32_t *indices;
    const SceneSubMeshData *submeshes;
    uint32_t vertex_count;
    uint32_t index_count;
    uint32_t submesh_count;
    uint32_t submesh_struct_size;
    float local_to_world[16];
};

static_assert(sizeof(SceneSubMeshData) == 64u);
static_assert(sizeof(SceneMeshDelta) == 136u);

struct DirectionalLightData {
    float color[3];
    float illuminance_lux;
    float direction[3];
    uint32_t enabled;
};

struct SceneDeltaData {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t revision;
    const SceneMeshDelta *meshes;
    uint32_t mesh_count;
    uint32_t mesh_struct_size;
    DirectionalLightData light;
    uint32_t light_changed;
};

static_assert(sizeof(DirectionalLightData) == 32u);
static_assert(sizeof(SceneDeltaData) == 72u);

struct PathTraceEventData {
    uint32_t abi_version;
    uint32_t struct_size;
    ID3D12Resource *output;
    uint32_t width;
    uint32_t height;
    uint32_t view_id;
    uint32_t flip_output_y;
    float camera_to_world[16];
    float vertical_fov_degrees;
    float pre_exposure;
    uint32_t reset_accumulation;
};

struct NativeTextureDesc {
    D3D12_RESOURCE_STATES init_state;
    DXGI_FORMAT custom_format;
    bool allow_uav;
};

namespace {

[[nodiscard]] bool finite(float3 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] float4x4 load_matrix(const float *values) noexcept {
    float4x4 matrix{};
    for (auto column = 0u; column < 4u; column++) {
        for (auto row = 0u; row < 4u; row++) {
            matrix[column][row] = values[column * 4u + row];
        }
    }
    return matrix;
}

[[nodiscard]] bool matrix_near(
    const float4x4 &lhs,
    const float4x4 &rhs,
    float epsilon = 1e-6f) noexcept {
    for (auto column = 0u; column < 4u; column++) {
        for (auto row = 0u; row < 4u; row++) {
            if (std::abs(lhs[column][row] - rhs[column][row]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

struct MeshSnapshot {
    struct SubMesh {
        uint32_t index_offset;
        uint32_t index_count;
        float3 emissive_color;
        float emissive_luminance_nits;
        bool double_sided;
        ExternalTextureEncoding texture_encoding;
        luisa::vector<float4> emissive_pixels;
        uint2 texture_size;
        float2 uv_scale;
        float2 uv_offset;

        [[nodiscard]] bool emissive() const noexcept {
            return emissive_luminance_nits > 0.0f && any(emissive_color > 0.0f);
        }
    };

    luisa::vector<float3> positions;
    luisa::vector<float3> normals;
    luisa::vector<float2> uvs;
    luisa::vector<uint3> triangles;
    luisa::vector<SubMesh> submeshes;
    float4x4 local_to_world;
    uint64_t geometry_revision{};

    [[nodiscard]] bool has_emissive() const noexcept {
        for (auto &&submesh : submeshes) {
            if (submesh.emissive()) {
                return true;
            }
        }
        return false;
    }
};

struct DesiredScene {
    luisa::unordered_map<uint64_t, MeshSnapshot> meshes;
    Yutrel::ExternalDirectionalLightState light{
        .color = make_float3(0.0f),
        .illuminance_lux = 0.0f,
        .direction = make_float3(0.0f, 0.0f, 1.0f),
        .enabled = 0u,
    };
    uint64_t revision{};

    [[nodiscard]] bool has_emissive() const noexcept {
        for (auto &&[id, mesh] : meshes) {
            static_cast<void>(id);
            if (mesh.has_emissive()) {
                return true;
            }
        }
        return false;
    }
};

[[nodiscard]] bool copy_mesh(
    const SceneMeshDelta &data,
    uint64_t revision,
    MeshSnapshot &mesh) {
    if (data.positions == nullptr || data.normals == nullptr ||
        data.indices == nullptr || data.submeshes == nullptr ||
        data.vertex_count == 0u || data.index_count == 0u ||
        data.index_count % 3u != 0u || data.submesh_count == 0u ||
        data.submesh_struct_size != sizeof(SceneSubMeshData)) {
        return false;
    }
    mesh.positions.reserve(data.vertex_count);
    mesh.normals.reserve(data.vertex_count);
    for (auto i = 0u; i < data.vertex_count; i++) {
        auto position = make_float3(
            data.positions[i * 3u],
            data.positions[i * 3u + 1u],
            data.positions[i * 3u + 2u]);
        auto normal = make_float3(
            data.normals[i * 3u],
            data.normals[i * 3u + 1u],
            data.normals[i * 3u + 2u]);
        auto normal_length_squared = dot(normal, normal);
        if (!finite(position) || !finite(normal) ||
            !std::isfinite(normal_length_squared) || normal_length_squared < 1e-12f) {
            return false;
        }
        mesh.positions.emplace_back(position);
        mesh.normals.emplace_back(normalize(normal));
    }
    if (data.uvs != nullptr) {
        mesh.uvs.reserve(data.vertex_count);
        for (auto i = 0u; i < data.vertex_count; i++) {
            auto uv = make_float2(data.uvs[i * 2u], data.uvs[i * 2u + 1u]);
            if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) {
                return false;
            }
            mesh.uvs.emplace_back(uv);
        }
    }

    mesh.triangles.reserve(data.index_count / 3u);
    for (auto i = 0u; i < data.index_count; i += 3u) {
        auto triangle = make_uint3(
            data.indices[i],
            data.indices[i + 1u],
            data.indices[i + 2u]);
        if (triangle.x >= data.vertex_count ||
            triangle.y >= data.vertex_count || triangle.z >= data.vertex_count) {
            return false;
        }
        mesh.triangles.emplace_back(triangle);
    }

    auto expected_index_offset = 0u;
    mesh.submeshes.reserve(data.submesh_count);
    for (auto submesh_index = 0u; submesh_index < data.submesh_count; submesh_index++) {
        auto &&source = data.submeshes[submesh_index];
        auto color = make_float3(
            source.emissive_color[0],
            source.emissive_color[1],
            source.emissive_color[2]);
        auto texture_size = make_uint2(source.texture_width, source.texture_height);
        auto texture_pixel_count = static_cast<uint64_t>(texture_size.x) * texture_size.y;
        auto has_texture = texture_size.x != 0u || texture_size.y != 0u;
        if (source.index_offset != expected_index_offset || source.index_count == 0u ||
            source.index_count % 3u != 0u ||
            static_cast<uint64_t>(source.index_offset) + source.index_count > data.index_count ||
            !finite(color) || color.x < 0.0f || color.y < 0.0f || color.z < 0.0f ||
            !std::isfinite(source.emissive_luminance_nits) ||
            source.emissive_luminance_nits < 0.0f || source.double_sided > 1u ||
            source.texture_encoding > ExternalTextureEncoding::srgb ||
            (has_texture && (texture_size.x == 0u || texture_size.y == 0u ||
                             source.emissive_pixels == nullptr || mesh.uvs.empty())) ||
            (!has_texture && source.emissive_pixels != nullptr) ||
            texture_pixel_count > std::numeric_limits<uint32_t>::max() ||
            !std::isfinite(source.uv_scale[0]) || !std::isfinite(source.uv_scale[1]) ||
            !std::isfinite(source.uv_offset[0]) || !std::isfinite(source.uv_offset[1])) {
            return false;
        }
        MeshSnapshot::SubMesh submesh{
            .index_offset = source.index_offset,
            .index_count = source.index_count,
            .emissive_color = color,
            .emissive_luminance_nits = source.emissive_luminance_nits,
            .double_sided = source.double_sided != 0u,
            .texture_encoding = source.texture_encoding,
            .texture_size = texture_size,
            .uv_scale = make_float2(source.uv_scale[0], source.uv_scale[1]),
            .uv_offset = make_float2(source.uv_offset[0], source.uv_offset[1]),
        };
        submesh.emissive_pixels.reserve(texture_pixel_count);
        for (auto pixel_index = 0ull; pixel_index < texture_pixel_count; pixel_index++) {
            auto pixel = make_float4(
                source.emissive_pixels[pixel_index * 4u],
                source.emissive_pixels[pixel_index * 4u + 1u],
                source.emissive_pixels[pixel_index * 4u + 2u],
                source.emissive_pixels[pixel_index * 4u + 3u]);
            if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
                !std::isfinite(pixel.z) || !std::isfinite(pixel.w) ||
                any(pixel < 0.0f)) {
                return false;
            }
            submesh.emissive_pixels.emplace_back(pixel);
        }
        mesh.submeshes.emplace_back(std::move(submesh));
        expected_index_offset += source.index_count;
    }
    if (expected_index_offset != data.index_count) {
        return false;
    }

    mesh.local_to_world = load_matrix(data.local_to_world);
    if (Yutrel::validate_camera_to_world(mesh.local_to_world)) {
        return false;
    }
    mesh.geometry_revision = revision;
    return true;
}

[[nodiscard]] luisa::optional<Yutrel::ExternalDirectionalLightState>
copy_light(const DirectionalLightData &data) {
    auto light = Yutrel::ExternalDirectionalLightState{
        .color = make_float3(data.color[0], data.color[1], data.color[2]),
        .illuminance_lux = data.illuminance_lux,
        .direction = make_float3(data.direction[0], data.direction[1], data.direction[2]),
        .enabled = data.enabled,
    };
    auto direction_length_squared = dot(light.direction, light.direction);
    if (!finite(light.color) || light.color.x < 0.0f || light.color.y < 0.0f ||
        light.color.z < 0.0f || !std::isfinite(light.illuminance_lux) ||
        light.illuminance_lux < 0.0f || !finite(light.direction) ||
        !std::isfinite(direction_length_squared) ||
        std::abs(direction_length_squared - 1.0f) > 1e-4f || light.enabled > 1u) {
        return luisa::nullopt;
    }
    light.direction = normalize(light.direction);
    return light;
}

class UnityImageTexture final : public Yutrel::Texture {
public:
    class Instance final : public Yutrel::Texture::Instance {
    private:
        uint32_t _texture_id;

    public:
        Instance(
            const Yutrel::Renderer &renderer,
            const Yutrel::Texture *texture,
            uint32_t texture_id) noexcept
            : Yutrel::Texture::Instance{renderer, texture},
              _texture_id{texture_id} {}

        [[nodiscard]] Float4 evaluate(
            const Yutrel::Interaction &interaction,
            Expr<float> time) const noexcept override {
            static_cast<void>(time);
            auto texture = base<UnityImageTexture>();
            auto uv = interaction.uv * texture->uv_scale() + texture->uv_offset();
            auto value = renderer().tex2d(_texture_id).sample(uv);
            if (texture->encoding() == ExternalTextureEncoding::srgb) {
                auto rgb = value.xyz();
                auto linear = ite(
                    rgb <= 0.04045f,
                    rgb * (1.0f / 12.92f),
                    pow((rgb + 0.055f) * (1.0f / 1.055f), 2.4f));
                value = make_float4(linear, value.w);
            }
            return value;
        }
    };

private:
    luisa::vector<float4> _pixels;
    uint2 _size;
    ExternalTextureEncoding _encoding;
    float2 _uv_scale;
    float2 _uv_offset;

public:
    UnityImageTexture(
        luisa::vector<float4> pixels,
        uint2 size,
        ExternalTextureEncoding encoding,
        float2 uv_scale,
        float2 uv_offset) noexcept
        : _pixels{std::move(pixels)},
          _size{size},
          _encoding{encoding},
          _uv_scale{uv_scale},
          _uv_offset{uv_offset} {}

    [[nodiscard]] luisa::unique_ptr<Yutrel::Texture::Instance> build(
        Yutrel::Renderer &renderer,
        Yutrel::CommandBuffer &command_buffer) const noexcept override {
        auto image = renderer.create<Image<float>>(PixelStorage::FLOAT4, _size);
        auto texture_id = renderer.register_bindless(
            *image,
            Yutrel::TextureSampler::linear_point_repeat());
        command_buffer << image->copy_from(_pixels.data()) << commit();
        return luisa::make_unique<Instance>(renderer, this, texture_id);
    }

    [[nodiscard]] uint channels() const noexcept override { return 4u; }
    [[nodiscard]] uint2 resolution() const noexcept override { return _size; }
    [[nodiscard]] auto encoding() const noexcept { return _encoding; }
    [[nodiscard]] auto uv_scale() const noexcept { return _uv_scale; }
    [[nodiscard]] auto uv_offset() const noexcept { return _uv_offset; }
};

class UnityImageTextureSpec final : public Yutrel::TextureSpec {
private:
    luisa::vector<float4> _pixels;
    uint2 _size;
    ExternalTextureEncoding _encoding;
    float2 _uv_scale;
    float2 _uv_offset;

public:
    UnityImageTextureSpec(
        luisa::vector<float4> pixels,
        uint2 size,
        ExternalTextureEncoding encoding,
        float2 uv_scale,
        float2 uv_offset) noexcept
        : _pixels{std::move(pixels)},
          _size{size},
          _encoding{encoding},
          _uv_scale{uv_scale},
          _uv_offset{uv_offset} {}

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override {
        auto pixel_count = static_cast<uint64_t>(_size.x) * _size.y;
        return _size.x != 0u && _size.y != 0u && pixel_count == _pixels.size()
                   ? luisa::nullopt
                   : Yutrel::spec_validation_error("Unity image texture dimensions are invalid.");
    }

    [[nodiscard]] const Yutrel::Texture *build(
        Yutrel::SceneBuilder &builder) const noexcept override {
        return builder.emplace<Yutrel::Texture, UnityImageTexture>(
            _pixels,
            _size,
            _encoding,
            _uv_scale,
            _uv_offset);
    }
};

[[nodiscard]] luisa::unique_ptr<Yutrel::Scene> create_scene(
    const DesiredScene &snapshot,
    const Yutrel::ExternalCameraState &camera,
    luisa::vector<uint64_t> &instance_ids) {
    using namespace Yutrel;

    SceneSpecBuilder builder;
    SourceLocation source{.file = "<unity-static-scene>"};
    auto albedo = builder.add_anonymous_texture<ConstantTextureSpec>(
        source, make_float4(0.5f, 0.5f, 0.5f, 1.0f));
    auto emission = builder.add_anonymous_texture<ConstantTextureSpec>(
        source, make_float4(1.0f));
    auto surface = builder.add_anonymous_surface<DiffuseSurfaceSpec>(source, albedo, true);
    auto spectrum = builder.add_anonymous_spectrum<HeroWavelengthSpectrumSpec>(source);
    auto environment = builder.add_anonymous_environment<DistantEnvironmentSpec>(
        source,
        emission,
        1.0f,
        make_float3(0.0f, 0.0f, 1.0f));
    auto camera_ref = builder.add_anonymous_camera<PinholeCameraSpec>(
        source,
        camera.camera_to_world,
        make_float3(camera.camera_to_world[1]),
        make_float2(0.0f),
        0u,
        camera.vertical_fov_degrees);
    auto film = builder.add_anonymous_film<RGBFilmSpec>(
        source,
        camera.resolution,
        false,
        "unity-unused.exr",
        1.0f);
    auto filter = builder.add_anonymous_filter<BoxFilterSpec>(source, 0.5f);
    auto sampler = builder.add_anonymous_sampler<IndependentSamplerSpec>(source, 1u, 0u);
    auto integrator = builder.add_anonymous_integrator<PathIntegratorSpec>(source, 4u);
    auto contains_emissive = snapshot.has_emissive();
    if (!contains_emissive) {
        instance_ids.reserve(snapshot.meshes.size());
        for (auto &&[mesh_id, mesh] : snapshot.meshes) {
            auto shape = builder.add_anonymous_shape<InlineMeshShapeSpec>(
                source,
                mesh.positions,
                mesh.normals,
                mesh.uvs,
                mesh.triangles);
            builder.add_instance(ShapeInstanceSpec{
                .source = source,
                .shape = shape,
                .surface = surface,
                .transform = mesh.local_to_world,
            });
            instance_ids.emplace_back(mesh_id);
        }
    } else {
        for (auto &&[mesh_id, mesh] : snapshot.meshes) {
            static_cast<void>(mesh_id);
            for (auto &&submesh : mesh.submeshes) {
                auto triangle_begin = submesh.index_offset / 3u;
                auto triangle_count = submesh.index_count / 3u;
                luisa::vector<uint3> triangles;
                triangles.reserve(triangle_count);
                for (auto triangle_index = 0u; triangle_index < triangle_count; triangle_index++) {
                    triangles.emplace_back(mesh.triangles[triangle_begin + triangle_index]);
                }
                auto shape = builder.add_anonymous_shape<InlineMeshShapeSpec>(
                    source,
                    mesh.positions,
                    mesh.normals,
                    mesh.uvs,
                    std::move(triangles));
                ShapeInstanceSpec instance{
                    .source = source,
                    .shape = shape,
                    .surface = surface,
                    .transform = mesh.local_to_world,
                };
                if (submesh.emissive()) {
                    auto emissive_texture = [&]() -> TextureRef {
                        if (submesh.emissive_pixels.empty()) {
                            return builder.add_anonymous_texture<ConstantTextureSpec>(
                                source,
                                make_float4(submesh.emissive_color, 1.0f));
                        }
                        auto image = builder.add_anonymous_texture<UnityImageTextureSpec>(
                            source,
                            submesh.emissive_pixels,
                            submesh.texture_size,
                            submesh.texture_encoding,
                            submesh.uv_scale,
                            submesh.uv_offset);
                        return builder.add_anonymous_texture<ScaleTextureSpec>(
                            source,
                            image,
                            make_float4(submesh.emissive_color, 1.0f),
                            make_float4(0.0f));
                    }();
                    instance.light = builder.add_anonymous_light<DiffuseLightSpec>(
                        source,
                        emissive_texture,
                        submesh.emissive_luminance_nits,
                        submesh.double_sided);
                }
                builder.add_instance(std::move(instance));
            }
        }
    }
    builder.set_render(RenderSpec{
        .spectrum = spectrum,
        .environment = environment,
        .camera = camera_ref,
        .film = film,
        .filter = filter,
        .sampler = sampler,
        .integrator = integrator,
    });
    auto spec = builder.finish();
    return Scene::create(spec);
}

}// namespace

class Plugin final {
private:
    struct ViewRuntime {
        Image<float> accumulation;
        Yutrel::ExternalCameraState camera{};
        uint sample_index{};
        bool camera_valid{};
    };

    struct SceneRuntime {
        struct AppliedMesh {
            uint64_t geometry_revision;
            float4x4 local_to_world;
        };

        luisa::unique_ptr<Yutrel::Scene> scene;
        luisa::unique_ptr<Yutrel::Renderer> renderer;
        luisa::vector<luisa::unique_ptr<Yutrel::InlineMesh>> dynamic_shapes;
        luisa::unordered_map<uint64_t, AppliedMesh> applied_meshes;
        luisa::unordered_map<uint32_t, ViewRuntime> views;
        Yutrel::ExternalCameraState active_camera{};
        bool active_camera_valid{};
        bool contains_emissive{};
        uint64_t applied_revision{};

        [[nodiscard]] ViewRuntime &view(uint32_t view_id) {
            return views[view_id];
        }

        ~SceneRuntime() noexcept {
            for (auto &[view_id, view] : views) {
                static_cast<void>(view_id);
                view.accumulation = {};
            }
            views.clear();
            renderer = nullptr;
            dynamic_shapes.clear();
            scene = nullptr;
        }
    };

    enum PathError : uint32_t {
        path_error_output = 1u << 0u,
        path_error_scene = 1u << 1u,
        path_error_runtime = 1u << 2u,
    };

    Context _context;
    Device _device;
    Stream _stream;
    UnityDeviceConfig *_unity_config{nullptr};
    Shader2D<Image<float>> _clear_shader;
    Shader2D<Image<float>> _black_shader;
    Shader2D<Image<float>> _clear_accumulation_shader;
    Shader2D<Image<float>, Image<float>, float, uint> _present_shader;
    DesiredScene _desired_scene;
    luisa::unique_ptr<SceneRuntime> _scene_runtime;
    uint32_t _reported_path_errors{};

public:
    Plugin(luisa::string module_path, luisa::unique_ptr<UnityDeviceConfig> config)
        : _context{module_path},
          _unity_config{config.get()} {
        DeviceConfig device_config{
            .extension = std::move(config),
            .inqueue_buffer_limit = false,
        };
        _device = _context.create_device("dx", &device_config);
        _stream = _device.create_stream(StreamTag::GRAPHICS);

        Kernel2D clear = [](ImageFloat output) noexcept {
            output.write(dispatch_id().xy(), make_float4(0.18f, 0.36f, 0.72f, 1.0f));
        };
        Kernel2D black = [](ImageFloat output) noexcept {
            output.write(dispatch_id().xy(), make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        };
        Kernel2D clear_accumulation = [](ImageFloat accumulation) noexcept {
            accumulation.write(dispatch_id().xy(), make_float4(0.0f));
        };
        Kernel2D present = [](
                               ImageFloat accumulation,
                               ImageFloat output,
                               Float pre_exposure,
                               UInt flip_output_y) noexcept {
            auto pixel = dispatch_id().xy();
            auto sum = accumulation.read(pixel);
            auto rgb = ite(sum.w > 0.0f, sum.xyz() / sum.w, make_float3(0.0f));
            auto invalid = any(compute::isnan(rgb)) | any(compute::isinf(rgb));
            rgb = ite(invalid, make_float3(0.0f), rgb);
            auto output_y = ite(
                flip_output_y != 0u,
                dispatch_size().y - 1u - pixel.y,
                pixel.y);
            auto output_pixel = make_uint2(pixel.x, output_y);
            output.write(output_pixel, make_float4(rgb * pre_exposure, 1.0f));
        };
        _clear_shader = _device.compile(clear);
        _black_shader = _device.compile(black);
        _clear_accumulation_shader = _device.compile(clear_accumulation);
        _present_shader = _device.compile(present);
    }

    ~Plugin() noexcept {
        try {
            if (_stream) {
                _stream << synchronize();
            }
        } catch (...) {
            LUISA_WARNING_WITH_LOCATION("Failed to synchronize YutrelUnityPlugin during shutdown.");
        }
        _scene_runtime = nullptr;
    }

    [[nodiscard]] int submit_scene_delta(const SceneDeltaData &data) noexcept {
        if (data.revision <= _desired_scene.revision) {
            return 2;
        }
        try {
            if ((data.mesh_count != 0u && data.meshes == nullptr) ||
                data.mesh_struct_size != sizeof(SceneMeshDelta) ||
                data.light_changed > 1u) {
                return 1;
            }
            struct OwnedDelta {
                uint64_t id;
                SceneMeshOperation operation;
                luisa::optional<MeshSnapshot> mesh;
                float4x4 transform;
            };
            luisa::vector<OwnedDelta> deltas;
            deltas.reserve(data.mesh_count);
            std::unordered_set<uint64_t> ids;
            for (auto i = 0u; i < data.mesh_count; i++) {
                auto &source = data.meshes[i];
                if (source.mesh_id == 0u || !ids.emplace(source.mesh_id).second ||
                    source.operation < mesh_add_or_replace || source.operation > mesh_remove) {
                    return 1;
                }
                auto operation = static_cast<SceneMeshOperation>(source.operation);
                auto existing = _desired_scene.meshes.find(source.mesh_id);
                OwnedDelta delta{
                    .id = source.mesh_id,
                    .operation = operation,
                    .transform = load_matrix(source.local_to_world),
                };
                if (operation == mesh_add_or_replace) {
                    MeshSnapshot mesh;
                    if (!copy_mesh(source, data.revision, mesh)) {
                        return 1;
                    }
                    delta.transform = mesh.local_to_world;
                    delta.mesh = std::move(mesh);
                } else if (existing == _desired_scene.meshes.end() ||
                           (operation == mesh_transform &&
                            Yutrel::validate_camera_to_world(delta.transform))) {
                    return 1;
                }
                deltas.emplace_back(std::move(delta));
            }

            auto light = luisa::optional<Yutrel::ExternalDirectionalLightState>{};
            if (data.light_changed != 0u) {
                light = copy_light(data.light);
                if (!light) {
                    return 1;
                }
            }
            auto resulting_mesh_count = _desired_scene.meshes.size();
            for (auto &delta : deltas) {
                auto exists = _desired_scene.meshes.find(delta.id) != _desired_scene.meshes.end();
                if (delta.operation == mesh_add_or_replace && !exists) {
                    resulting_mesh_count++;
                } else if (delta.operation == mesh_remove) {
                    resulting_mesh_count--;
                }
            }
            if (resulting_mesh_count > external_instance_capacity) {
                return 1;
            }
            for (auto &delta : deltas) {
                switch (delta.operation) {
                    case mesh_add_or_replace:
                        _desired_scene.meshes.insert_or_assign(delta.id, std::move(*delta.mesh));
                        break;
                    case mesh_transform:
                        _desired_scene.meshes.at(delta.id).local_to_world = delta.transform;
                        break;
                    case mesh_remove:
                        _desired_scene.meshes.erase(delta.id);
                        break;
                }
            }
            if (light) {
                _desired_scene.light = *light;
            }
            _desired_scene.revision = data.revision;
            return 0;
        } catch (...) {
            return 3;
        }
    }

    void clear(const ClearEventData &data) noexcept {
        try {
            auto output = wrap_output(data.output, 0u, 0u);
            if (!output) {
                return;
            }
            auto commands = CommandList::create(1u, 1u);
            commands << _clear_shader(output).dispatch(output.size());
            commands.add_callback([output = std::move(output)]() mutable {});
            _stream << commands.commit() << synchronize();
        } catch (const std::exception &exception) {
            LUISA_WARNING_WITH_LOCATION("Luisa fixed-color render event failed: {}", exception.what());
        } catch (...) {
            LUISA_WARNING_WITH_LOCATION("Luisa fixed-color render event failed with an unknown error.");
        }
    }

    void path_trace(const PathTraceEventData &data) noexcept {
        Image<float> output;
        try {
            output = wrap_output(data.output, data.width, data.height);
            if (!output) {
                report_path_error_once(path_error_output, "Unity Path Tracing output is invalid.");
                return;
            }
            if (!render_path(output, data) && output) {
                write_black(std::move(output));
            }
        } catch (const std::exception &exception) {
            report_path_error_once(
                path_error_runtime,
                luisa::format("Unity Path Tracing event failed: {}", exception.what()));
            if (output) {
                write_black(std::move(output));
            }
        } catch (...) {
            report_path_error_once(path_error_runtime, "Unity Path Tracing event failed.");
            if (output) {
                write_black(std::move(output));
            }
        }
    }

private:
    [[nodiscard]] Image<float> wrap_output(
        ID3D12Resource *resource,
        uint32_t expected_width,
        uint32_t expected_height) {
        if (resource == nullptr) {
            return {};
        }
        auto description = resource->GetDesc();
        if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
            description.Format != DXGI_FORMAT_R16G16B16A16_FLOAT ||
            description.Width == 0u ||
            description.Width > std::numeric_limits<uint32_t>::max() ||
            description.Height == 0u ||
            (expected_width != 0u && description.Width != expected_width) ||
            (expected_height != 0u && description.Height != expected_height)) {
            return {};
        }

        auto native_resources = _device.extension<NativeResourceExt>();
        if (native_resources == nullptr) {
            return {};
        }
        NativeTextureDesc native_description{
            .init_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            .custom_format = DXGI_FORMAT_UNKNOWN,
            .allow_uav = true,
        };
        auto output = native_resources->create_native_image<float>(
            resource,
            static_cast<uint32_t>(description.Width),
            description.Height,
            PixelStorage::HALF4,
            1u,
            &native_description);
        if (output) {
            _unity_config->register_resource(resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        return output;
    }

    [[nodiscard]] bool ensure_runtime(
        const Yutrel::ExternalCameraState &camera) {
        if (_scene_runtime != nullptr) {
            return true;
        }
        if (_desired_scene.meshes.empty()) {
            return false;
        }

        auto runtime = luisa::make_unique<SceneRuntime>();
        luisa::vector<uint64_t> instance_ids;
        runtime->contains_emissive = _desired_scene.has_emissive();
        runtime->scene = create_scene(_desired_scene, camera, instance_ids);
        if (runtime->scene == nullptr) {
            report_path_error_once(path_error_scene, "Failed to create the Unity Yutrel scene.");
            return false;
        }
        if (dynamic_cast<const Yutrel::HeroWavelengthSpectrum *>(
                runtime->scene->spectrum()) == nullptr) {
            report_path_error_once(path_error_scene, "Unity Yutrel scene did not create the required Hero spectrum.");
            return false;
        }
        runtime->renderer = Yutrel::Renderer::create(_device, _stream, *runtime->scene);
        if (runtime->renderer == nullptr) {
            report_path_error_once(path_error_scene, "Failed to create the Unity Path Tracing renderer.");
            return false;
        }
        auto instances = runtime->scene->instances();
        if (instances.empty()) {
            return false;
        }
        Yutrel::CommandBuffer commands{_stream};
        if ((!runtime->contains_emissive &&
             !runtime->renderer->prepare_external_scene_updates(
                 commands,
                 instance_ids,
                 instances.front().surface)) ||
            !runtime->renderer->update_external_scene(
                commands,
                {},
                _desired_scene.light)) {
            commands << commit();
            report_path_error_once(path_error_scene, "Failed to prepare Unity scene hot updates.");
            return false;
        }
        commands << synchronize();
        if (!runtime->renderer->prepare_external_render()) {
            report_path_error_once(path_error_scene, "Failed to compile the Unity Path Tracing renderer.");
            return false;
        }
        for (auto &[id, mesh] : _desired_scene.meshes) {
            runtime->applied_meshes.emplace(id, SceneRuntime::AppliedMesh{
                .geometry_revision = mesh.geometry_revision,
                .local_to_world = mesh.local_to_world,
            });
        }
        runtime->applied_revision = _desired_scene.revision;
        _scene_runtime = std::move(runtime);
        return true;
    }

    [[nodiscard]] bool scene_requires_rebuild(const SceneRuntime &runtime) const noexcept {
        auto desired_contains_emissive = _desired_scene.has_emissive();
        if (runtime.contains_emissive != desired_contains_emissive) {
            return true;
        }
        if (!runtime.contains_emissive) {
            return false;
        }
        if (runtime.applied_meshes.size() != _desired_scene.meshes.size()) {
            return true;
        }
        for (auto &&[id, mesh] : _desired_scene.meshes) {
            auto applied = runtime.applied_meshes.find(id);
            if (applied == runtime.applied_meshes.end() ||
                applied->second.geometry_revision != mesh.geometry_revision ||
                !matrix_near(applied->second.local_to_world, mesh.local_to_world)) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool apply_scene_updates(
        SceneRuntime &runtime,
        Yutrel::CommandBuffer &commands) {
        if (runtime.applied_revision == _desired_scene.revision) {
            return true;
        }

        luisa::vector<Yutrel::ExternalMeshUpdate> updates;
        luisa::vector<luisa::unique_ptr<Yutrel::InlineMesh>> new_shapes;
        for (auto &[id, applied] : runtime.applied_meshes) {
            static_cast<void>(applied);
            if (_desired_scene.meshes.find(id) == _desired_scene.meshes.end()) {
                updates.emplace_back(Yutrel::ExternalMeshUpdate{
                    .id = id,
                    .operation = Yutrel::ExternalMeshOp::remove,
                });
            }
        }
        for (auto &[id, mesh] : _desired_scene.meshes) {
            auto applied = runtime.applied_meshes.find(id);
            if (applied == runtime.applied_meshes.end() ||
                applied->second.geometry_revision != mesh.geometry_revision) {
                auto shape = luisa::make_unique<Yutrel::InlineMesh>(
                    mesh.positions,
                    mesh.normals,
                    mesh.uvs,
                    mesh.triangles);
                updates.emplace_back(Yutrel::ExternalMeshUpdate{
                    .id = id,
                    .operation = Yutrel::ExternalMeshOp::add_or_replace,
                    .shape = shape.get(),
                    .local_to_world = mesh.local_to_world,
                });
                new_shapes.emplace_back(std::move(shape));
            } else if (!matrix_near(applied->second.local_to_world, mesh.local_to_world)) {
                updates.emplace_back(Yutrel::ExternalMeshUpdate{
                    .id = id,
                    .operation = Yutrel::ExternalMeshOp::transform,
                    .local_to_world = mesh.local_to_world,
                });
            }
        }
        if (!runtime.renderer->update_external_scene(
                commands,
                updates,
                _desired_scene.light)) {
            return false;
        }
        for (auto &shape : new_shapes) {
            runtime.dynamic_shapes.emplace_back(std::move(shape));
        }
        runtime.applied_meshes.clear();
        for (auto &[id, mesh] : _desired_scene.meshes) {
            runtime.applied_meshes.emplace(id, SceneRuntime::AppliedMesh{
                .geometry_revision = mesh.geometry_revision,
                .local_to_world = mesh.local_to_world,
            });
        }
        runtime.applied_revision = _desired_scene.revision;
        for (auto &[view_id, view] : runtime.views) {
            static_cast<void>(view_id);
            if (view.accumulation) {
                commands << _clear_accumulation_shader(view.accumulation).dispatch(view.accumulation.size());
            }
            view.sample_index = 0u;
        }
        return true;
    }

    [[nodiscard]] bool render_path(
        Image<float> &output,
        const PathTraceEventData &data) {
        if (data.width == 0u || data.height == 0u ||
            data.view_id == 0u ||
            data.flip_output_y > 1u ||
            !std::isfinite(data.vertical_fov_degrees) ||
            data.vertical_fov_degrees <= 0.0f || data.vertical_fov_degrees >= 180.0f ||
            !std::isfinite(data.pre_exposure) || data.pre_exposure <= 0.0f) {
            report_path_error_once(path_error_output, "Unity Path Tracing event parameters are invalid.");
            return false;
        }
        Yutrel::ExternalCameraState camera{
            .camera_to_world = load_matrix(data.camera_to_world),
            .resolution = make_uint2(data.width, data.height),
            .vertical_fov_degrees = data.vertical_fov_degrees,
        };
        if (Yutrel::validate_camera_to_world(camera.camera_to_world)) {
            report_path_error_once(path_error_output, "Unity Path Tracing camera is invalid.");
            return false;
        }
        if (!ensure_runtime(camera)) {
            return false;
        }

        if (scene_requires_rebuild(*_scene_runtime)) {
            _scene_runtime = nullptr;
            if (!ensure_runtime(camera)) {
                report_path_error_once(path_error_scene, "Failed to rebuild the Unity emissive scene.");
                return false;
            }
        }

        auto &runtime = *_scene_runtime;
        Yutrel::CommandBuffer commands{_stream};
        if (!apply_scene_updates(runtime, commands)) {
            commands << synchronize();
            _scene_runtime = nullptr;
            report_path_error_once(path_error_runtime, "Failed to apply a Unity scene update.");
            return false;
        }
        auto &view = runtime.view(data.view_id);
        auto resolution_changed = !view.accumulation ||
                                  view.accumulation.size().x != camera.resolution.x ||
                                  view.accumulation.size().y != camera.resolution.y;
        auto view_changed = !view.camera_valid || resolution_changed ||
                            !matrix_near(view.camera.camera_to_world, camera.camera_to_world) ||
                            std::abs(view.camera.vertical_fov_degrees - camera.vertical_fov_degrees) > 1e-6f;
        auto camera_upload_needed = !runtime.active_camera_valid ||
                                    runtime.active_camera.resolution.x != camera.resolution.x ||
                                    runtime.active_camera.resolution.y != camera.resolution.y ||
                                    !matrix_near(runtime.active_camera.camera_to_world, camera.camera_to_world) ||
                                    std::abs(runtime.active_camera.vertical_fov_degrees - camera.vertical_fov_degrees) > 1e-6f;
        if (resolution_changed) {
            _stream << synchronize();
            view.accumulation = _device.create_image<float>(PixelStorage::FLOAT4, camera.resolution);
        }

        if (camera_upload_needed && !runtime.renderer->update_external_camera(commands, camera)) {
            commands << commit();
            report_path_error_once(path_error_runtime, "Failed to update the Unity Path Tracing camera.");
            return false;
        }
        runtime.active_camera = camera;
        runtime.active_camera_valid = true;

        auto reset = data.reset_accumulation != 0u || view_changed;
        if (reset) {
            commands << _clear_accumulation_shader(view.accumulation).dispatch(camera.resolution);
            view.sample_index = 0u;
        }
        view.camera = camera;
        view.camera_valid = true;

        if (!runtime.renderer->render_external_sample(
                commands,
                view.accumulation,
                camera.resolution,
                view.sample_index)) {
            commands << commit();
            report_path_error_once(path_error_runtime, "Failed to record a Unity Path Tracing sample.");
            return false;
        }
        view.sample_index++;
        commands
            << _present_shader(
                   view.accumulation,
                   output,
                   data.pre_exposure,
                   data.flip_output_y)
                   .dispatch(camera.resolution)
            << [output = std::move(output)]() mutable {}
            << synchronize();
        return true;
    }

    void write_black(Image<float> output) noexcept {
        try {
            auto size = output.size();
            auto commands = CommandList::create(1u, 1u);
            commands << _black_shader(output).dispatch(size);
            commands.add_callback([output = std::move(output)]() mutable {});
            _stream << commands.commit() << synchronize();
        } catch (...) {
            report_path_error_once(path_error_runtime, "Failed to write the Unity Path Tracing black fallback.");
        }
    }

    void report_path_error_once(uint32_t bit, luisa::string_view message) noexcept {
        if ((_reported_path_errors & bit) != 0u) {
            return;
        }
        _reported_path_errors |= bit;
        LUISA_WARNING_WITH_LOCATION("{}", message);
    }
};

namespace {

IUnityGraphicsD3D12v8 *unity_graphics = nullptr;
luisa::unique_ptr<Plugin> plugin;
std::mutex plugin_mutex;

[[nodiscard]] luisa::string current_module_path() {
    HMODULE module = nullptr;
    auto address = reinterpret_cast<LPCSTR>(&current_module_path);
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address,
            &module)) {
        return {};
    }

    luisa::vector<char> path(32768u, '\0');
    auto length = GetModuleFileNameA(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0u || length >= path.size()) {
        return {};
    }
    return luisa::string{path.data(), length};
}

void UNITY_INTERFACE_API on_render_event(int event_id, void *event_data) {
    std::scoped_lock lock{plugin_mutex};
    switch (event_id) {
        case clear_event_id: {
            auto data = std::unique_ptr<ClearEventData>{static_cast<ClearEventData *>(event_data)};
            if (data != nullptr && data->abi_version == abi_version &&
                data->struct_size == sizeof(ClearEventData) && plugin != nullptr) {
                plugin->clear(*data);
            }
            break;
        }
        case path_trace_event_id: {
            auto data = std::unique_ptr<PathTraceEventData>{static_cast<PathTraceEventData *>(event_data)};
            if (data != nullptr && data->abi_version == abi_version &&
                data->struct_size == sizeof(PathTraceEventData) && plugin != nullptr) {
                plugin->path_trace(*data);
            }
            break;
        }
        default:
            break;
    }
}

}// namespace

}// namespace yutrel::unity

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces *interfaces) {
    using namespace yutrel::unity;
    std::scoped_lock lock{plugin_mutex};
    unity_graphics = interfaces == nullptr
                         ? nullptr
                         : interfaces->Get<IUnityGraphicsD3D12v8>();
    if (unity_graphics == nullptr) {
        return;
    }

    UnityD3D12PluginEventConfig config{};
    config.graphicsQueueAccess = kUnityD3D12GraphicsQueueAccess_Allow;
    config.flags = kUnityD3D12EventConfigFlag_FlushCommandBuffers |
                   kUnityD3D12EventConfigFlag_SyncWorkerThreads |
                   kUnityD3D12EventConfigFlag_ModifiesCommandBuffersState;
    config.ensureActiveRenderTextureIsBound = false;
    unity_graphics->ConfigureEvent(clear_event_id, &config);
    unity_graphics->ConfigureEvent(path_trace_event_id, &config);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
UnityPluginUnload() {
    using namespace yutrel::unity;
    std::scoped_lock lock{plugin_mutex};
    plugin = nullptr;
    unity_graphics = nullptr;
}

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API
YutrelUnityInitialize() {
    using namespace yutrel::unity;
    std::scoped_lock lock{plugin_mutex};
    if (plugin != nullptr) {
        return 0;
    }
    if (unity_graphics == nullptr) {
        return 1;
    }

    try {
        auto config = luisa::make_unique<UnityDeviceConfig>(unity_graphics);
        if (!config->valid()) {
            return 2;
        }
        auto module_path = current_module_path();
        if (module_path.empty()) {
            return 3;
        }
        plugin = luisa::make_unique<Plugin>(std::move(module_path), std::move(config));
        return 0;
    } catch (const std::exception &exception) {
        LUISA_WARNING_WITH_LOCATION(
            "YutrelUnityPlugin initialization failed: {}",
            exception.what());
    } catch (...) {
        LUISA_WARNING_WITH_LOCATION("YutrelUnityPlugin initialization failed with an unknown error.");
    }
    plugin = nullptr;
    return 4;
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
YutrelUnityShutdown() {
    using namespace yutrel::unity;
    std::scoped_lock lock{plugin_mutex};
    plugin = nullptr;
}

extern "C" UNITY_INTERFACE_EXPORT UnityRenderingEventAndData UNITY_INTERFACE_API
YutrelUnityGetRenderEventFunc() {
    return yutrel::unity::on_render_event;
}

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API
YutrelUnitySubmitSceneDelta(const yutrel::unity::SceneDeltaData *data) {
    using namespace yutrel::unity;
    if (data == nullptr || data->abi_version != abi_version ||
        data->struct_size != sizeof(SceneDeltaData)) {
        return 1;
    }
    std::scoped_lock lock{plugin_mutex};
    return plugin == nullptr ? 4 : plugin->submit_scene_delta(*data);
}

extern "C" UNITY_INTERFACE_EXPORT void *UNITY_INTERFACE_API
YutrelUnityCreateClearEvent(const yutrel::unity::ClearEventData *data) {
    using namespace yutrel::unity;
    if (data == nullptr || data->abi_version != abi_version ||
        data->struct_size != sizeof(ClearEventData) || data->output == nullptr) {
        return nullptr;
    }
    return new (std::nothrow) ClearEventData{*data};
}

extern "C" UNITY_INTERFACE_EXPORT void *UNITY_INTERFACE_API
YutrelUnityCreatePathTraceEvent(const yutrel::unity::PathTraceEventData *data) {
    using namespace yutrel::unity;
    if (data == nullptr || data->abi_version != abi_version ||
        data->struct_size != sizeof(PathTraceEventData) || data->output == nullptr ||
        data->width == 0u || data->height == 0u ||
        data->view_id == 0u ||
        data->flip_output_y > 1u) {
        return nullptr;
    }
    return new (std::nothrow) PathTraceEventData{*data};
}
