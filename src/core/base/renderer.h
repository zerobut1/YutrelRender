#pragma once

#include <array>

#include <luisa/luisa-compute.h>

#include "base/camera.h"
#include "base/environment.h"
#include "base/external_scene.h"
#include "base/integrator.h"
#include "base/light.h"
#include "base/medium.h"
#include "base/spectrum.h"
#include "base/surface.h"
#include "base/texture.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;
using TextureSampler = compute::Sampler;

class Scene;
class Geometry;

struct RenderDiagnostics
{
    uint path_nan{};
    uint path_inf{};
    uint film_nan{};
    uint film_inf{};

    [[nodiscard]] uint64_t total() const noexcept
    {
        return static_cast<uint64_t>(path_nan) + path_inf + film_nan + film_inf;
    }
};

struct ExternalCameraState
{
    float4x4 camera_to_world;
    uint2 resolution;
    float vertical_fov_degrees;
};

class Renderer final
{
private:
    static constexpr uint diagnostic_path_nan = 0u;
    static constexpr uint diagnostic_path_inf = 1u;
    static constexpr uint diagnostic_film_nan = 2u;
    static constexpr uint diagnostic_film_inf = 3u;
    static constexpr size_t diagnostic_count  = 4u;

    Device& m_device;
    Buffer<uint> m_diagnostics;
    std::array<uint, diagnostic_count> m_diagnostic_zeros{};
    luisa::vector<luisa::unique_ptr<Resource>> m_resources;
    BindlessArray m_bindless_array;
    size_t m_bindless_buffer_count{0u};
    size_t m_bindless_tex2d_count{0u};
    size_t m_bindless_tex3d_count{0u};
    Polymorphic<Surface::Instance> m_surfaces;
    Polymorphic<Light::Instance> m_lights;
    Polymorphic<Medium::Instance> m_media;
    luisa::unordered_map<const Surface*, uint> m_surface_tags;
    luisa::unordered_map<const Light*, uint> m_light_tags;
    luisa::unordered_map<const Medium*, uint> m_medium_tags;
    luisa::unordered_map<const Texture*, luisa::unique_ptr<Texture::Instance>> m_textures;
    luisa::vector<Light::Handle> m_light_handles;

    luisa::unique_ptr<Spectrum::Instance> m_spectrum;
    luisa::unique_ptr<Environment::Instance> m_environment;
    luisa::unique_ptr<Camera::Instance> m_camera;
    luisa::unique_ptr<Integrator::Instance> m_integrator;
    luisa::unique_ptr<Geometry> m_geometry;

    luisa::unordered_map<luisa::string, uint> m_named_ids;

public:
    explicit Renderer(Device& device) noexcept;
    ~Renderer() noexcept;

    Renderer() noexcept                  = delete;
    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&)                 = delete;
    Renderer& operator=(Renderer&&)      = delete;

public:
    template <typename T, typename... Args>
        requires std::is_base_of_v<Resource, T>
    [[nodiscard]] auto create(Args&&... args) noexcept -> T*
    {
        auto resource = luisa::make_unique<T>(m_device.create<T>(std::forward<Args>(args)...));
        auto p        = resource.get();
        m_resources.emplace_back(std::move(resource));
        return p;
    }

    template <typename T>
    [[nodiscard]] BufferView<T> arena_buffer(size_t n) noexcept
    {
        return create<Buffer<T>>(n)->view();
    }

    template <typename T>
    [[nodiscard]] std::pair<BufferView<T>, uint /* bindless id */> bindless_arena_buffer(size_t n) noexcept
    {
        auto view      = arena_buffer<T>(n);
        auto buffer_id = register_bindless(view);
        return std::make_pair(view, buffer_id);
    }

    template <typename T>
    [[nodiscard]] auto register_bindless(BufferView<T> buffer) noexcept
    {
        auto buffer_id = m_bindless_buffer_count++;
        m_bindless_array.emplace_on_update(buffer_id, buffer);
        return static_cast<uint>(buffer_id);
    }

    template <typename T>
    [[nodiscard]] auto register_bindless(const Buffer<T>& buffer) noexcept
    {
        return register_bindless(buffer.view());
    }

    template <typename T>
    [[nodiscard]] auto register_bindless(const Image<T>& image, TextureSampler sampler) noexcept
    {
        auto tex2d_id = m_bindless_tex2d_count++;
        m_bindless_array.emplace_on_update(tex2d_id, image, sampler);
        return static_cast<uint>(tex2d_id);
    }

    template <typename T>
    [[nodiscard]] auto register_bindless(const Volume<T>& volume, TextureSampler sampler) noexcept
    {
        auto tex3d_id = m_bindless_tex3d_count++;
        m_bindless_array.emplace_on_update(tex3d_id, volume, sampler);
        return static_cast<uint>(tex3d_id);
    }

    [[nodiscard]] uint register_surface(CommandBuffer& command_buffer, const Surface* surface) noexcept;
    [[nodiscard]] uint register_light(CommandBuffer& command_buffer, const Light* light) noexcept;
    [[nodiscard]] uint register_medium(CommandBuffer& command_buffer, const Medium* medium) noexcept;

    template <typename Create>
    uint register_named_id(luisa::string_view identifier, Create&& create_id) noexcept
    {
        if (auto it = m_named_ids.find(identifier); it != m_named_ids.end())
        {
            return it->second;
        }
        auto new_id = std::invoke(std::forward<Create>(create_id));
        m_named_ids.emplace(identifier, new_id);
        return new_id;
    }

public:
    [[nodiscard]] static luisa::unique_ptr<Renderer> create(Device& device, Stream& stream, const Scene& scene) noexcept;

    void render(Stream& stream, bool enable_display);
    void render_interactive(Stream& stream);

    [[nodiscard]] bool prepare_external_render() noexcept;
    [[nodiscard]] bool prepare_external_scene_updates(
        CommandBuffer& command_buffer,
        luisa::span<const uint64_t> initial_instance_ids,
        const Surface* default_surface) noexcept;
    [[nodiscard]] bool update_external_scene(
        CommandBuffer& command_buffer,
        luisa::span<const ExternalMeshUpdate> mesh_updates,
        luisa::optional<ExternalDirectionalLightState> light_update) noexcept;
    [[nodiscard]] bool update_external_camera(
        CommandBuffer& command_buffer,
        const ExternalCameraState& state) noexcept;
    [[nodiscard]] bool render_external_sample(
        CommandBuffer& command_buffer,
        ImageView<float> accumulation,
        uint2 resolution,
        uint sample_index) noexcept;

    void reset_diagnostics(CommandBuffer& command_buffer) noexcept;
    void download_diagnostics(CommandBuffer& command_buffer, std::array<uint, diagnostic_count>& values) const noexcept;
    void record_path_non_finite(Expr<bool> has_nan, Expr<bool> has_inf) const noexcept;
    void record_film_non_finite(Expr<bool> has_nan, Expr<bool> has_inf) const noexcept;

    [[nodiscard]] auto& device() const noexcept { return m_device; }
    [[nodiscard]] auto& bindless_array() noexcept { return m_bindless_array; }
    [[nodiscard]] auto& bindless_array() const noexcept { return m_bindless_array; }
    [[nodiscard]] auto spectrum() const noexcept { return m_spectrum.get(); }
    [[nodiscard]] auto environment() const noexcept { return m_environment.get(); }
    [[nodiscard]] auto camera() const noexcept { return m_camera.get(); }
    [[nodiscard]] auto integrator() const noexcept { return m_integrator.get(); }
    [[nodiscard]] auto geometry() const noexcept { return m_geometry.get(); }
    [[nodiscard]] auto& surfaces() const noexcept { return m_surfaces; }
    [[nodiscard]] auto& lights() const noexcept { return m_lights; }
    [[nodiscard]] auto light_handles() const noexcept { return luisa::span{m_light_handles}; }
    [[nodiscard]] auto& media() const noexcept { return m_media; }
    [[nodiscard]] bool has_lighting() const noexcept { return !m_light_handles.empty() || m_environment != nullptr; }

    [[nodiscard]] const Texture::Instance* build_texture(CommandBuffer& command_buffer, const Texture* texture) noexcept;

    template <typename T, typename I>
    [[nodiscard]] auto buffer(I&& id) const noexcept { return m_bindless_array->buffer<T>(std::forward<I>(id)); }
    template <typename T>
    [[nodiscard]] auto tex2d(T&& id) const noexcept { return m_bindless_array->tex2d(std::forward<T>(id)); }
};

} // namespace Yutrel
