#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>

#include <luisa/luisa-compute.h>

#include "base/film.h"
#include "base/renderer.h"
#include "base/scene.h"
#include "cameras/pinhole.h"
#include "environments/latlong.h"
#include "filters/box.h"
#include "integrators/path.h"
#include "samplers/independent.h"
#include "scene/scene_spec_builder.h"
#include "shapes/inline_mesh.h"
#include "spectrum/hero.h"
#include "surfaces/diffuse.h"
#include "textures/constant.h"
#include "textures/image.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{

constexpr auto test_width = 8u;
constexpr auto test_height = 4u;

class TemporaryDirectory
{
private:
    std::filesystem::path _path;

public:
    TemporaryDirectory()
        : _path{std::filesystem::temp_directory_path() /
                ("yutrel_latlong_environment_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))}
    {
        std::filesystem::create_directories(_path);
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(_path, error);
    }

    [[nodiscard]] const auto& path() const noexcept { return _path; }
};

[[nodiscard]] uint32_t byte_swap_32(uint32_t v) noexcept
{
    return ((v & 0x000000ffu) << 24u) |
           ((v & 0x0000ff00u) << 8u) |
           ((v & 0x00ff0000u) >> 8u) |
           ((v & 0xff000000u) >> 24u);
}

void write_latlong_pfm(
    const std::filesystem::path& path,
    uint width,
    uint height,
    luisa::span<const float> top_down_values)
{
    std::ofstream stream{path, std::ios::binary};
    stream << "PF\n" << width << ' ' << height << "\n-1\n";
    constexpr auto host_little_endian = std::endian::native == std::endian::little;
    for (auto y = height; y-- > 0u;)
    {
        for (auto x = 0u; x < width; x++)
        {
            auto value = top_down_values[static_cast<size_t>(y) * width + x];
            for (auto c = 0u; c < 3u; c++)
            {
                uint32_t bits{};
                std::memcpy(&bits, &value, sizeof(bits));
                if (!host_little_endian)
                {
                    bits = byte_swap_32(bits);
                }
                stream.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
            }
        }
    }
}

[[nodiscard]] bool nearly_equal(
    float a, float b, float epsilon = 2.0e-4f) noexcept
{
    auto scale = std::max({1.0f, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= epsilon * scale;
}

[[nodiscard]] luisa::unique_ptr<Scene> create_scene(
    const std::filesystem::path& image_path,
    float scale,
    float3x3 transform_to_world)
{
    SceneSpecBuilder builder;
    SourceLocation source{.file = "latlong-environment-test"};
    auto image = builder.add_anonymous_texture<ImageTextureSpec>(
        source,
        image_path,
        TextureSampler::linear_point_repeat(),
        Texture::Encoding::LINEAR);
    auto albedo = builder.add_anonymous_texture<ConstantTextureSpec>(
        source, make_float4(0.5f, 0.5f, 0.5f, 1.0f));
    auto surface = builder.add_anonymous_surface<DiffuseSurfaceSpec>(source, albedo, true);
    auto shape = builder.add_anonymous_shape<InlineMeshShapeSpec>(
        source,
        luisa::vector{
            make_float3(-0.5f, -0.5f, -3.0f),
            make_float3(0.5f, -0.5f, -3.0f),
            make_float3(0.0f, 0.5f, -3.0f)},
        luisa::vector(3u, make_float3(0.0f, 0.0f, 1.0f)),
        luisa::vector<float2>{},
        luisa::vector{make_uint3(0u, 1u, 2u)});
    auto spectrum = builder.add_anonymous_spectrum<HeroWavelengthSpectrumSpec>(source);
    auto environment = builder.add_anonymous_environment<LatLongEnvironmentSpec>(
        source, image, scale, transform_to_world);
    auto camera = builder.add_anonymous_camera<PinholeCameraSpec>(
        source,
        make_float4x4(1.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float2(0.0f),
        0u,
        45.0f);
    auto film = builder.add_anonymous_film<RGBFilmSpec>(
        source, make_uint2(1u), false, image_path.parent_path() / "unused.exr");
    auto filter = builder.add_anonymous_filter<BoxFilterSpec>(source, 0.5f);
    auto sampler = builder.add_anonymous_sampler<IndependentSamplerSpec>(source, 1u, 0u);
    auto integrator = builder.add_anonymous_integrator<PathIntegratorSpec>(source, 1u);
    builder.add_instance(ShapeInstanceSpec{
        .source = source,
        .shape = shape,
        .surface = surface,
    });
    builder.set_render(RenderSpec{
        .spectrum = spectrum,
        .environment = environment,
        .camera = camera,
        .film = film,
        .filter = filter,
        .sampler = sampler,
        .integrator = integrator,
    });
    auto spec = builder.finish();
    return Scene::create(spec);
}

[[nodiscard]] bool test_mapping(Device& device, Stream& stream)
{
    std::array directions{
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, -1.0f),
        make_float3(-1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(0.0f, -1.0f, 0.0f),
        normalize(make_float3(0.3f, 0.7f, 0.2f)),
        normalize(make_float3(-0.8f, -0.1f, 0.5f)),
    };
    auto input = device.create_buffer<float3>(directions.size());
    auto output = device.create_buffer<float4>(directions.size());
    Kernel1D kernel = [](BufferFloat3 input_buffer, BufferFloat4 output_buffer) noexcept
    {
        auto i = dispatch_id().x;
        auto direction = input_buffer.read(i);
        auto uv = LatLongEnvironment::direction_to_uv(direction);
        auto recovered = LatLongEnvironment::uv_to_direction(uv);
        output_buffer.write(i, make_float4(uv, dot(direction, recovered), length(recovered)));
    };
    auto shader = device.compile(kernel);
    std::array<float4, directions.size()> values{};
    stream << input.copy_from(directions.data())
           << shader(input, output).dispatch(directions.size())
           << output.copy_to(values.data())
           << synchronize();

    auto seam_u = std::min(std::abs(values[2u].x), std::abs(values[2u].x - 1.0f));
    return nearly_equal(values[0u].x, 0.5f) && nearly_equal(values[0u].y, 0.5f) &&
           nearly_equal(values[1u].x, 0.75f) && nearly_equal(values[1u].y, 0.5f) &&
           seam_u < 1.0e-5f && nearly_equal(values[2u].y, 0.5f) &&
           nearly_equal(values[3u].x, 0.25f) && nearly_equal(values[3u].y, 0.5f) &&
           nearly_equal(values[4u].y, 0.0f) && nearly_equal(values[5u].y, 1.0f) &&
           std::all_of(values.begin(), values.end(), [](auto value) noexcept
           {
               return value.z > 0.999f && nearly_equal(value.w, 1.0f, 1.0e-4f);
           });
}

[[nodiscard]] bool test_spec_validation()
{
    SceneSpecBuilder builder;
    SourceLocation source{.file = "latlong-validation-test"};
    auto texture = builder.add_anonymous_texture<ConstantTextureSpec>(
        source, make_float4(1.0f));
    auto identity = make_float3x3(1.0f);
    auto scaled = identity;
    scaled[0] *= 2.0f;
    auto non_finite = identity;
    non_finite[1].y = std::numeric_limits<float>::infinity();
    return !LatLongEnvironmentSpec{texture, 1.0f, identity}.validate() &&
           LatLongEnvironmentSpec{texture, -1.0f, identity}.validate().has_value() &&
           LatLongEnvironmentSpec{texture, 1.0f, scaled}.validate().has_value() &&
           LatLongEnvironmentSpec{texture, 1.0f, non_finite}.validate().has_value();
}

[[nodiscard]] bool test_distribution(
    Device& device,
    Stream& stream,
    Renderer& renderer,
    uint width,
    uint height,
    luisa::span<const float> radiance)
{
    if (renderer.environment() == nullptr)
    {
        return false;
    }
    auto pixel_count = width * height;
    auto output = device.create_buffer<float4>(pixel_count + 4u);
    Kernel1D kernel = [&](BufferFloat4 result) noexcept
    {
        auto swl = renderer.spectrum()->sample(0.5f);
        for (auto i = 0u; i < pixel_count; i++)
        {
            auto uv = make_float2(
                (static_cast<float>(i % width) + 0.5f) / static_cast<float>(width),
                (static_cast<float>(i / width) + 0.5f) / static_cast<float>(height));
            auto wi = LatLongEnvironment::uv_to_direction(uv);
            auto complete = renderer.environment()->evaluate(wi, swl, 0.0f, false);
            auto incomplete = renderer.environment()->evaluate(wi, swl, 0.0f, true);
            result.write(i, make_float4(
                complete.pdf, incomplete.pdf, length(wi), complete.L[0u]));
        }

        auto complete_sample = renderer.environment()->sample(
            swl, 0.0f, make_float2(0.37f, 0.73f), false);
        auto complete_evaluation = renderer.environment()->evaluate(
            complete_sample.wi, swl, 0.0f, false);
        result.write(pixel_count, make_float4(
            complete_sample.eval.pdf,
            complete_evaluation.pdf,
            complete_sample.eval.L[0u],
            complete_evaluation.L[0u]));

        auto incomplete_sample = renderer.environment()->sample(
            swl, 0.0f, make_float2(0.19f, 0.61f), true);
        auto incomplete_evaluation = renderer.environment()->evaluate(
            incomplete_sample.wi, swl, 0.0f, true);
        result.write(pixel_count + 1u, make_float4(
            incomplete_sample.eval.pdf,
            incomplete_evaluation.pdf,
            incomplete_sample.eval.L[0u],
            incomplete_evaluation.L[0u]));

        auto north = renderer.environment()->evaluate(
            make_float3(0.0f, 1.0f, 0.0f), swl, 0.0f, false);
        auto south = renderer.environment()->evaluate(
            make_float3(0.0f, -1.0f, 0.0f), swl, 0.0f, false);
        result.write(pixel_count + 2u, make_float4(
            north.pdf, south.pdf, north.L[0u], south.L[0u]));

        auto seam_a = renderer.environment()->evaluate(
            normalize(make_float3(1.0e-6f, 0.0f, -1.0f)), swl, 0.0f, false);
        auto seam_b = renderer.environment()->evaluate(
            normalize(make_float3(-1.0e-6f, 0.0f, -1.0f)), swl, 0.0f, false);
        result.write(pixel_count + 3u, make_float4(
            seam_a.pdf, seam_b.pdf, seam_a.L[0u], seam_b.L[0u]));
    };
    auto shader = device.compile(kernel);
    luisa::vector<float4> values(pixel_count + 4u);
    stream << shader(output).dispatch(1u)
           << output.copy_to(values.data())
           << synchronize();

    luisa::vector<double> weights(pixel_count);
    for (auto i = 0u; i < pixel_count; i++)
    {
        auto y = i / width;
        weights[i] = radiance[i] * std::sin(
            std::acos(-1.0) * (static_cast<double>(y) + 0.5) / height);
    }
    auto total = std::accumulate(weights.begin(), weights.end(), 0.0);
    auto average = total / pixel_count;
    luisa::vector<double> compensated(pixel_count);
    std::transform(weights.begin(), weights.end(), compensated.begin(),
                   [average](auto value) noexcept { return std::max(value - average, 0.0); });
    auto compensated_total = std::accumulate(compensated.begin(), compensated.end(), 0.0);
    if (compensated_total == 0.0)
    {
        std::fill(compensated.begin(), compensated.end(), 1.0);
        compensated_total = pixel_count;
    }

    auto complete_integral = 0.0;
    auto incomplete_integral = 0.0;
    auto pi_d = std::acos(-1.0);
    for (auto i = 0u; i < pixel_count; i++)
    {
        auto y = i / width;
        auto sin_theta = std::sin(pi_d * (static_cast<double>(y) + 0.5) / height);
        auto jacobian = 2.0 * pi_d * pi_d * sin_theta;
        auto expected_complete = weights[i] / total * pixel_count / jacobian;
        auto expected_incomplete = compensated[i] / compensated_total * pixel_count / jacobian;
        if (!nearly_equal(values[i].x, static_cast<float>(expected_complete), 5.0e-4f) ||
            !nearly_equal(values[i].y, static_cast<float>(expected_incomplete), 5.0e-4f) ||
            !nearly_equal(values[i].z, 1.0f, 1.0e-4f) ||
            !std::isfinite(values[i].w) || values[i].w <= 0.0f)
        {
            return false;
        }
        complete_integral += values[i].x * jacobian / pixel_count;
        incomplete_integral += values[i].y * jacobian / pixel_count;
    }

    auto complete_sample = values[pixel_count];
    auto incomplete_sample = values[pixel_count + 1u];
    auto poles = values[pixel_count + 2u];
    auto seam = values[pixel_count + 3u];
    auto finite_positive = [](float value) noexcept
    {
        return std::isfinite(value) && value > 0.0f;
    };
    return nearly_equal(static_cast<float>(complete_integral), 1.0f, 5.0e-4f) &&
           nearly_equal(static_cast<float>(incomplete_integral), 1.0f, 5.0e-4f) &&
           nearly_equal(complete_sample.x, complete_sample.y, 5.0e-4f) &&
           nearly_equal(complete_sample.z, complete_sample.w, 5.0e-4f) &&
           nearly_equal(incomplete_sample.x, incomplete_sample.y, 5.0e-4f) &&
           nearly_equal(incomplete_sample.z, incomplete_sample.w, 5.0e-4f) &&
           finite_positive(complete_sample.x) && finite_positive(incomplete_sample.x) &&
           poles.x == 0.0f && poles.y == 0.0f &&
           finite_positive(poles.z) && finite_positive(poles.w) &&
           finite_positive(seam.x) && finite_positive(seam.y) &&
           finite_positive(seam.z) && finite_positive(seam.w);
}

[[nodiscard]] float2 evaluate_direction(
    Device& device,
    Stream& stream,
    Renderer& renderer,
    float3 direction)
{
    auto output = device.create_buffer<float2>(1u);
    Kernel1D kernel = [&](BufferFloat2 result) noexcept
    {
        auto swl = renderer.spectrum()->sample(0.5f);
        auto evaluation = renderer.environment()->evaluate(direction, swl, 0.0f, false);
        result.write(0u, make_float2(evaluation.L[0u], evaluation.pdf));
    };
    auto shader = device.compile(kernel);
    float2 value{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(&value)
           << synchronize();
    return value;
}

[[nodiscard]] bool test_scale_and_rotation(
    Device& device,
    Stream& stream,
    const Scene& identity_scene,
    const Scene& scaled_scene,
    const Scene& rotated_scene)
{
    auto identity = Renderer::create(device, stream, identity_scene);
    auto scaled = Renderer::create(device, stream, scaled_scene);
    auto rotated = Renderer::create(device, stream, rotated_scene);
    if (!identity || !scaled || !rotated ||
        identity->environment() == nullptr || scaled->environment() == nullptr ||
        rotated->environment() == nullptr)
    {
        return false;
    }
    auto identity_z = evaluate_direction(
        device, stream, *identity, make_float3(0.0f, 0.0f, 1.0f));
    auto scaled_z = evaluate_direction(
        device, stream, *scaled, make_float3(0.0f, 0.0f, 1.0f));
    auto rotated_x = evaluate_direction(
        device, stream, *rotated, make_float3(1.0f, 0.0f, 0.0f));
    return identity_z.x > 0.0f &&
           nearly_equal(scaled_z.x, 2.0f * identity_z.x, 5.0e-4f) &&
           nearly_equal(scaled_z.y, identity_z.y, 5.0e-4f) &&
           nearly_equal(rotated_x.x, identity_z.x, 5.0e-4f) &&
           nearly_equal(rotated_x.y, identity_z.y, 5.0e-4f);
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        LUISA_WARNING("Usage: {} <backend>", argv[0]);
        return 1;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    if (!test_mapping(device, stream))
    {
        LUISA_WARNING("Lat-long direction mapping test failed.");
        return 2;
    }
    if (!test_spec_validation())
    {
        LUISA_WARNING("Lat-long spec validation test failed.");
        return 3;
    }

    TemporaryDirectory directory;
    luisa::vector<float> hotspot(test_width * test_height);
    for (auto y = 0u; y < test_height; y++)
    {
        for (auto x = 0u; x < test_width; x++)
        {
            hotspot[y * test_width + x] =
                0.25f + 0.1f * static_cast<float>(x) + 0.2f * static_cast<float>(y);
        }
    }
    hotspot[1u * test_width + 6u] += 4.0f;
    auto hotspot_path = directory.path() / "hotspot.pfm";
    write_latlong_pfm(hotspot_path, test_width, test_height, hotspot);

    luisa::vector<float> constant(8u, 2.0f);
    auto constant_path = directory.path() / "constant.pfm";
    write_latlong_pfm(constant_path, 4u, 2u, constant);

    luisa::vector<float> black(8u, 0.0f);
    auto black_path = directory.path() / "black.pfm";
    write_latlong_pfm(black_path, 4u, 2u, black);

    auto identity = make_float3x3(1.0f);
    auto hotspot_scene = create_scene(hotspot_path, 1.0f, identity);
    auto hotspot_renderer = Renderer::create(device, stream, *hotspot_scene);
    if (!hotspot_renderer ||
        !test_distribution(
            device, stream, *hotspot_renderer,
            test_width, test_height, hotspot))
    {
        LUISA_WARNING("Lat-long hotspot distribution test failed.");
        return 4;
    }

    auto constant_scene = create_scene(constant_path, 1.0f, identity);
    auto constant_renderer = Renderer::create(device, stream, *constant_scene);
    if (!constant_renderer ||
        !test_distribution(device, stream, *constant_renderer, 4u, 2u, constant))
    {
        LUISA_WARNING("Lat-long constant distribution test failed.");
        return 5;
    }

    auto scaled_scene = create_scene(hotspot_path, 2.0f, identity);
    auto rotation_y_90 = make_float3x3(
        make_float3(0.0f, 0.0f, -1.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(1.0f, 0.0f, 0.0f));
    auto rotated_scene = create_scene(hotspot_path, 1.0f, rotation_y_90);
    if (!test_scale_and_rotation(
            device, stream, *hotspot_scene, *scaled_scene, *rotated_scene))
    {
        LUISA_WARNING("Lat-long scale/rotation test failed.");
        return 6;
    }

    auto black_scene = create_scene(black_path, 1.0f, identity);
    auto black_renderer = Renderer::create(device, stream, *black_scene);
    auto zero_scale_scene = create_scene(hotspot_path, 0.0f, identity);
    auto zero_scale_renderer = Renderer::create(device, stream, *zero_scale_scene);
    if (!black_renderer || black_renderer->environment() != nullptr ||
        !zero_scale_renderer || zero_scale_renderer->environment() != nullptr)
    {
        LUISA_WARNING("Lat-long black/zero-scale fallback test failed.");
        return 7;
    }
    return 0;
}
