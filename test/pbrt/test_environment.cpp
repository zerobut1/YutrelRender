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

#include "base/renderer.h"
#include "base/scene.h"
#include "environments/distant.h"
#include "environments/grouped.h"
#include "environments/uniform.h"
#include "pbrt/pbrt_importer.h"
#include "pbrt/pbrt_parser.h"
#include "utils/image_io.h"
#include "utils/sampling.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{

[[nodiscard]] uint32_t byte_swap_32(uint32_t v) noexcept
{
    return ((v & 0x000000ffu) << 24u) |
           ((v & 0x0000ff00u) << 8u) |
           ((v & 0x00ff0000u) >> 8u) |
           ((v & 0xff000000u) >> 24u);
}

void write_pfm(const std::filesystem::path& path, const char* magic,
               uint width, uint height, float scale,
               luisa::span<const float> values)
{
    std::ofstream stream{path, std::ios::binary};
    stream << magic << '\n'
           << width << ' ' << height << '\n'
           << scale << '\n';
    auto file_little_endian           = scale < 0.0f;
    constexpr auto host_little_endian = std::endian::native == std::endian::little;
    for (auto value : values)
    {
        uint32_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        if (file_little_endian != host_little_endian)
        {
            bits = byte_swap_32(bits);
        }
        stream.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
    }
}

[[nodiscard]] bool nearly_equal(float a, float b, float epsilon = 2e-4f) noexcept
{
    return std::abs(a - b) <= epsilon;
}

[[nodiscard]] bool test_alias_tables() noexcept
{
    std::array zeros{0.0f, 0.0f, 0.0f, 0.0f};
    auto [zero_aliases, zero_pdf] = create_alias_table(zeros);
    for (auto i = 0u; i < zeros.size(); i++)
    {
        if (zero_aliases[i].prob != 1.0f || zero_aliases[i].alias != i ||
            !nearly_equal(zero_pdf[i], 0.25f))
        {
            return false;
        }
    }
    std::array values{1.0f, 2.0f, 3.0f, 4.0f};
    auto [aliases, pdf] = create_alias_table(values);
    auto sum            = 0.0f;
    for (auto p : pdf)
    {
        if (!std::isfinite(p) || p < 0.0f)
        {
            return false;
        }
        sum += p;
    }
    return aliases.size() == values.size() && nearly_equal(sum, 1.0f);
}

[[nodiscard]] bool test_pfm_loading()
{
    auto directory = std::filesystem::temp_directory_path();
    auto rgb_path  = directory / "yutrel_environment_rgb.pfm";
    auto gray_path = directory / "yutrel_environment_gray.pfm";

    // PFM scanlines are stored bottom-up. The file values below contain the
    // bottom row first and the top row second.
    std::array rgb_values{
        10.0f,
        11.0f,
        12.0f,
        20.0f,
        21.0f,
        22.0f,
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
    };
    write_pfm(rgb_path, "PF", 2u, 2u, -2.0f, rgb_values);
    auto rgb        = LoadedImage::load(rgb_path);
    auto rgb_pixels = static_cast<const float*>(rgb.pixels());
    auto rgb_valid  = rgb.size().x == 2u && rgb.size().y == 2u &&
                      rgb.pixel_storage() == PixelStorage::FLOAT4 &&
                      nearly_equal(rgb_pixels[0u], 2.0f) &&
                      nearly_equal(rgb_pixels[1u], 4.0f) &&
                      nearly_equal(rgb_pixels[2u], 6.0f) &&
                      nearly_equal(rgb_pixels[3u], 1.0f) &&
                      nearly_equal(rgb_pixels[8u], 20.0f);

    std::array gray_values{8.0f, 4.0f};
    write_pfm(gray_path, "Pf", 1u, 2u, 0.5f, gray_values);
    auto gray        = LoadedImage::load(gray_path);
    auto gray_pixels = static_cast<const float*>(gray.pixels());
    auto gray_valid  = gray.size().x == 1u && gray.size().y == 2u &&
                       gray.pixel_storage() == PixelStorage::FLOAT1 &&
                       nearly_equal(gray_pixels[0u], 2.0f) &&
                       nearly_equal(gray_pixels[1u], 4.0f);

    std::error_code error;
    std::filesystem::remove(rgb_path, error);
    std::filesystem::remove(gray_path, error);
    return rgb_valid && gray_valid;
}

[[nodiscard]] bool test_equal_area_mapping(Device& device, Stream& stream)
{
    std::array directions{
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(-1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(0.0f, -1.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 0.0f, -1.0f),
    };
    auto direction_input      = device.create_buffer<float3>(directions.size());
    auto direction_output     = device.create_buffer<float4>(directions.size());
    Kernel1D direction_kernel = [](BufferFloat3 input, BufferFloat4 output) noexcept
    {
        auto i         = dispatch_id().x;
        auto direction = input.read(i);
        auto uv        = equal_area_sphere_to_square(direction);
        auto recovered = equal_area_square_to_sphere(uv);
        output.write(i, make_float4(recovered, length(recovered)));
    };
    auto test_directions = device.compile(direction_kernel);
    std::array<float4, directions.size()> recovered_directions{};
    stream << direction_input.copy_from(directions.data())
           << test_directions(direction_input, direction_output).dispatch(directions.size())
           << direction_output.copy_to(recovered_directions.data())
           << synchronize();
    for (auto i = 0u; i < directions.size(); i++)
    {
        auto recovered = recovered_directions[i].xyz();
        if (!nearly_equal(recovered_directions[i].w, 1.0f, 1e-3f) ||
            dot(recovered, directions[i]) < 0.999f)
        {
            return false;
        }
    }

    std::array<float2, 32u> uvs{};
    for (auto i = 0u; i < uvs.size(); i++)
    {
        // Stay away from the measure-zero square boundary where equivalent
        // seam representations are expected.
        uvs[i] = make_float2(
            0.02f + 0.96f * std::fmod(i * 0.61803398875f, 1.0f),
            0.02f + 0.96f * std::fmod(i * 0.41421356237f, 1.0f));
    }
    auto uv_input      = device.create_buffer<float2>(uvs.size());
    auto uv_output     = device.create_buffer<float2>(uvs.size());
    Kernel1D uv_kernel = [](BufferFloat2 input, BufferFloat2 output) noexcept
    {
        auto i  = dispatch_id().x;
        auto uv = input.read(i);
        output.write(i, equal_area_sphere_to_square(equal_area_square_to_sphere(uv)));
    };
    auto test_uvs = device.compile(uv_kernel);
    std::array<float2, uvs.size()> recovered_uvs{};
    stream << uv_input.copy_from(uvs.data())
           << test_uvs(uv_input, uv_output).dispatch(uvs.size())
           << uv_output.copy_to(recovered_uvs.data())
           << synchronize();
    for (auto i = 0u; i < uvs.size(); i++)
    {
        if (!nearly_equal(uvs[i].x, recovered_uvs[i].x, 2e-3f) ||
            !nearly_equal(uvs[i].y, recovered_uvs[i].y, 2e-3f))
        {
            return false;
        }
    }
    return true;
}

class TemporaryDirectory
{
private:
    std::filesystem::path _path;

public:
    TemporaryDirectory()
        : _path{std::filesystem::temp_directory_path() /
                ("yutrel_environment_distribution_" +
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

[[nodiscard]] std::filesystem::path write_environment_scene(
    const std::filesystem::path& root, const char* name,
    const std::array<float, 4u>& weights)
{
    auto directory = root / name;
    std::filesystem::create_directories(directory);
    auto image_path = directory / "environment.pfm";
    luisa::vector<float> pixels;
    pixels.reserve(12u);
    for (auto y = 2u; y-- > 0u;)
    {
        for (auto x = 0u; x < 2u; x++)
        {
            auto value = weights[y * 2u + x];
            pixels.emplace_back(value);
            pixels.emplace_back(value);
            pixels.emplace_back(value);
        }
    }
    write_pfm(image_path, "PF", 2u, 2u, -1.0f, pixels);

    auto scene_path = directory / "scene.pbrt";
    std::ofstream scene{scene_path};
    scene << "Integrator \"path\"\n"
             "Sampler \"independent\"\n"
             "    \"integer pixelsamples\" [ 1 ]\n"
             "PixelFilter \"triangle\"\n"
             "Film \"rgb\"\n"
             "    \"string filename\" [ \"output.exr\" ]\n"
             "    \"integer xresolution\" [ 1 ]\n"
             "    \"integer yresolution\" [ 1 ]\n"
             "Camera \"perspective\"\n"
             "WorldBegin\n"
             "Material \"diffuse\"\n"
             "    \"rgb reflectance\" [ 0.5 0.5 0.5 ]\n"
             "Shape \"sphere\"\n"
             "LightSource \"infinite\"\n"
             "    \"string filename\" [ \"environment.pfm\" ]\n";
    return scene_path;
}

[[nodiscard]] bool test_environment_distribution_case(
    Device& device, Stream& stream, const std::filesystem::path& scene_path,
    const std::array<float, 4u>& weights)
{
    auto parsed = PbrtParser::parse(scene_path);
    auto spec = PbrtImporter::import(std::move(parsed));
    auto scene = Scene::create(spec);
    auto renderer = Renderer::create(device, stream, *scene);
    if (renderer->environment() == nullptr)
    {
        return false;
    }

    auto output = device.create_buffer<float4>(5u);
    Kernel1D kernel = [&renderer](BufferFloat4 result) noexcept
    {
        auto swl = renderer->spectrum()->sample(0.5f);
        for (auto i = 0u; i < 4u; i++)
        {
            auto uv = make_float2(
                (static_cast<float>(i % 2u) + 0.5f) * 0.5f,
                (static_cast<float>(i / 2u) + 0.5f) * 0.5f);
            auto wi = equal_area_square_to_sphere(uv);
            auto complete = renderer->environment()->evaluate(wi, swl, 0.0f, false);
            auto incomplete = renderer->environment()->evaluate(wi, swl, 0.0f, true);
            result.write(i, make_float4(complete.pdf, incomplete.pdf, length(wi), 0.0f));
        }
        auto complete_sample = renderer->environment()->sample(
            swl, 0.0f, make_float2(0.37f, 0.73f), false);
        auto incomplete_sample = renderer->environment()->sample(
            swl, 0.0f, make_float2(0.37f, 0.73f), true);
        auto complete_evaluation = renderer->environment()->evaluate(
            complete_sample.wi, swl, 0.0f, false);
        auto incomplete_evaluation = renderer->environment()->evaluate(
            incomplete_sample.wi, swl, 0.0f, true);
        result.write(4u, make_float4(
                             complete_sample.eval.pdf,
                             complete_evaluation.pdf,
                             incomplete_sample.eval.pdf,
                             incomplete_evaluation.pdf));
    };
    auto shader = device.compile(kernel);
    std::array<float4, 5u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(result.data())
           << synchronize();

    auto total = std::accumulate(weights.begin(), weights.end(), 0.0);
    auto average = static_cast<float>(total / weights.size());
    auto compensated = weights;
    for (auto& weight : compensated)
    {
        weight = std::max(weight - average, 0.0f);
    }
    auto compensated_total = std::accumulate(compensated.begin(), compensated.end(), 0.0);
    if (compensated_total == 0.0)
    {
        compensated.fill(1.0f);
        compensated_total = compensated.size();
    }

    constexpr auto cell_count = 4.0;
    auto inv_four_pi = 0.25 / std::acos(-1.0);
    auto complete_integral = 0.0;
    auto incomplete_integral = 0.0;
    for (auto i = 0u; i < 4u; i++)
    {
        auto expected_complete = weights[i] / total * cell_count * inv_four_pi;
        auto expected_incomplete = compensated[i] / compensated_total * cell_count * inv_four_pi;
        if (!nearly_equal(result[i].x, static_cast<float>(expected_complete), 1e-5f) ||
            !nearly_equal(result[i].y, static_cast<float>(expected_incomplete), 1e-5f) ||
            !nearly_equal(result[i].z, 1.0f, 1e-3f))
        {
            return false;
        }
        complete_integral += result[i].x * std::acos(-1.0);
        incomplete_integral += result[i].y * std::acos(-1.0);
    }
    auto sampled = result[4u];
    return nearly_equal(static_cast<float>(complete_integral), 1.0f, 1e-5f) &&
           nearly_equal(static_cast<float>(incomplete_integral), 1.0f, 1e-5f) &&
           nearly_equal(sampled.x, sampled.y, 1e-5f) &&
           nearly_equal(sampled.z, sampled.w, 1e-5f) &&
           std::isfinite(sampled.x) && sampled.x > 0.0f &&
           std::isfinite(sampled.z) && sampled.z > 0.0f;
}

[[nodiscard]] bool test_environment_distributions(Device& device, Stream& stream)
{
    TemporaryDirectory directory;
    std::array hotspot{1.0f, 1.0f, 1.0f, 5.0f};
    std::array constant{2.0f, 2.0f, 2.0f, 2.0f};
    auto hotspot_scene = write_environment_scene(directory.path(), "hotspot", hotspot);
    auto constant_scene = write_environment_scene(directory.path(), "constant", constant);
    return test_environment_distribution_case(device, stream, hotspot_scene, hotspot) &&
           test_environment_distribution_case(device, stream, constant_scene, constant);
}

[[nodiscard]] luisa::unique_ptr<Scene> load_distant_scene()
{
    auto parsed = PbrtParser::parse("test/scenes/distant_basic.pbrt");
    auto spec   = PbrtImporter::import(std::move(parsed));
    auto scene  = Scene::create(spec);
    return dynamic_cast<const DistantEnvironment*>(scene->environment()) == nullptr
               ? nullptr
               : std::move(scene);
}

[[nodiscard]] bool test_distant_environment(
    Device& device, Stream& stream, const Scene& scene)
{
    auto renderer = Renderer::create(device, stream, scene);
    if (renderer->environment() == nullptr || !renderer->lights().empty())
    {
        return false;
    }

    auto output = device.create_buffer<float4>(4u);
    Kernel1D kernel = [&renderer](BufferFloat4 result) noexcept
    {
        auto swl       = renderer->spectrum()->sample(0.5f);
        auto sample_a  = renderer->environment()->sample(swl, 0.0f, make_float2(0.1f, 0.2f), false);
        auto sample_b  = renderer->environment()->sample(swl, 0.0f, make_float2(0.8f, 0.9f), true);
        auto evaluated_complete = renderer->environment()->evaluate(sample_a.wi, swl, 0.0f, false);
        auto evaluated_incomplete = renderer->environment()->evaluate(sample_a.wi, swl, 0.0f, true);
        result.write(0u, make_float4(sample_a.wi, sample_a.eval.pdf));
        result.write(1u, make_float4(sample_b.wi, sample_b.eval.pdf));
        result.write(2u, make_float4(
            ite(sample_a.delta, 1.0f, 0.0f),
            ite(sample_b.delta, 1.0f, 0.0f),
            evaluated_complete.pdf,
            sample_a.eval.L[0u]));
        result.write(3u, make_float4(
                             evaluated_complete.L[0u], evaluated_incomplete.L[0u],
                             evaluated_complete.pdf, evaluated_incomplete.pdf));
    };
    auto shader = device.compile(kernel);
    std::array<float4, 4u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(result.data())
           << synchronize();

    auto a = result[0u];
    auto b = result[1u];
    auto values = result[2u];
    return dot(a.xyz(), make_float3(1.0f, 0.0f, 0.0f)) > 0.999f &&
           dot(a.xyz(), b.xyz()) > 0.999f &&
           nearly_equal(a.w, 1.0f) && nearly_equal(b.w, 1.0f) &&
           nearly_equal(values.x, 1.0f) && nearly_equal(values.y, 1.0f) &&
           nearly_equal(values.z, 0.0f) && std::isfinite(values.w) && values.w > 0.0f &&
           nearly_equal(result[3u].x, result[3u].y) &&
           nearly_equal(result[3u].z, result[3u].w);
}

[[nodiscard]] luisa::unique_ptr<Scene> load_uniform_scene()
{
    auto parsed = PbrtParser::parse("test/scenes/infinite_uniform.pbrt");
    auto spec   = PbrtImporter::import(std::move(parsed));
    auto scene  = Scene::create(spec);
    return dynamic_cast<const UniformEnvironment*>(scene->environment()) == nullptr
               ? nullptr
               : std::move(scene);
}

[[nodiscard]] bool test_uniform_environment(
    Device& device, Stream& stream, const Scene& scene)
{
    auto renderer = Renderer::create(device, stream, scene);
    if (renderer->environment() == nullptr || !renderer->lights().empty())
    {
        return false;
    }

    auto output = device.create_buffer<float4>(4u);
    Kernel1D kernel = [&renderer](BufferFloat4 result) noexcept
    {
        auto swl = renderer->spectrum()->sample(0.5f);
        auto sample_complete = renderer->environment()->sample(
            swl, 0.0f, make_float2(0.1f, 0.2f), false);
        auto sample_incomplete = renderer->environment()->sample(
            swl, 0.0f, make_float2(0.8f, 0.9f), true);
        auto evaluated_complete = renderer->environment()->evaluate(
            make_float3(1.0f, 0.0f, 0.0f), swl, 0.0f, false);
        auto evaluated_incomplete = renderer->environment()->evaluate(
            make_float3(0.0f, 1.0f, 0.0f), swl, 0.0f, true);
        result.write(0u, make_float4(sample_complete.wi, sample_complete.eval.pdf));
        result.write(1u, make_float4(sample_incomplete.wi, sample_incomplete.eval.pdf));
        result.write(2u, make_float4(
                             ite(sample_complete.delta, 1.0f, 0.0f),
                             ite(sample_incomplete.delta, 1.0f, 0.0f),
                             evaluated_complete.pdf,
                             evaluated_incomplete.pdf));
        result.write(3u, make_float4(
                             sample_complete.eval.L[0u],
                             sample_incomplete.eval.L[0u],
                             evaluated_complete.L[0u],
                             evaluated_incomplete.L[0u]));
    };
    auto shader = device.compile(kernel);
    std::array<float4, 4u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(result.data())
           << synchronize();

    auto expected_pdf = 0.25f / std::acos(-1.0f);
    auto flags_and_pdfs = result[2u];
    auto radiance = result[3u];
    return nearly_equal(length(result[0u].xyz()), 1.0f, 1e-5f) &&
           nearly_equal(length(result[1u].xyz()), 1.0f, 1e-5f) &&
           nearly_equal(result[0u].w, expected_pdf, 1e-6f) &&
           nearly_equal(result[1u].w, expected_pdf, 1e-6f) &&
           nearly_equal(flags_and_pdfs.x, 0.0f) &&
           nearly_equal(flags_and_pdfs.y, 0.0f) &&
           nearly_equal(flags_and_pdfs.z, expected_pdf, 1e-6f) &&
           nearly_equal(flags_and_pdfs.w, expected_pdf, 1e-6f) &&
           std::isfinite(radiance.x) && radiance.x > 0.0f &&
           nearly_equal(radiance.x, radiance.y, 1e-6f) &&
           nearly_equal(radiance.x, radiance.z, 1e-6f) &&
           nearly_equal(radiance.x, radiance.w, 1e-6f);
}

[[nodiscard]] luisa::unique_ptr<Scene> load_grouped_scene()
{
    auto parsed = PbrtParser::parse("test/scenes/multiple_environment_lights.pbrt");
    auto spec   = PbrtImporter::import(std::move(parsed));
    auto scene  = Scene::create(spec);
    auto grouped = dynamic_cast<const GroupedEnvironment*>(scene->environment());
    return grouped == nullptr || grouped->environments().size() != 3u
               ? nullptr
               : std::move(scene);
}

[[nodiscard]] bool test_grouped_environment(
    Device& device, Stream& stream, const Scene& scene)
{
    auto renderer = Renderer::create(device, stream, scene);
    if (renderer->environment() == nullptr || !renderer->lights().empty())
    {
        return false;
    }

    auto output = device.create_buffer<float4>(5u);
    Kernel1D kernel = [&renderer](BufferFloat4 result) noexcept
    {
        auto swl = renderer->spectrum()->sample(0.5f);
        auto continuous = renderer->environment()->sample(
            swl, 0.0f, make_float2(0.1f, 0.2f), false);
        auto distant_a = renderer->environment()->sample(
            swl, 0.0f, make_float2(0.5f, 0.2f), false);
        auto distant_b = renderer->environment()->sample(
            swl, 0.0f, make_float2(0.9f, 0.2f), false);
        auto evaluation = renderer->environment()->evaluate(
            make_float3(0.0f, 0.0f, 1.0f), swl, 0.0f, false);
        result.write(0u, make_float4(
                             ite(continuous.delta, 1.0f, 0.0f),
                             continuous.eval.pdf,
                             continuous.eval.L[0u],
                             length(continuous.wi)));
        result.write(1u, make_float4(distant_a.wi, distant_a.eval.pdf));
        result.write(2u, make_float4(distant_b.wi, distant_b.eval.pdf));
        result.write(3u, make_float4(
                             ite(distant_a.delta, 1.0f, 0.0f),
                             ite(distant_b.delta, 1.0f, 0.0f),
                             evaluation.pdf,
                             evaluation.L[0u]));
        result.write(4u, make_float4(
                             continuous.eval.L[0u],
                             distant_a.eval.L[0u],
                             distant_b.eval.L[0u],
                             0.0f));
    };
    auto shader = device.compile(kernel);
    std::array<float4, 5u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(result.data())
           << synchronize();

    auto expected_continuous_pdf = 0.25f / (3.0f * std::acos(-1.0f));
    auto expected_delta_pdf      = 1.0f / 3.0f;
    return nearly_equal(result[0u].x, 0.0f) &&
           nearly_equal(result[0u].y, expected_continuous_pdf, 1e-6f) &&
           nearly_equal(result[0u].w, 1.0f, 1e-5f) &&
           dot(result[1u].xyz(), make_float3(1.0f, 0.0f, 0.0f)) > 0.999f &&
           dot(result[2u].xyz(), make_float3(0.0f, 1.0f, 0.0f)) > 0.999f &&
           nearly_equal(result[1u].w, expected_delta_pdf, 1e-6f) &&
           nearly_equal(result[2u].w, expected_delta_pdf, 1e-6f) &&
           nearly_equal(result[3u].x, 1.0f) &&
           nearly_equal(result[3u].y, 1.0f) &&
           nearly_equal(result[3u].z, expected_continuous_pdf, 1e-6f) &&
           nearly_equal(result[0u].z, result[3u].w, 1e-6f) &&
           nearly_equal(result[4u].x, result[3u].w, 1e-6f) &&
           result[4u].y > result[4u].z;
}

} // namespace

int main(int argc, char* argv[])
{
    if (!test_alias_tables() || !test_pfm_loading())
    {
        return 2;
    }
    if (argc < 2)
    {
        return 0;
    }

    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();
    if (!test_equal_area_mapping(device, stream))
    {
        return 4;
    }
    if (!test_environment_distributions(device, stream))
    {
        return 5;
    }
    auto distant_scene = load_distant_scene();
    if (distant_scene == nullptr)
    {
        return 7;
    }
    if (!test_distant_environment(device, stream, *distant_scene))
    {
        return 8;
    }
    auto uniform_scene = load_uniform_scene();
    if (uniform_scene == nullptr)
    {
        return 9;
    }
    if (!test_uniform_environment(device, stream, *uniform_scene))
    {
        return 10;
    }
    auto grouped_scene = load_grouped_scene();
    if (grouped_scene == nullptr)
    {
        return 11;
    }
    if (!test_grouped_environment(device, stream, *grouped_scene))
    {
        return 12;
    }
    return 0;
}
