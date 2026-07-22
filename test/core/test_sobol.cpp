#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

#include <luisa/luisa-compute.h>

#include "base/renderer.h"
#include "samplers/sobol.h"
#include "utils/command_buffer.h"
#include "utils/sobol_matrices.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{

struct SampleBatch
{
    uint2 resolution;
    uint spp;
    uint seed;
    std::vector<float4> first;
    std::vector<float> second;
};

[[nodiscard]] uint reverse_bits(uint v) noexcept
{
    v = ((v & 0x55555555u) << 1u) | ((v >> 1u) & 0x55555555u);
    v = ((v & 0x33333333u) << 2u) | ((v >> 2u) & 0x33333333u);
    v = ((v & 0x0f0f0f0fu) << 4u) | ((v >> 4u) & 0x0f0f0f0fu);
    v = ((v & 0x00ff00ffu) << 8u) | ((v >> 8u) & 0x00ff00ffu);
    return (v << 16u) | (v >> 16u);
}

[[nodiscard]] uint fast_owen_scramble(uint seed, uint v) noexcept
{
    v = reverse_bits(v);
    v ^= v * 0x3d20adeau;
    v += seed;
    v *= (seed >> 16u) | 1u;
    v ^= v * 0x05526c56u;
    v ^= v * 0x53a22864u;
    return reverse_bits(v);
}

[[nodiscard]] uint xxhash32(uint2 p) noexcept
{
    constexpr auto prime32_2 = 2246822519u;
    constexpr auto prime32_3 = 3266489917u;
    constexpr auto prime32_4 = 668265263u;
    constexpr auto prime32_5 = 374761393u;
    auto h                   = p.y + prime32_5 + p.x * prime32_3;
    h                        = prime32_4 * std::rotl(h, 17);
    h                        = prime32_2 * (h ^ (h >> 15u));
    h                        = prime32_3 * (h ^ (h >> 13u));
    return h ^ (h >> 16u);
}

[[nodiscard]] float uint_to_uniform_float(uint v) noexcept
{
    return std::min(0x1.fffffep-1f, static_cast<float>(v) * 0x1p-32f);
}

[[nodiscard]] float sobol_sample(uint64_t index, uint dimension, uint seed, bool scramble) noexcept
{
    auto value        = 0u;
    auto matrix_index = dimension * SobolMatrixSize;
    while (index != 0ull)
    {
        if ((index & 1ull) != 0ull)
        {
            value ^= SobolMatrices32[matrix_index];
        }
        index >>= 1ull;
        matrix_index++;
    }
    if (scramble)
    {
        value = fast_owen_scramble(seed, value);
    }
    return uint_to_uniform_float(value);
}

[[nodiscard]] uint64_t sobol_interval_to_index(uint m, uint frame, uint2 pixel) noexcept
{
    if (m == 0u)
    {
        return frame;
    }
    auto index           = static_cast<uint64_t>(frame) << (2u * m);
    auto delta           = 0ull;
    auto column          = 0u;
    auto remaining_frame = frame;
    while (remaining_frame != 0u)
    {
        if ((remaining_frame & 1u) != 0u)
        {
            delta ^= VdCSobolMatrices[m - 1u][column];
        }
        remaining_frame >>= 1u;
        column++;
    }
    auto b = delta ^ ((static_cast<uint64_t>(pixel.x) << m) | pixel.y);
    column = 0u;
    while (b != 0ull)
    {
        if ((b & 1ull) != 0ull)
        {
            index ^= VdCSobolMatricesInv[m - 1u][column];
        }
        b >>= 1ull;
        column++;
    }
    return index;
}

[[nodiscard]] bool nearly_equal(float a, float b) noexcept
{
    return std::abs(a - b) <= 1e-7f;
}

[[nodiscard]] SampleBatch generate_samples(
    Device& device, Stream& stream, uint2 resolution, uint spp, uint seed)
{
    Renderer renderer{device};
    SobolSampler sampler{spp, seed};
    auto instance = sampler.build(renderer);

    CommandBuffer command_buffer{stream};
    instance->reset(command_buffer, resolution, resolution.x * resolution.y);
    command_buffer << synchronize();

    auto count      = resolution.x * resolution.y * spp;
    auto first      = device.create_buffer<float4>(count);
    auto second     = device.create_buffer<float>(count);
    Kernel1D kernel = [&instance, resolution, spp](BufferFloat4 first, BufferFloat second) noexcept
    {
        auto i            = dispatch_id().x;
        auto sample_index = i % spp;
        auto pixel_index  = i / spp;
        auto pixel        = make_uint2(pixel_index % resolution.x, pixel_index / resolution.x);
        instance->start(pixel, sample_index);
        auto pixel_sample = instance->generate_pixel_2d();
        auto u1           = instance->generate_1d();
        auto u2           = instance->generate_2d();
        first.write(i, make_float4(pixel_sample, u1, u2.x));
        second.write(i, u2.y);
    };
    auto shader = device.compile(kernel);

    SampleBatch batch{
        .resolution = resolution,
        .spp        = spp,
        .seed       = seed,
        .first      = std::vector<float4>(count),
        .second     = std::vector<float>(count),
    };
    stream << shader(first, second).dispatch(count)
           << first.copy_to(batch.first.data())
           << second.copy_to(batch.second.data())
           << synchronize();
    return batch;
}

[[nodiscard]] bool validate_reference(const SampleBatch& batch) noexcept
{
    auto scale = next_pow2(std::max(batch.resolution.x, batch.resolution.y));
    auto m     = std::bit_width(scale) - 1u;
    for (auto i = 0u; i < batch.first.size(); i++)
    {
        auto sample_index  = i % batch.spp;
        auto pixel_index   = i / batch.spp;
        auto pixel         = make_uint2(pixel_index % batch.resolution.x, pixel_index / batch.resolution.x);
        auto index         = sobol_interval_to_index(m, sample_index, pixel);
        auto expected_x    = std::clamp(sobol_sample(index, 0u, 0u, false) * scale - pixel.x, 0.0f, 0x1.fffffep-1f);
        auto expected_y    = std::clamp(sobol_sample(index, 1u, 0u, false) * scale - pixel.y, 0.0f, 0x1.fffffep-1f);
        auto expected_1d   = sobol_sample(index, 2u, xxhash32(make_uint2(2u, batch.seed)), true);
        auto expected_2d_x = sobol_sample(index, 3u, xxhash32(make_uint2(3u, batch.seed)), true);
        auto expected_2d_y = sobol_sample(index, 4u, xxhash32(make_uint2(4u, batch.seed)), true);
        auto actual        = batch.first[i];
        if (!nearly_equal(actual.x, expected_x) ||
            !nearly_equal(actual.y, expected_y) ||
            !nearly_equal(actual.z, expected_1d) ||
            !nearly_equal(actual.w, expected_2d_x) ||
            !nearly_equal(batch.second[i], expected_2d_y))
        {
            return false;
        }
        if (actual.x < 0.0f || actual.x >= 1.0f ||
            actual.y < 0.0f || actual.y >= 1.0f)
        {
            return false;
        }
        auto global = (make_float2(pixel) + actual.xy()) / static_cast<float>(scale);
        if (static_cast<uint>(global.x * scale) != pixel.x ||
            static_cast<uint>(global.y * scale) != pixel.y)
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool validate_elementary_intervals(const SampleBatch& batch) noexcept
{
    if (batch.resolution.x != 1u || batch.resolution.y != 1u || batch.spp != 16u)
    {
        return false;
    }
    std::array<bool, 16u> occupied{};
    for (auto sample : batch.first)
    {
        auto x    = std::min(static_cast<uint>(sample.x * 4.0f), 3u);
        auto y    = std::min(static_cast<uint>(sample.y * 4.0f), 3u);
        auto cell = y * 4u + x;
        if (occupied[cell])
        {
            return false;
        }
        occupied[cell] = true;
    }
    return std::all_of(occupied.cbegin(), occupied.cend(), [](bool value) noexcept
    {
        return value;
    });
}

[[nodiscard]] bool validate_reproducibility(
    const SampleBatch& a, const SampleBatch& b, const SampleBatch& different_seed) noexcept
{
    if (a.first.size() != b.first.size() || a.first.size() != different_seed.first.size())
    {
        return false;
    }
    auto scrambled_dimension_changed = false;
    for (auto i = 0u; i < a.first.size(); i++)
    {
        if (a.first[i].x != b.first[i].x ||
            a.first[i].y != b.first[i].y ||
            a.first[i].z != b.first[i].z ||
            a.first[i].w != b.first[i].w ||
            a.second[i] != b.second[i])
        {
            return false;
        }
        if (a.first[i].x != different_seed.first[i].x ||
            a.first[i].y != different_seed.first[i].y)
        {
            return false;
        }
        scrambled_dimension_changed |= a.first[i].z != different_seed.first[i].z ||
                                       a.first[i].w != different_seed.first[i].w ||
                                       a.second[i] != different_seed.second[i];
    }
    return scrambled_dimension_changed;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        return 1;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    auto unit_pixel = generate_samples(device, stream, make_uint2(1u), 16u, 42u);
    if (!validate_reference(unit_pixel) || !validate_elementary_intervals(unit_pixel))
    {
        return 2;
    }

    auto non_power_of_two_resolution = generate_samples(device, stream, make_uint2(4u, 3u), 5u, 42u);
    auto repeated                    = generate_samples(device, stream, make_uint2(4u, 3u), 5u, 42u);
    auto different_seed              = generate_samples(device, stream, make_uint2(4u, 3u), 5u, 43u);
    if (!validate_reference(non_power_of_two_resolution))
    {
        return 3;
    }
    if (!validate_reference(different_seed))
    {
        return 4;
    }
    if (!validate_reproducibility(non_power_of_two_resolution, repeated, different_seed))
    {
        return 5;
    }
    auto large_seed = generate_samples(device, stream, make_uint2(1u), 4096u, 20120712u);
    if (!validate_reference(large_seed))
    {
        return 6;
    }
    return 0;
}
