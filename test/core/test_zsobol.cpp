// Tests for the ZSobol sampler.
// This test covers CPU/GPU agreement, elementary intervals, valid indices,
// deterministic scrambling, and dimension wraparound.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include <luisa/luisa-compute.h>

#include "base/renderer.h"
#include "samplers/zsobol.h"
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

constexpr std::array<std::array<uint, 4u>, 24u> permutations{{
    {0u, 1u, 2u, 3u}, {0u, 1u, 3u, 2u}, {0u, 2u, 1u, 3u}, {0u, 2u, 3u, 1u},
    {0u, 3u, 2u, 1u}, {0u, 3u, 1u, 2u}, {1u, 0u, 2u, 3u}, {1u, 0u, 3u, 2u},
    {1u, 2u, 0u, 3u}, {1u, 2u, 3u, 0u}, {1u, 3u, 2u, 0u}, {1u, 3u, 0u, 2u},
    {2u, 1u, 0u, 3u}, {2u, 1u, 3u, 0u}, {2u, 0u, 1u, 3u}, {2u, 0u, 3u, 1u},
    {2u, 3u, 0u, 1u}, {2u, 3u, 1u, 0u}, {3u, 1u, 2u, 0u}, {3u, 1u, 0u, 2u},
    {3u, 2u, 1u, 0u}, {3u, 2u, 0u, 1u}, {3u, 0u, 2u, 1u}, {3u, 0u, 1u, 2u},
}};

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

[[nodiscard]] uint xxhash32(uint dimension, uint seed) noexcept
{
    constexpr auto prime32_2 = 2246822519u;
    constexpr auto prime32_3 = 3266489917u;
    constexpr auto prime32_4 = 668265263u;
    constexpr auto prime32_5 = 374761393u;
    auto h                    = seed + prime32_5 + dimension * prime32_3;
    h                         = prime32_4 * std::rotl(h, 17);
    h                         = prime32_2 * (h ^ (h >> 15u));
    h                         = prime32_3 * (h ^ (h >> 13u));
    return h ^ (h >> 16u);
}

[[nodiscard]] uint64_t mix_bits(uint64_t v) noexcept
{
    v ^= v >> 31u;
    v *= 0x7fb5d329728ea185ull;
    v ^= v >> 27u;
    v *= 0x81dadef4bc2dd44dull;
    v ^= v >> 33u;
    return v;
}

[[nodiscard]] uint64_t spread_bits(uint64_t v) noexcept
{
    v = (v ^ (v << 16u)) & 0x0000ffff0000ffffull;
    v = (v ^ (v << 8u)) & 0x00ff00ff00ff00ffull;
    v = (v ^ (v << 4u)) & 0x0f0f0f0f0f0f0f0full;
    v = (v ^ (v << 2u)) & 0x3333333333333333ull;
    v = (v ^ (v << 1u)) & 0x5555555555555555ull;
    return v;
}

[[nodiscard]] uint64_t encode_morton(uint2 pixel) noexcept
{
    return (spread_bits(pixel.y) << 1u) | spread_bits(pixel.x);
}

[[nodiscard]] uint64_t sample_index(
    uint2 resolution, uint spp, uint2 pixel, uint sample, uint dimension) noexcept
{
    auto log2_spp            = std::countr_zero(spp);
    auto log2_resolution     = std::bit_width(std::max(resolution.x, resolution.y) - 1u);
    auto n_base4_digits      = log2_resolution + (log2_spp + 1u) / 2u;
    auto odd_log2_spp        = (log2_spp & 1u) != 0u;
    auto morton_index        = (encode_morton(pixel) << log2_spp) | sample;
    auto result              = 0ull;
    auto last_digit          = odd_log2_spp ? 1 : 0;
    for (auto i = static_cast<int>(n_base4_digits) - 1; i >= last_digit; i--)
    {
        auto shift       = static_cast<uint>(2 * i - (odd_log2_spp ? 1 : 0));
        auto digit       = static_cast<uint>((morton_index >> shift) & 3ull);
        auto higher      = morton_index >> (shift + 2u);
        auto permutation = static_cast<uint>((mix_bits(higher ^ (uint64_t{dimension} * 0x55555555u)) >> 24u) % 24u);
        result           |= static_cast<uint64_t>(permutations[permutation][digit]) << shift;
    }
    if (odd_log2_spp)
    {
        auto digit = static_cast<uint>(morton_index & 1ull);
        result |= digit ^ (mix_bits((morton_index >> 1u) ^ (uint64_t{dimension} * 0x55555555u)) & 1ull);
    }
    return result;
}

[[nodiscard]] float sobol_sample(uint64_t index, uint dimension, uint hash) noexcept
{
    auto value        = 0u;
    auto matrix_index = dimension * SobolMatrixSize;
    while (index != 0ull)
    {
        if ((index & 1ull) != 0ull)
        {
            value ^= SobolMatrices32[matrix_index];
        }
        index >>= 1u;
        matrix_index++;
    }
    value = fast_owen_scramble(hash, value);
    return std::min(0x1.fffffep-1f, static_cast<float>(value) * 0x1p-32f);
}

[[nodiscard]] bool nearly_equal(float a, float b) noexcept
{
    return std::abs(a - b) <= 1e-7f;
}

[[nodiscard]] SampleBatch generate_samples(
    Device& device, Stream& stream, uint2 resolution, uint spp, uint seed)
{
    Renderer renderer{device};
    ZSobolSampler sampler{spp, seed};
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
        auto sample       = i % spp;
        auto pixel_index  = i / spp;
        auto pixel        = make_uint2(pixel_index % resolution.x, pixel_index / resolution.x);
        instance->start(pixel, sample);
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
    for (auto i = 0u; i < batch.first.size(); i++)
    {
        auto sample      = i % batch.spp;
        auto pixel_index = i / batch.spp;
        auto pixel       = make_uint2(pixel_index % batch.resolution.x, pixel_index / batch.resolution.x);
        auto pixel_index_z = sample_index(batch.resolution, batch.spp, pixel, sample, 0u);
        auto u1_index      = sample_index(batch.resolution, batch.spp, pixel, sample, 2u);
        auto u2_index      = sample_index(batch.resolution, batch.spp, pixel, sample, 3u);
        auto expected      = make_float4(
            sobol_sample(pixel_index_z, 0u, xxhash32(0u, batch.seed)),
            sobol_sample(pixel_index_z, 1u, xxhash32(1u, batch.seed)),
            sobol_sample(u1_index, 0u, xxhash32(2u, batch.seed)),
            sobol_sample(u2_index, 0u, xxhash32(3u, batch.seed)));
        auto expected_second = sobol_sample(u2_index, 1u, xxhash32(4u, batch.seed));
        auto actual          = batch.first[i];
        if (!nearly_equal(actual.x, expected.x) ||
            !nearly_equal(actual.y, expected.y) ||
            !nearly_equal(actual.z, expected.z) ||
            !nearly_equal(actual.w, expected.w) ||
            !nearly_equal(batch.second[i], expected_second))
        {
            return false;
        }
        if (actual.x < 0.0f || actual.x >= 1.0f ||
            actual.y < 0.0f || actual.y >= 1.0f ||
            actual.z < 0.0f || actual.z >= 1.0f ||
            actual.w < 0.0f || actual.w >= 1.0f ||
            batch.second[i] < 0.0f || batch.second[i] >= 1.0f)
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool validate_elementary_intervals(const SampleBatch& batch) noexcept
{
    auto log2_spp = std::countr_zero(batch.spp);
    auto pixel_count = batch.resolution.x * batch.resolution.y;
    for (auto pixel = 0u; pixel < pixel_count; pixel++)
    {
        for (auto x_bits = 0u; x_bits <= log2_spp; x_bits++)
        {
            auto nx = 1u << x_bits;
            auto ny = 1u << (log2_spp - x_bits);
            std::vector<bool> occupied(batch.spp);
            for (auto sample = 0u; sample < batch.spp; sample++)
            {
                auto p    = batch.first[pixel * batch.spp + sample];
                auto x    = std::min(static_cast<uint>(p.x * nx), nx - 1u);
                auto y    = std::min(static_cast<uint>(p.y * ny), ny - 1u);
                auto cell = y * nx + x;
                if (occupied[cell])
                {
                    return false;
                }
                occupied[cell] = true;
            }
        }
    }
    return true;
}

[[nodiscard]] bool validate_indices() noexcept
{
    constexpr auto resolution = uint2{16u, 9u};
    for (auto log2_spp = 0u; log2_spp <= 8u; log2_spp++)
    {
        auto spp = 1u << log2_spp;
        for (auto dimension : {0u, 3u, 6u})
        {
            std::set<uint64_t> indices;
            for (auto y = 0u; y < resolution.y; y++)
            {
                for (auto x = 0u; x < resolution.x; x++)
                {
                    auto pixel = make_uint2(x, y);
                    auto base  = sample_index(resolution, spp, pixel, 0u, dimension) / spp;
                    for (auto sample = 0u; sample < spp; sample++)
                    {
                        auto index = sample_index(resolution, spp, pixel, sample, dimension);
                        if (!indices.emplace(index).second || index / spp != base)
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

[[nodiscard]] bool validate_reproducibility(
    const SampleBatch& a, const SampleBatch& b, const SampleBatch& different_seed) noexcept
{
    auto changed = false;
    for (auto i = 0u; i < a.first.size(); i++)
    {
        if (luisa::any(a.first[i] != b.first[i]) || a.second[i] != b.second[i])
        {
            return false;
        }
        changed |= luisa::any(a.first[i] != different_seed.first[i]) ||
                   a.second[i] != different_seed.second[i];
    }
    return changed;
}

[[nodiscard]] bool validate_dimension_wrap(Device& device, Stream& stream)
{
    Renderer renderer{device};
    ZSobolSampler sampler{4u, 42u};
    auto instance = sampler.build(renderer);
    CommandBuffer command_buffer{stream};
    instance->reset(command_buffer, make_uint2(4u, 3u), 1u);
    command_buffer << synchronize();

    auto output    = device.create_buffer<float4>(2u);
    Kernel1D kernel = [&instance](BufferFloat4 output) noexcept
    {
        auto pixel = make_uint2(2u, 1u);
        instance->start(pixel, 1u);
        $for(i, 1023u)
        {
            static_cast<void>(i);
            static_cast<void>(instance->generate_1d());
        };
        auto wrapped_2d = instance->generate_2d();
        instance->start(pixel, 1u);
        auto fresh_2d = instance->generate_2d();
        output.write(0u, make_float4(wrapped_2d, fresh_2d));

        instance->start(pixel, 1u);
        $for(i, 1024u)
        {
            static_cast<void>(i);
            static_cast<void>(instance->generate_1d());
        };
        auto wrapped_1d = instance->generate_1d();
        instance->start(pixel, 1u);
        auto fresh_1d = instance->generate_1d();
        output.write(1u, make_float4(wrapped_1d, fresh_1d, 0.0f, 0.0f));
    };
    auto shader = device.compile(kernel);
    std::array<float4, 2u> values{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(values.data())
           << synchronize();
    return nearly_equal(values[0].x, values[0].z) &&
           nearly_equal(values[0].y, values[0].w) &&
           nearly_equal(values[1].x, values[1].y);
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

    for (auto test_case : std::array{
             std::pair{make_uint2(1u), 1u},
             std::pair{make_uint2(3u, 5u), 2u},
             std::pair{make_uint2(4u, 3u), 4u},
             std::pair{make_uint2(8u, 5u), 16u},
         })
    {
        auto batch = generate_samples(device, stream, test_case.first, test_case.second, 42u);
        if (!validate_reference(batch) || !validate_elementary_intervals(batch))
        {
            return 2;
        }
    }

    auto original       = generate_samples(device, stream, make_uint2(4u, 3u), 16u, 42u);
    auto repeated       = generate_samples(device, stream, make_uint2(4u, 3u), 16u, 42u);
    auto different_seed = generate_samples(device, stream, make_uint2(4u, 3u), 16u, 43u);
    if (!validate_reproducibility(original, repeated, different_seed))
    {
        return 3;
    }
    if (!validate_indices())
    {
        return 4;
    }
    if (!validate_dimension_wrap(device, stream))
    {
        return 5;
    }
    return 0;
}
