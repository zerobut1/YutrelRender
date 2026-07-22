#include "zsobol.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

#include <luisa/dsl/sugar.h>

#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/rng.h"
#include "utils/sobol_matrices.h"

namespace Yutrel
{
namespace
{
[[nodiscard]] uint xxhash32_host(uint dimension, uint seed) noexcept
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
} // namespace

ULong ZSobolSampler::Instance::_encode_morton(Expr<uint2> pixel) noexcept
{
    static Callable impl = [](UInt2 pixel) noexcept
    {
        static constexpr auto spread_bits = [](ULong v) noexcept
        {
            v = (v ^ (v << 16ull)) & 0x0000ffff0000ffffull;
            v = (v ^ (v << 8ull)) & 0x00ff00ff00ff00ffull;
            v = (v ^ (v << 4ull)) & 0x0f0f0f0f0f0f0f0full;
            v = (v ^ (v << 2ull)) & 0x3333333333333333ull;
            v = (v ^ (v << 1ull)) & 0x5555555555555555ull;
            return v;
        };
        return (spread_bits(cast<ulong>(pixel.y)) << 1ull) |
               spread_bits(cast<ulong>(pixel.x));
    };
    return impl(pixel);
}

UInt ZSobolSampler::Instance::_fast_owen_scramble(Expr<uint> seed, UInt v) noexcept
{
    v = reverse(v);
    v ^= v * 0x3d20adeau;
    v += seed;
    v *= (seed >> 16u) | 1u;
    v ^= v * 0x05526c56u;
    v ^= v * 0x53a22864u;
    return reverse(v);
}

ULong ZSobolSampler::Instance::_get_sample_index(Expr<uint> dimension) const noexcept
{
    LUISA_ASSERT(_get_sample_index_impl != nullptr, "ZSobol sampler must be reset before sampling.");
    return (*_get_sample_index_impl)(*_morton_index, dimension);
}

Float ZSobolSampler::Instance::_sobol_sample(ULong index, uint dimension, Expr<uint> hash) const noexcept
{
    static Callable impl = [](ULong index, UInt dimension, UInt hash, BufferVar<uint> matrices) noexcept
    {
        UInt value        = 0u;
        UInt matrix_index = dimension * SobolMatrixSize;
        $while(index != 0ull)
        {
            value = ite((index & 1ull) != 0ull, value ^ matrices.read(matrix_index), value);
            index >>= 1ull;
            matrix_index += 1u;
        };
        return uniform_uint_to_float(_fast_owen_scramble(hash, value));
    };
    return impl(index, dimension, hash, _sobol_matrices.view());
}

ZSobolSampler::Instance::Instance(const Renderer& renderer, const ZSobolSampler* sampler) noexcept
    : Sampler::Instance{renderer, sampler},
      _sobol_matrices{renderer.device().create_buffer<uint>(SobolMatrixSize * 2u)}
{
    std::array<uint2, MaxDimension> sample_hash{};
    for (auto dimension = 0u; dimension < MaxDimension; dimension++)
    {
        sample_hash[dimension] = make_uint2(
            xxhash32_host(dimension, sampler->seed()),
            xxhash32_host(dimension + 1u, sampler->seed()));
    }
    _sample_hash = luisa::make_unique<Constant<uint2>>(sample_hash);
}

void ZSobolSampler::Instance::reset(CommandBuffer& command_buffer, uint2 resolution, uint state_count) noexcept
{
    static_cast<void>(state_count);
    LUISA_ASSERT(resolution.x != 0u && resolution.y != 0u, "ZSobol sampler resolution must be non-zero.");

    auto spp = base<ZSobolSampler>()->spp();
    LUISA_ASSERT(spp != 0u, "ZSobol sampler requires at least one sample per pixel.");
    LUISA_ASSERT(std::has_single_bit(spp),
                 "ZSobol sampler requires a power-of-two sample count, got {}.",
                 spp);

    _log2_spp            = std::countr_zero(spp);
    auto max_resolution  = std::max(resolution.x, resolution.y);
    auto log2_resolution = std::bit_width(max_resolution - 1u);
    auto index_bits      = 2u * log2_resolution + _log2_spp;
    LUISA_ASSERT(index_bits <= SobolMatrixSize,
                 "ZSobol sampler index requires {} bits at resolution {}x{} and {} spp; at most {} are supported.",
                 index_bits,
                 resolution.x,
                 resolution.y,
                 spp,
                 SobolMatrixSize);
    _n_base4_digits = log2_resolution + (_log2_spp + 1u) / 2u;

    if (!_matrices_uploaded)
    {
        command_buffer << _sobol_matrices.copy_from(luisa::span{SobolMatrices32, SobolMatrixSize * 2u})
                       << commit();
        _matrices_uploaded = true;
    }

    auto n_base4_digits = _n_base4_digits;
    auto odd_log2_spp   = (_log2_spp & 1u) != 0u;
    _get_sample_index_impl = luisa::make_unique<Callable<ulong(ulong, uint)>>(
        [n_base4_digits, odd_log2_spp](ULong morton_index, UInt dimension) noexcept
        {
            static Constant<uint4> permutations{std::array{
                make_uint4(0u, 1u, 2u, 3u), make_uint4(0u, 1u, 3u, 2u), make_uint4(0u, 2u, 1u, 3u), make_uint4(0u, 2u, 3u, 1u),
                make_uint4(0u, 3u, 2u, 1u), make_uint4(0u, 3u, 1u, 2u), make_uint4(1u, 0u, 2u, 3u), make_uint4(1u, 0u, 3u, 2u),
                make_uint4(1u, 2u, 0u, 3u), make_uint4(1u, 2u, 3u, 0u), make_uint4(1u, 3u, 2u, 0u), make_uint4(1u, 3u, 0u, 2u),
                make_uint4(2u, 1u, 0u, 3u), make_uint4(2u, 1u, 3u, 0u), make_uint4(2u, 0u, 1u, 3u), make_uint4(2u, 0u, 3u, 1u),
                make_uint4(2u, 3u, 0u, 1u), make_uint4(2u, 3u, 1u, 0u), make_uint4(3u, 1u, 2u, 0u), make_uint4(3u, 1u, 0u, 2u),
                make_uint4(3u, 2u, 1u, 0u), make_uint4(3u, 2u, 0u, 1u), make_uint4(3u, 0u, 2u, 1u), make_uint4(3u, 0u, 1u, 2u)}};
            static constexpr auto mix_bits = [](ULong v) noexcept
            {
                v ^= v >> 31u;
                v *= 0x7fb5d329728ea185ull;
                v ^= v >> 27u;
                v *= 0x81dadef4bc2dd44dull;
                v ^= v >> 33u;
                return v;
            };

            ULong sample_index = 0ull;
            auto last_digit    = odd_log2_spp ? 1 : 0;
            for (auto i = static_cast<int>(n_base4_digits) - 1; i >= last_digit; i--)
            {
                auto digit_shift  = static_cast<uint>(2 * i - (odd_log2_spp ? 1 : 0));
                auto digit        = cast<uint>((morton_index >> digit_shift) & 3ull);
                auto higher       = morton_index >> (digit_shift + 2u);
                auto permutation  = cast<uint>((mix_bits(higher ^ cast<ulong>(dimension * 0x55555555u)) >> 24u) % 24ull);
                auto permuted     = permutations.read(permutation)[digit];
                sample_index     |= cast<ulong>(permuted) << digit_shift;
            }
            if (odd_log2_spp)
            {
                auto digit       = cast<uint>(morton_index & 1ull);
                auto permutation = cast<uint>(mix_bits((morton_index >> 1u) ^ cast<ulong>(dimension * 0x55555555u))) & 1u;
                sample_index    |= cast<ulong>(digit ^ permutation);
            }
            return sample_index;
        });
}

void ZSobolSampler::Instance::start(UInt2 pixel, UInt index) noexcept
{
    _dimension.emplace(def(0u));
    _morton_index.emplace((_encode_morton(pixel) << _log2_spp) | cast<ulong>(index));
}

Float ZSobolSampler::Instance::generate_1d() noexcept
{
    *_dimension   = ite(*_dimension >= MaxDimension, 0u, *_dimension);
    UInt dimension = *_dimension;
    auto index     = _get_sample_index(dimension);
    auto hash      = _sample_hash->read(dimension).x;
    *_dimension    = dimension + 1u;
    return _sobol_sample(index, 0u, hash);
}

Float2 ZSobolSampler::Instance::generate_2d() noexcept
{
    *_dimension    = ite(*_dimension + 1u >= MaxDimension, 0u, *_dimension);
    UInt dimension = *_dimension;
    auto index      = _get_sample_index(dimension);
    auto hash       = _sample_hash->read(dimension);
    *_dimension     = dimension + 2u;
    return make_float2(
        _sobol_sample(index, 0u, hash.x),
        _sobol_sample(index, 1u, hash.y));
}

Float2 ZSobolSampler::Instance::generate_pixel_2d() noexcept
{
    return generate_2d();
}

luisa::unique_ptr<Sampler::Instance> ZSobolSampler::build(const Renderer& renderer) const noexcept
{
    return luisa::make_unique<Instance>(renderer, this);
}

luisa::optional<luisa::string> ZSobolSamplerSpec::validate() const noexcept
{
    if (_spp == 0u)
    {
        return spec_validation_error("Sampler SPP must be greater than zero.");
    }
    if (!std::has_single_bit(_spp))
    {
        return spec_validation_error("ZSobol sampler SPP must be a power of two.");
    }
    return luisa::nullopt;
}

const Sampler* ZSobolSamplerSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Sampler, ZSobolSampler>(_spp, _seed);
}
} // namespace Yutrel
