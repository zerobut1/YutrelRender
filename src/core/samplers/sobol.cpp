#include "sobol.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>

#include <luisa/dsl/sugar.h>

#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/rng.h"
#include "utils/sobol_matrices.h"

namespace Yutrel
{
UInt SobolSampler::Instance::_fast_owen_scramble(Expr<uint> seed, UInt v) noexcept
{
    v = reverse(v);
    v ^= v * 0x3d20adeau;
    v += seed;
    v *= (seed >> 16u) | 1u;
    v ^= v * 0x05526c56u;
    v ^= v * 0x53a22864u;
    return reverse(v);
}

template <bool Scramble>
Float SobolSampler::Instance::_sobol_sample(ULong index, Expr<uint> dimension, Expr<uint> hash) const noexcept
{
    static Callable impl = [](ULong index, UInt dimension, BufferVar<uint> matrices, UInt hash) noexcept
    {
        UInt value        = 0u;
        UInt matrix_index = dimension * SobolMatrixSize;
        $while(index != 0ull)
        {
            value = ite((index & 1ull) != 0ull, value ^ matrices.read(matrix_index), value);
            index >>= 1ull;
            matrix_index += 1u;
        };
        if constexpr (Scramble)
        {
            value = _fast_owen_scramble(hash, value);
        }
        return uniform_uint_to_float(value);
    };
    return impl(index, dimension, _sobol_matrices.view(), hash);
}

ULong SobolSampler::Instance::_sobol_interval_to_index(uint m, UInt frame, Expr<uint2> pixel) const noexcept
{
    if (m == 0u)
    {
        return cast<ulong>(frame);
    }
    static Callable impl = [](UInt m, UInt frame, UInt2 pixel, BufferVar<ulong> vdc, BufferVar<ulong> vdc_inv) noexcept
    {
        UInt column = 0u;
        UInt m2     = m << 1u;
        ULong index = cast<ulong>(frame) << cast<ulong>(m2);
        ULong delta = 0ull;
        $while(frame != 0u)
        {
            $if((frame & 1u) != 0u)
            {
                delta ^= vdc.read(column);
            };
            frame >>= 1u;
            column += 1u;
        };

        ULong b = delta ^ ((cast<ulong>(pixel.x) << m) | cast<ulong>(pixel.y));
        column  = 0u;
        $while(b != 0ull)
        {
            $if((b & 1ull) != 0ull)
            {
                index ^= vdc_inv.read(column);
            };
            b >>= 1ull;
            column += 1u;
        };
        return index;
    };
    return impl(m, frame, pixel, _vdc_sobol_matrices.view(), _vdc_sobol_matrices_inv.view());
}

SobolSampler::Instance::Instance(const Renderer& renderer, const SobolSampler* sampler) noexcept
    : Sampler::Instance{renderer, sampler},
      _sobol_matrices{renderer.device().create_buffer<uint>(SobolMatrixSize * NSobolDimensions)},
      _vdc_sobol_matrices{renderer.device().create_buffer<ulong>(SobolMatrixSize)},
      _vdc_sobol_matrices_inv{renderer.device().create_buffer<ulong>(SobolMatrixSize)}
{
}

void SobolSampler::Instance::reset(CommandBuffer& command_buffer, uint2 resolution, uint state_count) noexcept
{
    static_cast<void>(state_count);
    LUISA_ASSERT(resolution.x != 0u && resolution.y != 0u, "Sobol sampler resolution must be non-zero.");

    _resolution = resolution;
    _scale      = next_pow2(std::max(resolution.x, resolution.y));
    LUISA_ASSERT(_scale != 0u, "Sobol sampler resolution is too large.");
    _log2_scale = std::bit_width(_scale) - 1u;
    LUISA_ASSERT(_log2_scale <= VdCSobolMatrixSize,
                 "Sobol sampler scale 2^{} exceeds the supported 2^{}.",
                 _log2_scale,
                 VdCSobolMatrixSize);

    auto spp = base<SobolSampler>()->spp();
    LUISA_ASSERT(spp != 0u, "Sobol sampler requires at least one sample per pixel.");
    LUISA_ASSERT(2u * _log2_scale + std::bit_width(spp - 1u) <= SobolMatrixSize,
                 "Sobol sampler index requires more than {} bits at resolution {}x{} and {} spp.",
                 SobolMatrixSize,
                 resolution.x,
                 resolution.y,
                 spp);

    if (!std::has_single_bit(spp))
    {
        static std::atomic_flag warned;
        if (!warned.test_and_set(std::memory_order_relaxed))
        {
            LUISA_WARNING_WITH_LOCATION("Non power-of-two samples per pixel ({}) are suboptimal for Sobol sampling.", spp);
        }
    }

    if (!_matrices_uploaded)
    {
        // Keep the large direction-number upload separate from the VdC uploads below.
        // Combining them in one command list leaves the first VdC entry zero on Metal,
        // which breaks Sobol interval-to-pixel mapping for all m > 0.
        command_buffer << _sobol_matrices.copy_from(SobolMatrices32)
                       << commit();
        _matrices_uploaded = true;
    }

    if (_log2_scale > 0u && (!_uploaded_vdc_m || *_uploaded_vdc_m != _log2_scale))
    {
        std::array<ulong, SobolMatrixSize> vdc{};
        std::array<ulong, SobolMatrixSize> vdc_inv{};
        for (auto i = 0u; i < SobolMatrixSize; i++)
        {
            vdc[i]     = VdCSobolMatrices[_log2_scale - 1u][i];
            vdc_inv[i] = VdCSobolMatricesInv[_log2_scale - 1u][i];
        }
        command_buffer << _vdc_sobol_matrices.copy_from(vdc.data())
                       << _vdc_sobol_matrices_inv.copy_from(vdc_inv.data())
                       << commit();
        _uploaded_vdc_m.emplace(_log2_scale);
    }
}

void SobolSampler::Instance::start(UInt2 pixel, UInt index) noexcept
{
    _pixel.emplace(pixel);
    _dimension.emplace(2u);
    _sobol_index.emplace(_sobol_interval_to_index(_log2_scale, index, pixel));
}

Float SobolSampler::Instance::generate_1d() noexcept
{
    *_dimension = ite(*_dimension >= NSobolDimensions, 2u, *_dimension);
    auto hash   = xxhash32(make_uint2(*_dimension, base<SobolSampler>()->seed()));
    auto u      = _sobol_sample<true>(*_sobol_index, *_dimension, hash);
    *_dimension += 1u;
    return u;
}

Float2 SobolSampler::Instance::generate_2d() noexcept
{
    *_dimension = ite(*_dimension + 1u >= NSobolDimensions, 2u, *_dimension);
    auto hash_x = xxhash32(make_uint2(*_dimension, base<SobolSampler>()->seed()));
    auto hash_y = xxhash32(make_uint2(*_dimension + 1u, base<SobolSampler>()->seed()));
    auto u      = make_float2(
        _sobol_sample<true>(*_sobol_index, *_dimension, hash_x),
        _sobol_sample<true>(*_sobol_index, *_dimension + 1u, hash_y));
    *_dimension += 2u;
    return u;
}

Float2 SobolSampler::Instance::generate_pixel_2d() noexcept
{
    auto global = make_float2(
        _sobol_sample<false>(*_sobol_index, 0u, 0u),
        _sobol_sample<false>(*_sobol_index, 1u, 0u));
    return clamp(
        global * static_cast<float>(_scale) - make_float2(*_pixel),
        0.0f,
        one_minus_epsilon);
}

luisa::unique_ptr<Sampler::Instance> SobolSampler::build(const Renderer& renderer) const noexcept
{
    return luisa::make_unique<Instance>(renderer, this);
}

luisa::optional<luisa::string> SobolSamplerSpec::validate() const noexcept
{
    return _spp == 0u ? spec_validation_error("Sampler SPP must be greater than zero.") : luisa::nullopt;
}

const Sampler* SobolSamplerSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Sampler, SobolSampler>(_spp, _seed);
}
} // namespace Yutrel
