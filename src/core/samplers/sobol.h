#pragma once

#include <luisa/core/stl/optional.h>
#include <luisa/runtime/buffer.h>

#include "base/sampler.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class SobolSampler final : public Sampler
{
public:
    class Instance final : public Sampler::Instance
    {
    private:
        uint2 _resolution{};
        uint _scale{};
        uint _log2_scale{};
        bool _matrices_uploaded{};
        luisa::optional<uint> _uploaded_vdc_m;
        luisa::optional<UInt2> _pixel;
        luisa::optional<UInt> _dimension;
        luisa::optional<ULong> _sobol_index;
        Buffer<uint> _sobol_matrices;
        Buffer<ulong> _vdc_sobol_matrices;
        Buffer<ulong> _vdc_sobol_matrices_inv;

    private:
        [[nodiscard]] static UInt _fast_owen_scramble(Expr<uint> seed, UInt v) noexcept;

        template <bool Scramble>
        [[nodiscard]] Float _sobol_sample(ULong index, Expr<uint> dimension, Expr<uint> hash) const noexcept;

        [[nodiscard]] ULong _sobol_interval_to_index(uint m, UInt frame, Expr<uint2> pixel) const noexcept;

    public:
        Instance(const Renderer& renderer, const SobolSampler* sampler) noexcept;

        void reset(CommandBuffer& command_buffer, uint2 resolution, uint state_count) noexcept override;
        void start(UInt2 pixel, UInt index) noexcept override;
        [[nodiscard]] Float generate_1d() noexcept override;
        [[nodiscard]] Float2 generate_2d() noexcept override;
        [[nodiscard]] Float2 generate_pixel_2d() noexcept override;
    };

private:
    uint _seed;

public:
    SobolSampler(uint spp, uint seed) noexcept
        : Sampler{spp}, _seed{seed} {}

    [[nodiscard]] uint seed() const noexcept override { return _seed; }
    [[nodiscard]] luisa::unique_ptr<Sampler::Instance> build(const Renderer& renderer) const noexcept override;
};

class SobolSamplerSpec final : public SamplerSpec
{
private:
    uint _spp;
    uint _seed;

public:
    SobolSamplerSpec(uint spp, uint seed) noexcept
        : _spp{spp}, _seed{seed} {}

    [[nodiscard]] uint spp() const noexcept { return _spp; }
    [[nodiscard]] uint seed() const noexcept { return _seed; }
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    [[nodiscard]] const Sampler* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
