#pragma once

#include <luisa/core/stl/optional.h>
#include <luisa/dsl/constant.h>
#include <luisa/runtime/buffer.h>

#include "base/sampler.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class ZSobolSampler final : public Sampler
{
public:
    class Instance final : public Sampler::Instance
    {
    public:
        static constexpr auto MaxDimension = 1024u;

    private:
        uint _log2_spp{};
        uint _n_base4_digits{};
        bool _matrices_uploaded{};
        luisa::optional<UInt> _dimension;
        luisa::optional<ULong> _morton_index;
        luisa::unique_ptr<Constant<uint2>> _sample_hash;
        Buffer<uint> _sobol_matrices;
        luisa::unique_ptr<Callable<ulong(ulong, uint)>> _get_sample_index_impl;

    private:
        [[nodiscard]] static ULong _encode_morton(Expr<uint2> pixel) noexcept;
        [[nodiscard]] static UInt _fast_owen_scramble(Expr<uint> seed, UInt v) noexcept;
        [[nodiscard]] ULong _get_sample_index(Expr<uint> dimension) const noexcept;
        [[nodiscard]] Float _sobol_sample(ULong index, uint dimension, Expr<uint> hash) const noexcept;

    public:
        Instance(const Renderer& renderer, const ZSobolSampler* sampler) noexcept;

        void reset(CommandBuffer& command_buffer, uint2 resolution, uint state_count) noexcept override;
        void start(UInt2 pixel, UInt index) noexcept override;
        [[nodiscard]] Float generate_1d() noexcept override;
        [[nodiscard]] Float2 generate_2d() noexcept override;
        [[nodiscard]] Float2 generate_pixel_2d() noexcept override;
    };

private:
    uint _seed;

public:
    ZSobolSampler(uint spp, uint seed) noexcept
        : Sampler{spp}, _seed{seed} {}

    [[nodiscard]] uint seed() const noexcept override { return _seed; }
    [[nodiscard]] luisa::unique_ptr<Sampler::Instance> build(const Renderer& renderer) const noexcept override;
};

class ZSobolSamplerSpec final : public SamplerSpec
{
private:
    uint _spp;
    uint _seed;

public:
    ZSobolSamplerSpec(uint spp, uint seed) noexcept
        : _spp{spp}, _seed{seed} {}

    [[nodiscard]] uint spp() const noexcept { return _spp; }
    [[nodiscard]] uint seed() const noexcept { return _seed; }
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    [[nodiscard]] const Sampler* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
