#pragma once

#include <luisa/core/stl/optional.h>
#include <luisa/runtime/buffer.h>

#include "base/sampler.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class IndependentSampler final : public Sampler
{
public:
    class Instance final : public Sampler::Instance
    {
    private:
        Buffer<uint> _states;
        luisa::optional<Var<uint>> _state;

    public:
        Instance(const Renderer& renderer, const IndependentSampler* sampler) noexcept;

        void reset(CommandBuffer& command_buffer, uint2 resolution, uint state_count) noexcept override;
        void start(UInt2 pixel, UInt index) noexcept override;
        [[nodiscard]] Float generate_1d() noexcept override;
        [[nodiscard]] Float2 generate_2d() noexcept override;
    };

private:
    uint _seed;

public:
    IndependentSampler(uint spp, uint seed) noexcept
        : Sampler{spp}, _seed{seed} {}

    [[nodiscard]] uint seed() const noexcept override { return _seed; }
    [[nodiscard]] luisa::unique_ptr<Sampler::Instance> build(const Renderer& renderer) const noexcept override;
};

class IndependentSamplerSpec final : public SamplerSpec
{
private:
    uint _spp;
    uint _seed;

public:
    IndependentSamplerSpec(uint spp, uint seed) noexcept
        : _spp{spp}, _seed{seed} {}

    [[nodiscard]] uint spp() const noexcept { return _spp; }
    [[nodiscard]] uint seed() const noexcept { return _seed; }
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override { return _spp == 0u ? spec_validation_error("Sampler SPP must be greater than zero.") : luisa::nullopt; }
    [[nodiscard]] const Sampler* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
