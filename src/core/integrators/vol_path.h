#pragma once

#include "base/integrator.h"
#include "utils/rng.h"
#include "utils/spectra.h"

namespace Yutrel
{

class VolPathIntegrator final : public ProgressiveIntegrator
{
public:
    class Instance final : public ProgressiveIntegrator::Instance
    {
    private:
        struct TransmittanceResult
        {
            SampledSpectrum T;
            SampledSpectrum r_l;
            SampledSpectrum r_u;
            Bool visible;
        };

    public:
        Instance(Renderer& renderer, CommandBuffer& command_buffer, const VolPathIntegrator* integrator, const Sampler* sampler) noexcept;

    private:
        [[nodiscard]] Float3 Li(const Camera::Instance* camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept override;
        [[nodiscard]] TransmittanceResult trace_transmittance(
            Var<Ray> ray, UInt current_medium, const SampledWavelengths& swl,
            Expr<float> time, PBRTRNG& rng) const noexcept;
    };

public:
    explicit VolPathIntegrator(uint max_depth) noexcept;

    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Renderer& renderer, CommandBuffer& command_buffer, const Sampler* sampler) const noexcept override;
};

class VolPathIntegratorSpec final : public IntegratorSpec
{
private:
    uint _max_depth;

public:
    explicit VolPathIntegratorSpec(uint max_depth) noexcept
        : _max_depth{max_depth} {}

    [[nodiscard]] const Integrator* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
