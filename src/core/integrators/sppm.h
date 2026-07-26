#pragma once

#include "base/integrator.h"

namespace Yutrel
{

class SPPMIntegrator final : public Integrator
{
public:
    class Instance final : public Integrator::Instance
    {
    private:
        uint _light_handle_buffer_id{0u};
        uint _n_lights{0u};

    public:
        Instance(Renderer& renderer, CommandBuffer& command_buffer, const SPPMIntegrator* integrator, const Sampler* sampler) noexcept;

        void render(Stream& stream, bool enable_display) override;
        void render_interactive(Stream& stream) override;

    private:
        [[nodiscard]] uint max_depth() const noexcept { return base<SPPMIntegrator>()->max_depth(); }
        [[nodiscard]] uint photons_per_iteration() const noexcept { return base<SPPMIntegrator>()->photons_per_iteration(); }
        [[nodiscard]] float initial_radius() const noexcept { return base<SPPMIntegrator>()->initial_radius(); }
    };

private:
    uint _max_depth;
    uint _photons_per_iteration;
    float _initial_radius;

public:
    SPPMIntegrator(uint max_depth, uint photons_per_iteration, float initial_radius) noexcept;

    [[nodiscard]] uint max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] uint photons_per_iteration() const noexcept { return _photons_per_iteration; }
    [[nodiscard]] float initial_radius() const noexcept { return _initial_radius; }

    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Renderer& renderer, CommandBuffer& command_buffer, const Sampler* sampler) const noexcept override;
};

class SPPMIntegratorSpec final : public IntegratorSpec
{
private:
    uint _max_depth;
    uint _photons_per_iteration;
    float _initial_radius;

public:
    SPPMIntegratorSpec(uint max_depth, uint photons_per_iteration, float initial_radius) noexcept
        : _max_depth{max_depth}, _photons_per_iteration{photons_per_iteration}, _initial_radius{initial_radius} {}

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        if (_max_depth == 0u)
        {
            return spec_validation_error("SPPM max depth must be positive.");
        }
        if (_photons_per_iteration == 0u)
        {
            return spec_validation_error("SPPM photons per iteration must be positive.");
        }
        if (!std::isfinite(_initial_radius) || _initial_radius <= 0.0f)
        {
            return spec_validation_error("SPPM initial radius must be finite and positive.");
        }
        return luisa::nullopt;
    }

    [[nodiscard]] const Integrator* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
