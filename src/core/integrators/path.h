#pragma once

#include "base/integrator.h"

namespace Yutrel
{

class PathIntegrator final : public ProgressiveIntegrator
{
public:
    class Instance final : public ProgressiveIntegrator::Instance
    {
    public:
        Instance(Renderer& renderer, CommandBuffer& command_buffer, const PathIntegrator* integrator, const Sampler* sampler) noexcept;

    private:
        [[nodiscard]] Float3 Li(const Camera::Instance* camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept override;
    };

public:
    explicit PathIntegrator(uint max_depth) noexcept;

    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Renderer& renderer, CommandBuffer& command_buffer, const Sampler* sampler) const noexcept override;
};

class PathIntegratorSpec final : public IntegratorSpec
{
private:
    uint _max_depth;

public:
    explicit PathIntegratorSpec(uint max_depth) noexcept
        : _max_depth{max_depth} {}

    [[nodiscard]] const Integrator* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
