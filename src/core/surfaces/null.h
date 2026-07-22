#pragma once

#include "base/interaction.h"
#include "base/surface.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
class NullSurface : public Surface
{
public:
    class Instance;
    class Closure;

    explicit NullSurface(bool two_sided = false) noexcept : Surface{two_sided} {}

    [[nodiscard]] bool is_null() const noexcept override { return true; }
    [[nodiscard]] uint properties() const noexcept override { return 0u; }
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class NullSurface::Instance final : public Surface::Instance
{
public:
    Instance(const Renderer& renderer, const NullSurface* surface) noexcept
        : Surface::Instance{renderer, surface} {}

    [[nodiscard]] luisa::string closure_identifier() const noexcept override { return "Null"; }
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(
        SampledWavelengths& swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure* closure, const Interaction& it,
                          Expr<float3> wo, Expr<float> eta_i) const noexcept override;
};

class NullSurface::Closure final : public Surface::Closure
{
public:
    struct Context
    {
        Interaction it;
    };

    using Surface::Closure::Closure;

    [[nodiscard]] const Interaction& it() const noexcept override { return context<Context>().it; }
    [[nodiscard]] UInt lobe_flags() const noexcept override { return 0u; }

private:
    [[nodiscard]] Surface::Sample sample_impl(
        Expr<float3>, Expr<float>, Expr<float2>, TransportMode, ScatterFlags) const noexcept override
    {
        return Surface::Sample::zero(swl().dimension());
    }
    [[nodiscard]] Surface::Evaluation evaluate_impl(
        Expr<float3>, Expr<float3>, TransportMode, ScatterFlags) const noexcept override
    {
        return Surface::Evaluation::zero(swl().dimension());
    }
};

class NullSurfaceSpec final : public SurfaceSpec
{
private:
    bool _two_sided;

public:
    explicit NullSurfaceSpec(bool two_sided = false) noexcept
        : _two_sided{two_sided} {}

    [[nodiscard]] const Surface* build(SceneBuilder& builder) const noexcept override
    {
        return builder.emplace<Surface, NullSurface>(_two_sided);
    }
};
} // namespace Yutrel
