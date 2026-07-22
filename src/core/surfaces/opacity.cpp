#include "opacity.h"

#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
bool OpacitySurface::maybe_non_opaque() const noexcept
{
    if (_base->maybe_non_opaque())
    {
        return true;
    }
    auto value = _alpha->evaluate_static();
    return !value || value->x < 1.0f;
}

luisa::unique_ptr<Surface::Instance> OpacitySurface::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto tag            = renderer.register_surface(command_buffer, _base);
    auto base_instance  = renderer.surfaces().impl(tag);
    auto alpha_instance = renderer.build_texture(command_buffer, _alpha);
    return luisa::make_unique<Instance>(renderer, this, base_instance, alpha_instance);
}

bool OpacitySurface::Instance::maybe_non_opaque() const noexcept
{
    return base<OpacitySurface>()->maybe_non_opaque();
}

luisa::optional<Float> OpacitySurface::Instance::evaluate_opacity(
    const Interaction& it, Expr<float> time) const noexcept
{
    auto opacity = _base->evaluate_opacity(it, time).value_or(1.0f);
    return opacity * _alpha->evaluate(it, time).x;
}

luisa::string OpacitySurface::Instance::closure_identifier() const noexcept
{
    return _base->closure_identifier();
}

luisa::unique_ptr<Surface::Closure> OpacitySurface::Instance::create_closure(
    SampledWavelengths& swl, Expr<float> time) const noexcept
{
    return _base->create_closure(swl, time);
}

void OpacitySurface::Instance::populate_closure(
    Closure* closure, const Interaction& it, Expr<float3> wo, Expr<float> eta_i) const noexcept
{
    _base->populate_closure(closure, it, wo, eta_i);
}

const Surface* OpacitySurfaceSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Surface, OpacitySurface>(
        builder.resolve(_base), builder.resolve(_alpha));
}
} // namespace Yutrel
