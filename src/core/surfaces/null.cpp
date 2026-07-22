#include "null.h"

#include "base/renderer.h"

namespace Yutrel
{
luisa::unique_ptr<Surface::Instance> NullSurface::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    return luisa::make_unique<Instance>(renderer, this);
}

luisa::unique_ptr<Surface::Closure> NullSurface::Instance::create_closure(
    SampledWavelengths& swl, Expr<float> time) const noexcept
{
    return luisa::make_unique<Closure>(renderer(), swl, time);
}

void NullSurface::Instance::populate_closure(
    Surface::Closure* closure, const Interaction& it,
    Expr<float3> wo, Expr<float> eta_i) const noexcept
{
    closure->bind(Closure::Context{.it = it});
}
} // namespace Yutrel
