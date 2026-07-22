#include "surface.h"

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>

#include "base/interaction.h"

namespace Yutrel
{
Surface::Surface(bool two_sided) noexcept
    : m_two_sided{two_sided} {}

void Surface::Instance::closure(PolymorphicCall<Closure>& call, const Interaction& it, Expr<float3> wo,
                                SampledWavelengths& swl, Expr<float> time, Expr<float> eta_i) const noexcept
{
    auto cls         = call.collect(closure_identifier(), [&]
    {
        return create_closure(swl, time);
    });
    auto oriented_it = it;
    if (base()->two_sided())
    {
        auto flip           = dot(wo, it.shading.n()) < 0.0f;
        oriented_it.shading = it.shading.flipped(flip);
    }
    populate_closure(cls, oriented_it, wo, eta_i);
}

Surface::Sample Surface::Closure::sample(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u,
                                         TransportMode mode, ScatterFlags flags) const noexcept
{
    // BxDF hemisphere tests use the authoritative shading frame, matching PBRT.
    auto s = Surface::Sample::zero(swl().dimension());
    $outline
    {
        s = sample_impl(wo, u_lobe, u, mode, flags);
    };

    return s;
}

Surface::Evaluation Surface::Closure::evaluate(Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode, ScatterFlags flags) const noexcept
{
    auto eval = Surface::Evaluation::zero(swl().dimension());
    $outline
    {
        eval = evaluate_impl(wo, wi, mode, flags);
    };
    return eval;
}
} // namespace Yutrel
