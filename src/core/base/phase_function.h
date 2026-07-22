#pragma once

#include <luisa/dsl/syntax.h>

#include "utils/frame.h"
#include "utils/sampling.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class HGPhaseFunction
{
public:
    struct Sample
    {
        Float p;
        Float3 wi;
        Float pdf;
    };

private:
    Float _g;

public:
    explicit HGPhaseFunction(Expr<float> g) noexcept
        : _g{clamp(g, -0.99f, 0.99f)} {}

    [[nodiscard]] Float p(Expr<float3> wo, Expr<float3> wi) const noexcept
    {
        auto denom = 1.0f + sqr(_g) + 2.0f * _g * dot(wo, wi);
        return 0.25f * inv_pi * (1.0f - sqr(_g)) / (denom * sqrt(denom));
    }

    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept
    {
        return p(wo, wi);
    }

    [[nodiscard]] Sample sample(Expr<float3> wo, Expr<float2> u) const noexcept
    {
        auto cos_theta_v = ite(
            abs(_g) < 1e-3f,
            1.0f - 2.0f * u.x,
            -0.5f / _g *
                (1.0f + sqr(_g) -
                 sqr((1.0f - sqr(_g)) / (1.0f + _g - 2.0f * _g * u.x))));
        auto sin_theta_v = sqrt(max(0.0f, 1.0f - sqr(cos_theta_v)));
        auto phi         = 2.0f * pi * u.y;
        auto wi          = Frame::make(wo).local_to_world(
            make_float3(sin_theta_v * cos(phi), sin_theta_v * sin(phi), cos_theta_v));
        auto pdf_value = p(wo, wi);
        return {.p = pdf_value, .wi = wi, .pdf = pdf_value};
    }
};

} // namespace Yutrel
