#pragma once

#include <luisa/dsl/syntax.h>

#include "base/shape.h"
#include "utils/frame.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

struct ShadingAttribute
{
    Float3 pg;
    Float3 ng;
    Float area;
    Float3 ps;
    Float3 ns;
    Float3 dpdu;
    Float3 dpdv;
    Float2 uv;
};

class Interaction
{
public:
    Shape::Handle shape;
    Float3 p_g;
    Float3 n_g;
    Float2 uv;
    Float3 p_s;
    Frame shading;
    UInt inst_id;
    UInt prim_id;
    Float prim_area;
    Bool front_face;

public:
    Interaction() noexcept;

    [[nodiscard]] static Interaction from_point(Expr<float3> p) noexcept;
    [[nodiscard]] static Interaction from_uv(Expr<float2> uv) noexcept;
    [[nodiscard]] static Interaction from_surface(
        Shape::Handle shape,
        Expr<float3> p_g,
        Expr<float3> n_g,
        Expr<float2> uv,
        Expr<float3> p_s,
        Frame shading,
        Expr<uint> inst_id,
        Expr<uint> prim_id,
        Expr<float> prim_area,
        Expr<bool> front_face) noexcept;

    [[nodiscard]] Bool is_surface_interaction() const noexcept { return inst_id != ~0u; }

public:
    static constexpr auto default_t_max = std::numeric_limits<float>::max();
    [[nodiscard]] Float3 p_robust(Expr<float3> w) const noexcept;
    [[nodiscard]] Var<Ray> spawn_ray(Expr<float3> wi, Expr<float> t_max = default_t_max) const noexcept;
    [[nodiscard]] Var<Ray> spawn_ray_to(Expr<float3> p) const noexcept;
};
} // namespace Yutrel
