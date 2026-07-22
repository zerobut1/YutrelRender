#include "interaction.h"

namespace Yutrel
{
Interaction::Interaction() noexcept
    : shape{Shape::Handle::decode(make_uint4(0u))},
      p_g{make_float3(0.0f)},
      n_g{make_float3(0.0f)},
      uv{make_float2(0.0f)},
      p_s{make_float3(0.0f)},
      shading{Frame::make(make_float3(0.0f, 1.0f, 0.0f))},
      inst_id{~0u},
      prim_id{~0u},
      prim_area{0.0f},
      front_face{false}
{
}

Interaction Interaction::from_point(Expr<float3> p) noexcept
{
    Interaction it;
    it.p_g = p;
    it.p_s = p;
    return it;
}

Interaction Interaction::from_uv(Expr<float2> uv) noexcept
{
    Interaction it;
    it.uv = uv;
    return it;
}

Interaction Interaction::from_surface(
    Shape::Handle shape,
    Expr<float3> p_g,
    Expr<float3> n_g,
    Expr<float2> uv,
    Expr<float3> p_s,
    Frame shading,
    Expr<uint> inst_id,
    Expr<uint> prim_id,
    Expr<float> prim_area,
    Expr<bool> front_face) noexcept
{
    Interaction it;
    it.shape      = std::move(shape);
    it.p_g        = p_g;
    it.n_g        = n_g;
    it.uv         = uv;
    it.p_s        = p_s;
    it.shading    = std::move(shading);
    it.inst_id    = inst_id;
    it.prim_id    = prim_id;
    it.prim_area  = prim_area;
    it.front_face = front_face;
    return it;
}

Float3 Interaction::p_robust(Expr<float3> w) const noexcept
{
    return ite(is_surface_interaction(), offset_ray_origin(p_s, n_g, w), p_s);
}

Var<Ray> Interaction::spawn_ray(Expr<float3> wi, Expr<float> t_max) const noexcept
{
    return make_ray(p_robust(wi), wi, 0.0f, t_max);
}

Var<Ray> Interaction::spawn_ray_to(Expr<float3> p) const noexcept
{
    auto p_from = p_robust(p - p_s);
    auto L      = p - p_from;
    auto d      = length(L);
    auto wi     = L * (1.0f / d);
    return make_ray(p_from, wi, 0.0f, d * 0.999f);
}
} // namespace Yutrel
