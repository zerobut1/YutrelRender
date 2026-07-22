#pragma once

#include <luisa/dsl/syntax.h>

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Frame
{
private:
    Float3 m_s;
    Float3 m_t;
    Float3 m_n;

public:
    Frame() noexcept;
    Frame(Expr<float3> s, Expr<float3> t, Expr<float3> n) noexcept;
    void flip() noexcept;
    [[nodiscard]] Frame flipped(Expr<bool> flip) const noexcept;
    [[nodiscard]] static Frame make(Expr<float3> n) noexcept;
    [[nodiscard]] static Frame make(Expr<float3> n, Expr<float3> s) noexcept;
    [[nodiscard]] Float3 local_to_world(Expr<float3> d) const noexcept;
    [[nodiscard]] Float3 world_to_local(Expr<float3> d) const noexcept;
    [[nodiscard]] Expr<float3> s() const noexcept { return m_s; }
    [[nodiscard]] Expr<float3> t() const noexcept { return m_t; }
    [[nodiscard]] Expr<float3> n() const noexcept { return m_n; }
};

[[nodiscard]] inline auto sqr(auto x) noexcept { return x * x; }
[[nodiscard]] inline auto one_minus_sqr(auto x) noexcept { return 1.f - sqr(x); }
[[nodiscard]] inline auto abs_dot(Expr<float3> u, Expr<float3> v) noexcept { return abs(dot(u, v)); }
[[nodiscard]] inline auto cos_theta(Expr<float3> w) { return w.z; }
[[nodiscard]] inline auto cos2_theta(Expr<float3> w) { return sqr(w.z); }
[[nodiscard]] inline auto abs_cos_theta(Expr<float3> w) { return abs(w.z); }
[[nodiscard]] inline auto sin2_theta(Expr<float3> w) { return max(0.0f, 1.0f - cos2_theta(w)); }
[[nodiscard]] inline auto sin_theta(Expr<float3> w) { return sqrt(sin2_theta(w)); }
[[nodiscard]] inline auto tan_theta(Expr<float3> w) { return sin_theta(w) / cos_theta(w); }
[[nodiscard]] inline auto tan2_theta(Expr<float3> w) { return sin2_theta(w) / cos2_theta(w); }
[[nodiscard]] inline auto cos_phi(Expr<float3> w)
{
    auto sin_theta_w = sin_theta(w);
    return ite(sin_theta_w == 0.0f, 1.0f, clamp(w.x / sin_theta_w, -1.0f, 1.0f));
}
[[nodiscard]] inline auto sin_phi(Expr<float3> w)
{
    auto sin_theta_w = sin_theta(w);
    return ite(sin_theta_w == 0.0f, 0.0f, clamp(w.y / sin_theta_w, -1.0f, 1.0f));
}
[[nodiscard]] inline auto cos2_phi(Expr<float3> w) { return sqr(cos_phi(w)); }
[[nodiscard]] inline auto sin2_phi(Expr<float3> w) { return sqr(sin_phi(w)); }
[[nodiscard]] inline auto same_hemisphere(Expr<float3> w, Expr<float3> wp) noexcept { return w.z * wp.z > 0.0f; }
} // namespace Yutrel
