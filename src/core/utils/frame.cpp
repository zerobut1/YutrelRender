#include "frame.h"

namespace Yutrel
{
Frame::Frame(Expr<float3> s,
             Expr<float3> t,
             Expr<float3> n) noexcept
    : m_s{s}, m_t{t}, m_n{n} {}

Frame::Frame() noexcept
    : m_s{make_float3(1.f, 0.f, 0.f)},
      m_t{make_float3(0.f, 1.f, 0.f)},
      m_n{make_float3(0.f, 0.f, 1.f)} {}

Frame Frame::make(Expr<float3> n) noexcept
{
    auto sgn = ite(n.z >= 0.0f, 1.0f, -1.0f);
    auto a   = -1.f / (sgn + n.z);
    auto b   = n.x * n.y * a;
    auto s   = make_float3(1.f + sgn * sqr(n.x) * a, sgn * b, -sgn * n.x);
    auto t   = make_float3(b, sgn + sqr(n.y) * a, -n.y);
    return {normalize(s), normalize(t), n};
}

Frame Frame::make(Expr<float3> n, Expr<float3> s) noexcept
{
    auto ss = normalize(s - n * dot(n, s));
    auto tt = normalize(cross(n, ss));
    return {ss, tt, n};
}

Frame Frame::make(Expr<float3> n, Expr<float3> s, Expr<float3> t) noexcept
{
    auto ss = normalize(s - n * dot(n, s));
    auto tt = normalize(cross(n, ss));
    tt *= ite(dot(tt, t) < 0.0f, -1.0f, 1.0f);
    return {ss, tt, n};
}

Float3 Frame::local_to_world(Expr<float3> d) const noexcept
{
    return normalize(d.x * m_s + d.y * m_t + d.z * m_n);
}

Float3 Frame::world_to_local(Expr<float3> d) const noexcept
{
    return normalize(make_float3(dot(d, m_s), dot(d, m_t), dot(d, m_n)));
}

void Frame::flip() noexcept
{
    m_n = -m_n;
    m_t = -m_t;
}

Frame Frame::flipped(Expr<bool> flip) const noexcept
{
    return {m_s,
            ite(flip, -m_t, m_t),
            ite(flip, -m_n, m_n)};
}
} // namespace Yutrel
