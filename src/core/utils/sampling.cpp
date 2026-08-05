#include "sampling.h"

#include <algorithm>
#include <numeric>

#include <luisa/core/logging.h>

namespace Yutrel
{

Float3 sample_uniform_triangle(Expr<float2> u) noexcept
{
    static Callable impl = [](Float2 u) noexcept
    {
        auto uv = ite(
            u.x < u.y,
            make_float2(0.5f * u.x, -0.5f * u.x + u.y),
            make_float2(-0.5f * u.y + u.x, 0.5f * u.y));
        return make_float3(uv, 1.0f - uv.x - uv.y);
    };
    return impl(u);
}

Float2 sample_uniform_disk_concentric(Expr<float2> u) noexcept
{
    static Callable impl = [](Float2 u_in) noexcept
    {
        auto u     = u_in * 2.0f - 1.0f;
        auto p     = abs(u.x) > abs(u.y);
        auto r     = ite(p, u.x, u.y);
        auto theta = ite(p, pi_over_four * (u.y / u.x), pi_over_two - pi_over_four * (u.x / u.y));
        return r * make_float2(cos(theta), sin(theta));
    };
    return impl(u);
}

Float2 sample_uniform_disk_polar(Expr<float2> u) noexcept
{
    auto r   = sqrt(u.x);
    auto phi = 2.0f * pi * u.y;
    return r * make_float2(cos(phi), sin(phi));
}

Float3 sample_cosine_hemisphere(Expr<float2> u) noexcept
{
    static Callable impl = [](Float2 u) noexcept
    {
        auto d = sample_uniform_disk_concentric(u);
        auto z = sqrt(max(1.0f - d.x * d.x - d.y * d.y, 0.0f));
        return make_float3(d.x, d.y, z);
    };
    return impl(u);
}

Float3 sample_uniform_sphere(Expr<float2> u) noexcept
{
    static Callable impl = [](Float2 u) noexcept
    {
        auto z   = 1.0f - 2.0f * u.x;
        auto r   = sqrt(max(1.0f - z * z, 0.0f));
        auto phi = 2.0f * pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), z);
    };
    return impl(u);
}

Float sample_exponential(Expr<float> u, Expr<float> a) noexcept
{
    return -log(1.0f - u) / a;
}

Float3 equal_area_square_to_sphere(Expr<float2> uv) noexcept
{
    static Callable impl = [](Float2 p) noexcept
    {
        auto u  = 2.0f * p.x - 1.0f;
        auto v  = 2.0f * p.y - 1.0f;
        auto up = abs(u);
        auto vp = abs(v);

        auto signed_distance = 1.0f - (up + vp);
        auto d               = abs(signed_distance);
        auto r               = 1.0f - d;
        auto phi             = ite(r == 0.0f, 1.0f, (vp - up) / r + 1.0f) * pi_over_four;
        auto z               = copysign(1.0f - r * r, signed_distance);
        auto cos_phi         = copysign(cos(phi), u);
        auto sin_phi         = copysign(sin(phi), v);
        auto radial          = r * sqrt(max(2.0f - r * r, 0.0f));
        return make_float3(cos_phi * radial, sin_phi * radial, z);
    };
    return impl(uv);
}

Float2 equal_area_sphere_to_square(Expr<float3> w) noexcept
{
    static Callable impl = [](Float3 d) noexcept
    {
        d      = normalize(d);
        auto x = abs(d.x);
        auto y = abs(d.y);
        auto z = abs(d.z);
        auto r = sqrt(max(1.0f - z, 0.0f));

        auto a = max(x, y);
        auto b = min(x, y);
        b      = ite(a == 0.0f, 0.0f, b / a);

        constexpr auto t1 = 0.406758566246788489601959989e-5f;
        constexpr auto t2 = 0.636226545274016134946890922156f;
        constexpr auto t3 = 0.61572017898280213493197203466e-2f;
        constexpr auto t4 = -0.247333733281268944196501420480f;
        constexpr auto t5 = 0.881770664775316294736387951347e-1f;
        constexpr auto t6 = 0.419038818029165735901852432784e-1f;
        constexpr auto t7 = -0.251390972343483509333252996350e-1f;
        auto phi          = ((((((t7 * b + t6) * b + t5) * b + t4) * b + t3) * b + t2) * b + t1);
        phi               = ite(x < y, 1.0f - phi, phi);

        auto vv         = phi * r;
        auto uu         = r - vv;
        auto south      = d.z < 0.0f;
        auto southern_u = 1.0f - vv;
        auto southern_v = 1.0f - uu;
        uu              = ite(south, southern_u, uu);
        vv              = ite(south, southern_v, vv);
        uu              = copysign(uu, d.x);
        vv              = copysign(vv, d.y);
        return clamp(0.5f * (make_float2(uu, vv) + 1.0f), 0.0f, 1.0f);
    };
    return impl(w);
}

std::pair<luisa::vector<AliasEntry>, luisa::vector<float>>
create_alias_table(luisa::span<const float> values) noexcept
{
    if (values.empty()) [[unlikely]]
    {
        return {};
    }

    auto sum = 0.0;
    for (auto v : values)
    {
        sum += std::abs(v);
    }
    luisa::vector<float> pdf(values.size());
    if (sum == 0.) [[unlikely]]
    {
        auto n = static_cast<double>(values.size());
        std::fill(pdf.begin(), pdf.end(), static_cast<float>(1.0 / n));
        luisa::vector<AliasEntry> table(values.size());
        for (auto i = 0u; i < values.size(); i++)
        {
            table[i] = {1.0f, i};
        }
        return std::make_pair(std::move(table), std::move(pdf));
    }
    auto inv_sum = 1.0 / sum;
    std::transform(
        values.begin(),
        values.end(),
        pdf.begin(),
        [inv_sum](auto v) noexcept
    {
        return static_cast<float>(std::abs(v) * inv_sum);
    });

    auto ratio = static_cast<double>(values.size()) / sum;
    static thread_local luisa::vector<uint> over;
    static thread_local luisa::vector<uint> under;
    over.clear();
    under.clear();
    over.reserve(next_pow2(values.size()));
    under.reserve(next_pow2(values.size()));

    luisa::vector<AliasEntry> table(values.size());
    for (auto i = 0u; i < values.size(); i++)
    {
        auto p   = static_cast<float>(values[i] * ratio);
        table[i] = {p, i};
        (p > 1.0f ? over : under).emplace_back(i);
    }

    while (!over.empty() && !under.empty())
    {
        auto o = over.back();
        auto u = under.back();
        over.pop_back();
        under.pop_back();
        table[o].prob -= 1.0f - table[u].prob;
        table[u].alias = o;
        if (table[o].prob > 1.0f)
        {
            over.push_back(o);
        }
        else if (table[o].prob < 1.0f)
        {
            under.push_back(o);
        }
    }
    for (auto i : over)
    {
        table[i] = {1.0f, i};
    }
    for (auto i : under)
    {
        table[i] = {1.0f, i};
    }

    return std::make_pair(std::move(table), std::move(pdf));
}

AliasDistribution2D create_alias_distribution_2d(
    luisa::span<const float> weights, uint2 resolution) noexcept
{
    auto pixel_count = static_cast<size_t>(resolution.x) * resolution.y;
    LUISA_ASSERT(
        weights.size() == pixel_count,
        "Invalid 2D alias distribution size: expected {}, got {}.",
        pixel_count,
        weights.size());

    luisa::vector<float> row_averages(resolution.y);
    AliasDistribution2D distribution{
        .aliases = luisa::vector<AliasEntry>(resolution.y + pixel_count),
        .pdfs = luisa::vector<float>(pixel_count),
    };
    for (auto y = 0u; y < resolution.y; y++)
    {
        auto row = weights.subspan(static_cast<size_t>(y) * resolution.x, resolution.x);
        auto row_sum = std::accumulate(row.begin(), row.end(), 0.0);
        row_averages[y] = static_cast<float>(row_sum / resolution.x);
        auto [row_aliases, row_pdfs] = create_alias_table(row);
        std::copy(row_aliases.begin(), row_aliases.end(),
                  distribution.aliases.begin() + resolution.y + static_cast<size_t>(y) * resolution.x);
        std::copy(row_pdfs.begin(), row_pdfs.end(),
                  distribution.pdfs.begin() + static_cast<size_t>(y) * resolution.x);
    }

    auto [marginal_aliases, marginal_pdfs] = create_alias_table(row_averages);
    std::copy(marginal_aliases.begin(), marginal_aliases.end(), distribution.aliases.begin());
    auto uv_cell_count = static_cast<float>(pixel_count);
    for (auto y = 0u; y < resolution.y; y++)
    {
        auto scale = marginal_pdfs[y] * uv_cell_count;
        for (auto x = 0u; x < resolution.x; x++)
        {
            distribution.pdfs[static_cast<size_t>(y) * resolution.x + x] *= scale;
        }
    }
    return distribution;
}

Float balance_heuristic(Expr<uint> nf, Expr<float> fPdf, Expr<uint> ng, Expr<float> gPdf) noexcept
{
    static Callable impl = [](UInt nf, Float fPdf, UInt ng, Float gPdf) noexcept
    {
        auto sum_f = nf * fPdf;
        auto sum   = sum_f + ng * gPdf;
        return ite(sum == 0.0f, 0.0f, sum_f / sum);
    };
    return impl(nf, fPdf, ng, gPdf);
}

Float power_heuristic(Expr<uint> nf, Expr<float> fPdf, Expr<uint> ng, Expr<float> gPdf) noexcept
{
    static Callable impl = [](UInt nf, Float fPdf, UInt ng, Float gPdf) noexcept
    {
        Float f = nf * fPdf, g = ng * gPdf;
        auto ff  = f * f;
        auto gg  = g * g;
        auto sum = ff + gg;
        return ite(luisa::compute::isinf(ff), 1.f, ite(sum == 0.f, 0.f, ff / sum));
    };
    return impl(nf, fPdf, ng, gPdf);
}

Float balance_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept
{
    return balance_heuristic(1u, fPdf, 1u, gPdf);
}

Float power_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept
{
    return power_heuristic(1u, fPdf, 1u, gPdf);
}

} // namespace Yutrel
