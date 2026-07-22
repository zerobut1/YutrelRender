#pragma once

#include <luisa/dsl/syntax.h>

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

[[nodiscard]] Float3 sample_uniform_triangle(Expr<float2> u) noexcept;
[[nodiscard]] Float2 sample_uniform_disk_concentric(Expr<float2> u) noexcept;
[[nodiscard]] Float2 sample_uniform_disk_polar(Expr<float2> u) noexcept;
[[nodiscard]] Float3 sample_cosine_hemisphere(Expr<float2> u) noexcept;
[[nodiscard]] Float3 sample_uniform_sphere(Expr<float2> u) noexcept;
[[nodiscard]] Float sample_exponential(Expr<float> u, Expr<float> a) noexcept;

// Clarberg's equal-area square/sphere mapping, matching PBRT-v4.
[[nodiscard]] Float3 equal_area_square_to_sphere(Expr<float2> uv) noexcept;
[[nodiscard]] Float2 equal_area_sphere_to_square(Expr<float3> w) noexcept;

struct AliasEntry
{
    float prob;
    uint alias;
};
// reference: https://github.com/AirGuanZ/agz-utils
[[nodiscard]] std::pair<luisa::vector<AliasEntry>, luisa::vector<float> /* pdf */>
create_alias_table(luisa::span<const float> values) noexcept;

template <typename Table>
[[nodiscard]] inline auto sample_alias_table(
    const Table& table, Expr<uint> n, Expr<float> u_in, Expr<uint> offset = 0u) noexcept
{
    using namespace luisa::compute;
    auto u          = u_in * cast<float>(n);
    auto i          = clamp(cast<uint>(u), 0u, n - 1u);
    auto u_remapped = fract(u);
    auto entry      = table->read(i + offset);
    auto index      = ite(u_remapped < entry.prob, i, entry.alias);
    auto uu         = ite(u_remapped < entry.prob, u_remapped / entry.prob, (u_remapped - entry.prob) / (1.0f - entry.prob));
    return std::make_pair(index, uu);
}

template <typename ProbTable, typename AliasTable>
[[nodiscard]] inline auto sample_alias_table(
    const ProbTable& probs, const AliasTable& indices,
    Expr<uint> n, Expr<float> u_in, Expr<uint> offset = 0u) noexcept
{
    using namespace luisa::compute;
    auto u          = u_in * cast<float>(n);
    auto i          = clamp(cast<uint>(u), 0u, n - 1u);
    auto u_remapped = fract(u);
    auto prob       = probs->read(i + offset);
    auto index      = ite(u_remapped < prob, i, indices->read(i + offset));
    auto uu         = ite(
        u_remapped < prob,
        u_remapped / prob,
        (u_remapped - prob) / (1.0f - prob));
    return std::make_pair(index, uu);
}

[[nodiscard]] Float balance_heuristic(Expr<uint> nf, Expr<float> fPdf, Expr<uint> ng, Expr<float> gPdf) noexcept;
[[nodiscard]] Float power_heuristic(Expr<uint> nf, Expr<float> fPdf, Expr<uint> ng, Expr<float> gPdf) noexcept;
[[nodiscard]] Float balance_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept;
[[nodiscard]] Float power_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept;

} // namespace Yutrel

LUISA_STRUCT(Yutrel::AliasEntry, prob, alias){};
