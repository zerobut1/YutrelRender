#pragma once

#include <luisa/dsl/syntax.h>

namespace Yutrel
{
using namespace luisa;

struct alignas(16) Vertex
{
    float px;
    float py;
    float pz;
    float nx;
    float ny;
    float nz;
    float u;
    float v;

    [[nodiscard]] static auto encode(float3 p, float3 n, float2 uv) noexcept
    {
        return Vertex{p.x, p.y, p.z, n.x, n.y, n.z, uv.x, uv.y};
    };
    [[nodiscard]] auto position() const noexcept { return make_float3(px, py, pz); }
    [[nodiscard]] auto normal() const noexcept { return make_float3(nx, ny, nz); }
    [[nodiscard]] auto uv() const noexcept { return make_float2(u, v); }
};

static_assert(sizeof(Vertex) == 32u);

} // namespace Yutrel

// clang-format off
LUISA_STRUCT(Yutrel::Vertex, px, py, pz, nx, ny, nz, u, v) {
    [[nodiscard]] auto position() const noexcept { return make_float3(px, py, pz); }
    [[nodiscard]] auto normal() const noexcept { return make_float3(nx, ny, nz); }
    [[nodiscard]] auto uv() const noexcept { return make_float2(u, v); }
};
// clang-format on