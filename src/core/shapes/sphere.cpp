#include "sphere.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <mutex>

#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>

#include "scene/scene_builder.h"

namespace Yutrel
{
namespace
{

static constexpr auto sphere_base_vertices = std::array{
    make_float3(0.f, -0.525731f, 0.850651f),
    make_float3(0.850651f, 0.f, 0.525731f),
    make_float3(0.850651f, 0.f, -0.525731f),
    make_float3(-0.850651f, 0.f, -0.525731f),
    make_float3(-0.850651f, 0.f, 0.525731f),
    make_float3(-0.525731f, 0.850651f, 0.f),
    make_float3(0.525731f, 0.850651f, 0.f),
    make_float3(0.525731f, -0.850651f, 0.f),
    make_float3(-0.525731f, -0.850651f, 0.f),
    make_float3(0.f, -0.525731f, -0.850651f),
    make_float3(0.f, 0.525731f, -0.850651f),
    make_float3(0.f, 0.525731f, 0.850651f),
};

static constexpr auto sphere_base_triangles = std::array{
    Triangle{1u, 2u, 6u},
    Triangle{1u, 7u, 2u},
    Triangle{3u, 4u, 5u},
    Triangle{4u, 3u, 8u},
    Triangle{6u, 5u, 11u},
    Triangle{5u, 6u, 10u},
    Triangle{9u, 10u, 2u},
    Triangle{10u, 9u, 3u},
    Triangle{7u, 8u, 9u},
    Triangle{8u, 7u, 0u},
    Triangle{11u, 0u, 1u},
    Triangle{0u, 11u, 4u},
    Triangle{6u, 2u, 10u},
    Triangle{1u, 6u, 11u},
    Triangle{3u, 5u, 10u},
    Triangle{5u, 4u, 11u},
    Triangle{2u, 7u, 9u},
    Triangle{7u, 1u, 0u},
    Triangle{3u, 9u, 8u},
    Triangle{4u, 8u, 0u},
};

[[nodiscard]] float3 normalize_host(float3 v) noexcept
{
    auto inv_length = 1.0f / std::sqrt(dot(v, v));
    return v * inv_length;
}

[[nodiscard]] uint64_t edge_key(uint a, uint b) noexcept
{
    auto lo = std::min(a, b);
    auto hi = std::max(a, b);
    return static_cast<uint64_t>(lo) << 32u | hi;
}

[[nodiscard]] float2 direction_to_uv(float3 w) noexcept
{
    static constexpr auto pi = 3.14159265358979323846f;
    auto theta               = std::acos(std::clamp(w.y, -1.0f, 1.0f));
    auto phi                 = std::atan2(w.x, w.z);
    auto u                   = phi * (0.5f / pi);
    if (u < 0.0f)
    {
        u += 1.0f;
    }
    return make_float2(u, theta / pi);
}

} // namespace

class SphereGeometry
{
private:
    luisa::vector<Vertex> m_vertices;
    luisa::vector<Triangle> m_triangles;

public:
    SphereGeometry(float radius, uint subdivision) noexcept
    {
        luisa::vector<float3> positions;
        positions.reserve(sphere_base_vertices.size());
        for (auto p : sphere_base_vertices)
        {
            positions.emplace_back(normalize_host(p));
        }

        m_triangles.assign(sphere_base_triangles.begin(), sphere_base_triangles.end());
        for (auto level = 0u; level < subdivision; level++)
        {
            luisa::unordered_map<uint64_t, uint> edge_midpoints;
            edge_midpoints.reserve(m_triangles.size() * 3u / 2u);
            luisa::vector<Triangle> next;
            next.reserve(m_triangles.size() * 4u);

            auto midpoint = [&](uint a, uint b) noexcept
            {
                auto key = edge_key(a, b);
                if (auto iter = edge_midpoints.find(key); iter != edge_midpoints.end())
                {
                    return iter->second;
                }
                auto index = static_cast<uint>(positions.size());
                positions.emplace_back(normalize_host(positions[a] + positions[b]));
                edge_midpoints.emplace(key, index);
                return index;
            };

            for (auto triangle : m_triangles)
            {
                auto ab = midpoint(triangle.i0, triangle.i1);
                auto bc = midpoint(triangle.i1, triangle.i2);
                auto ca = midpoint(triangle.i2, triangle.i0);
                next.emplace_back(Triangle{triangle.i0, ab, ca});
                next.emplace_back(Triangle{triangle.i1, bc, ab});
                next.emplace_back(Triangle{triangle.i2, ca, bc});
                next.emplace_back(Triangle{ab, bc, ca});
            }
            m_triangles = std::move(next);
        }

        m_vertices.reserve(positions.size());
        for (auto direction : positions)
        {
            m_vertices.emplace_back(Vertex::encode(radius * direction, direction, direction_to_uv(direction)));
        }
    }

    [[nodiscard]] MeshView mesh() const noexcept { return {m_vertices, m_triangles}; }

    [[nodiscard]] static luisa::shared_ptr<const SphereGeometry> create(float radius, uint subdivision) noexcept
    {
        auto key = static_cast<uint64_t>(std::bit_cast<uint32_t>(radius)) << 32u | subdivision;
        static luisa::unordered_map<uint64_t, luisa::shared_ptr<const SphereGeometry>> cache;
        static std::mutex mutex;
        std::scoped_lock lock{mutex};
        if (auto iter = cache.find(key); iter != cache.end())
        {
            return iter->second;
        }
        auto geometry = luisa::make_shared<const SphereGeometry>(radius, subdivision);
        cache.emplace(key, geometry);
        return geometry;
    }
};

Sphere::Sphere(float radius, uint subdivision) noexcept
    : m_geometry{[radius, subdivision]() noexcept
      {
          LUISA_ASSERT(std::isfinite(radius) && radius > 0.0f, "Sphere radius must be finite and positive.");
          LUISA_ASSERT(subdivision <= max_subdivision, "Sphere subdivision level {} exceeds maximum {}.", subdivision, max_subdivision);
          return SphereGeometry::create(radius, subdivision);
      }()} {}

MeshView Sphere::mesh() const noexcept
{
    return m_geometry->mesh();
}

luisa::optional<luisa::string> SphereShapeSpec::validate() const noexcept
{
    if (!std::isfinite(m_radius) || m_radius <= 0.0f)
    {
        return spec_validation_error("Sphere radius must be finite and positive.");
    }
    if (m_subdivision > Sphere::max_subdivision)
    {
        return spec_validation_error("Sphere subdivision level exceeds the supported maximum of 8.");
    }
    return luisa::nullopt;
}

const Shape* SphereShapeSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Shape, Sphere>(m_radius, m_subdivision);
}

} // namespace Yutrel
