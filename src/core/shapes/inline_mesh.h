#pragma once

#include <cmath>

#include "base/shape.h"
#include "scene/spec_base.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class InlineMesh final : public Shape
{
private:
    luisa::vector<Vertex> m_vertices;
    luisa::vector<Triangle> m_triangles;
    uint m_properties{};

public:
    InlineMesh(luisa::vector<float3> positions,
               luisa::vector<float3> normals,
               luisa::vector<float2> uvs,
               luisa::vector<uint3> indices) noexcept;
    ~InlineMesh() noexcept override = default;

public:
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return {m_vertices, m_triangles}; }
    [[nodiscard]] uint vertex_properties() const noexcept override { return m_properties; }
};

class InlineMeshShapeSpec final : public ShapeSpec
{
private:
    luisa::vector<float3> _positions;
    luisa::vector<float3> _normals;
    luisa::vector<float2> _uvs;
    luisa::vector<uint3> _indices;

public:
    InlineMeshShapeSpec(luisa::vector<float3> positions,
                        luisa::vector<float3> normals,
                        luisa::vector<float2> uvs,
                        luisa::vector<uint3> indices) noexcept
        : _positions{std::move(positions)},
          _normals{std::move(normals)},
          _uvs{std::move(uvs)},
          _indices{std::move(indices)} {}

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        if (_positions.empty() || _indices.empty())
        {
            return spec_validation_error("Inline mesh must contain vertices and triangles.");
        }
        if ((!_normals.empty() && _normals.size() != _positions.size()) || (!_uvs.empty() && _uvs.size() != _positions.size()))
        {
            return spec_validation_error("Inline mesh vertex attribute count does not match its positions.");
        }
        for (auto p : _positions)
        {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            {
                return spec_validation_error("Inline mesh position must be finite.");
            }
        }
        for (auto n : _normals)
        {
            if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z))
            {
                return spec_validation_error("Inline mesh normal must be finite.");
            }
        }
        for (auto uv : _uvs)
        {
            if (!std::isfinite(uv.x) || !std::isfinite(uv.y))
            {
                return spec_validation_error("Inline mesh texture coordinate must be finite.");
            }
        }
        for (auto triangle : _indices)
        {
            if (triangle.x >= _positions.size() || triangle.y >= _positions.size() || triangle.z >= _positions.size())
            {
                return spec_validation_error("Inline mesh triangle index is out of bounds.");
            }
        }
        return luisa::nullopt;
    }
    [[nodiscard]] const Shape* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
