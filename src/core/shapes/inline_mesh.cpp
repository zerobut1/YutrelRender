#include "inline_mesh.h"

#include "scene/scene_builder.h"

namespace Yutrel
{
InlineMesh::InlineMesh(luisa::vector<float3> positions,
                       luisa::vector<float3> normals,
                       luisa::vector<float2> uvs,
                       luisa::vector<uint3> indices) noexcept
{
    auto vertex_count = positions.size();

    if ((!normals.empty() && normals.size() != vertex_count) ||
        (!uvs.empty() && uvs.size() != vertex_count)) [[unlikely]]
    {
        LUISA_ERROR_WITH_LOCATION("Invalid inline mesh vertex attribute count.");
    }

    m_properties = (!uvs.empty() ? Shape::property_flag_has_vertex_uv : 0u) |
                   (!normals.empty() ? Shape::property_flag_has_vertex_normal : 0u);

    m_triangles.resize(indices.size());
    for (auto i = 0u; i < indices.size(); i++)
    {
        auto t = indices[i];
        if (t.x >= vertex_count || t.y >= vertex_count || t.z >= vertex_count) [[unlikely]]
        {
            LUISA_ERROR_WITH_LOCATION("Inline mesh triangle index out of bounds.");
        }
        m_triangles[i] = Triangle{t.x, t.y, t.z};
    }

    m_vertices.resize(vertex_count);
    for (auto i = 0u; i < vertex_count; i++)
    {
        auto p        = positions[i];
        auto n        = normals.empty() ? make_float3(0.0f, 0.0f, 1.0f) : normalize(normals[i]);
        auto uv       = uvs.empty() ? make_float2(0.0f) : uvs[i];
        m_vertices[i] = Vertex::encode(p, n, uv);
    }
}

const Shape* InlineMeshShapeSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Shape, InlineMesh>(_positions, _normals, _uvs, _indices);
}

} // namespace Yutrel
