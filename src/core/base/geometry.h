#pragma once

#include <luisa/dsl/syntax.h>
#include <luisa/runtime/rtx/accel.h>

#include "base/interaction.h"
#include "base/external_scene.h"
#include "base/light.h"
#include "base/shape.h"
#include "utils/command_buffer.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;
class Shape;

[[nodiscard]] inline UInt select_medium_interface(
    Expr<uint2> medium_interface, Expr<float3> n_g, Expr<float3> wi) noexcept
{
    return ite(dot(wi, n_g) > 0.0f, medium_interface.y, medium_interface.x);
}

class Geometry
{
public:
    static constexpr uint external_instance_capacity = 4096u;

    struct MeshData
    {
        Mesh* resource;
        uint geometry_buffer_id_base : 22;
        uint vertex_properties : 10;
    };

    struct MeshGeometry
    {
        Mesh* resource;
        uint buffer_id_base;
    };

    struct PreparedInstance
    {
        MeshData mesh;
        uint4 encoded;
        uint2 medium_interface;
        uint light_tag;
        bool has_light;
        bool opaque;
    };

private:
    Renderer& m_renderer;
    Accel m_accel;
    uint m_triangle_count{0u};
    uint m_instanced_triangle_count{0u};
    luisa::unordered_map<const Shape*, MeshData> m_meshes;
    luisa::unordered_map<uint64_t, MeshGeometry> m_mesh_cache;
    luisa::vector<uint4> m_instances;
    Buffer<uint4> m_instance_buffer;
    luisa::vector<uint2> m_medium_interfaces;
    Buffer<uint2> m_medium_interface_buffer;
    luisa::vector<Light::Handle> m_instanced_lights;
    luisa::vector<uint> m_non_opaque_surface_tags;
    bool m_any_non_opaque{false};
    bool m_external_updates_enabled{false};
    uint m_external_high_water{};
    const Surface* m_external_surface{};
    luisa::unordered_map<uint64_t, uint> m_external_slots;
    luisa::vector<uint64_t> m_external_ids;
    luisa::vector<uint> m_external_free_slots;
    luisa::vector<uint8_t> m_external_active;

public:
    explicit Geometry(Renderer& renderer) noexcept
        : m_renderer{renderer} {}

    void build(CommandBuffer& command_buffer, luisa::span<const ShapeInstance> instances) noexcept;
    [[nodiscard]] bool prepare_external_updates(
        CommandBuffer& command_buffer,
        luisa::span<const uint64_t> initial_instance_ids,
        const Surface* default_surface) noexcept;
    [[nodiscard]] bool update_external(
        CommandBuffer& command_buffer,
        luisa::span<const ExternalMeshUpdate> updates) noexcept;

    [[nodiscard]] auto instances() const noexcept { return luisa::span{m_instances}; }
    [[nodiscard]] auto light_instances() const noexcept { return luisa::span{m_instanced_lights}; }
    [[nodiscard]] Shape::Handle instance(Expr<uint> index) const noexcept;
    [[nodiscard]] Float4x4 instance_to_world(Expr<uint> index) const noexcept;
    [[nodiscard]] Var<Triangle> triangle(const Shape::Handle& instance, Expr<uint> index) const noexcept;
    [[nodiscard]] Var<TriangleHit> trace_closest(const Var<Ray>& ray_in) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(const Var<Ray> ray, const Var<TriangleHit> hit) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> intersect(const Var<Ray>& ray) const noexcept;
    [[nodiscard]] Bool intersect_any(const Var<Ray>& ray_in) const noexcept;
    [[nodiscard]] Float evaluate_opacity(const Interaction& interaction, Expr<float> time) const noexcept;
    [[nodiscard]] UInt2 medium_interface(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] UInt next_medium(const Interaction& it, Expr<float3> wi) const noexcept;
    [[nodiscard]] ShadingAttribute shading_point(const Shape::Handle& instance, const Var<Triangle>& triangle, const Var<float2>& bary, const Var<float4x4>& shape_to_world) const noexcept;

private:
    [[nodiscard]] MeshData prepare_mesh(CommandBuffer& command_buffer, const Shape* shape) noexcept;
    [[nodiscard]] PreparedInstance prepare_instance(CommandBuffer& command_buffer, const ShapeInstance& instance) noexcept;
    void append_instance(const PreparedInstance& instance, float4x4 transform) noexcept;
    void upload_external_slot(CommandBuffer& command_buffer, uint slot) noexcept;
    [[nodiscard]] Bool alpha_skip(const Var<Ray>& ray, const Var<TriangleHit>& hit) const noexcept;
};
} // namespace Yutrel
