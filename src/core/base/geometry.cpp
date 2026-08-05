#include "geometry.h"

#include <algorithm>

#include "base/interaction.h"
#include "base/renderer.h"
#include "utils/rng.h"
#include "utils/sampling.h"

namespace Yutrel
{
void Geometry::build(CommandBuffer& command_buffer, luisa::span<const ShapeInstance> instances) noexcept
{
    m_accel = m_renderer.device().create_accel(AccelOption{
        .hint = AccelOption::UsageHint::FAST_TRACE,
        .allow_update = true,
    });

    for (auto& instance : instances)
    {
        auto prepared = prepare_instance(command_buffer, instance);
        append_instance(prepared, instance.transform);
    }
    LUISA_INFO_WITH_LOCATION("Geometry built with {} unique triangles ({} instanced).",
                             m_triangle_count,
                             m_instanced_triangle_count);

    m_instance_buffer         = m_renderer.device().create_buffer<uint4>(m_instances.size());
    m_medium_interface_buffer = m_renderer.device().create_buffer<uint2>(m_medium_interfaces.size());
    command_buffer
        << m_instance_buffer.copy_from(m_instances.data())
        << m_medium_interface_buffer.copy_from(m_medium_interfaces.data())
        << m_accel.build()
        << commit();
}

Geometry::MeshData Geometry::prepare_mesh(
    CommandBuffer& command_buffer,
    const Shape* shape) noexcept
{
    if (auto it = m_meshes.find(shape); it != m_meshes.end())
    {
        return it->second;
    }
    auto [vertices, triangles] = shape->mesh();
    LUISA_ASSERT(!vertices.empty() && !triangles.empty(), "Empty mesh.");
    auto hash = luisa::hash64(vertices.data(), vertices.size_bytes(), luisa::hash64_default_seed);
    hash      = luisa::hash64(triangles.data(), triangles.size_bytes(), hash);
    auto mesh_geometry = [&]
    {
        if (auto mesh_it = m_mesh_cache.find(hash); mesh_it != m_mesh_cache.end())
        {
            return mesh_it->second;
        }
        m_triangle_count += triangles.size();
        auto vertex_buffer   = m_renderer.create<Buffer<Vertex>>(vertices.size());
        auto triangle_buffer = m_renderer.create<Buffer<Triangle>>(triangles.size());
        auto mesh            = m_renderer.create<compute::Mesh>(*vertex_buffer, *triangle_buffer, AccelOption{});
        command_buffer
            << vertex_buffer->copy_from(vertices.data())
            << triangle_buffer->copy_from(triangles.data())
            << commit()
            << mesh->build()
            << commit();
        auto vertex_buffer_id   = m_renderer.register_bindless(vertex_buffer->view());
        auto triangle_buffer_id = m_renderer.register_bindless(triangle_buffer->view());
        luisa::vector<float> triangle_areas(triangles.size());
        for (auto i = 0u; i < triangles.size(); i++)
        {
            auto t            = triangles[i];
            auto v0           = vertices[t.i0].position();
            auto v1           = vertices[t.i1].position();
            auto v2           = vertices[t.i2].position();
            triangle_areas[i] = std::abs(length(cross(v1 - v0, v2 - v0)));
        }
        auto [alias_table, pdf]                         = create_alias_table(triangle_areas);
        auto [alias_table_buffer_view, alias_buffer_id] = m_renderer.bindless_arena_buffer<AliasEntry>(alias_table.size());
        auto [pdf_buffer_view, pdf_buffer_id]           = m_renderer.bindless_arena_buffer<float>(pdf.size());
        LUISA_ASSERT(triangle_buffer_id - vertex_buffer_id == Shape::Handle::triangle_buffer_id_offset, "Invalid.");
        LUISA_ASSERT(alias_buffer_id - vertex_buffer_id == Shape::Handle::alias_table_buffer_id_offset, "Invalid.");
        LUISA_ASSERT(pdf_buffer_id - vertex_buffer_id == Shape::Handle::pdf_buffer_id_offset, "Invalid.");
        command_buffer
            << alias_table_buffer_view.copy_from(alias_table.data())
            << pdf_buffer_view.copy_from(pdf.data())
            << commit();
        auto geometry = MeshGeometry{
            .resource       = mesh,
            .buffer_id_base = vertex_buffer_id};
        m_mesh_cache.emplace(hash, geometry);
        return geometry;
    }();

    MeshData data{
        .resource                = mesh_geometry.resource,
        .geometry_buffer_id_base = mesh_geometry.buffer_id_base,
        .vertex_properties       = shape->vertex_properties()};
    m_meshes.emplace(shape, data);
    return data;
}

Geometry::PreparedInstance Geometry::prepare_instance(
    CommandBuffer& command_buffer,
    const ShapeInstance& instance) noexcept
{
    LUISA_ASSERT(instance.shape != nullptr && instance.shape->is_mesh(), "External shape is not a mesh.");
    auto mesh = prepare_mesh(command_buffer, instance.shape);
    auto surface_tag = 0u;
    auto properties = mesh.vertex_properties;
    auto maybe_non_opaque = false;
    if (instance.surface && (!instance.surface->is_null() || instance.surface->maybe_non_opaque()))
    {
        surface_tag = m_renderer.register_surface(command_buffer, instance.surface);
        if (!instance.surface->is_null())
        {
            properties |= Shape::property_flag_has_surface;
        }
        maybe_non_opaque = m_renderer.surfaces().impl(surface_tag)->maybe_non_opaque();
        if (maybe_non_opaque)
        {
            properties |= Shape::property_flag_maybe_non_opaque;
            m_any_non_opaque = true;
            if (std::find(m_non_opaque_surface_tags.begin(),
                          m_non_opaque_surface_tags.end(),
                          surface_tag) == m_non_opaque_surface_tags.end())
            {
                m_non_opaque_surface_tags.emplace_back(surface_tag);
            }
        }
    }

    auto inside_medium_tag = instance.inside_medium
                                 ? m_renderer.register_medium(command_buffer, instance.inside_medium)
                                 : Medium::vacuum_tag;
    auto outside_medium_tag = instance.outside_medium
                                  ? m_renderer.register_medium(command_buffer, instance.outside_medium)
                                  : Medium::vacuum_tag;
    if (inside_medium_tag != Medium::vacuum_tag || outside_medium_tag != Medium::vacuum_tag)
    {
        properties |= Shape::property_flag_has_medium;
    }

    auto light_tag = 0u;
    if (instance.light && !instance.light->is_null())
    {
        light_tag = m_renderer.register_light(command_buffer, instance.light);
        properties |= Shape::property_flag_has_light;
    }
    return PreparedInstance{
        .mesh = mesh,
        .encoded = Shape::Handle::encode(
            mesh.geometry_buffer_id_base,
            properties,
            surface_tag,
            light_tag,
            0u,
            mesh.resource->triangle_count(),
            0.0f,
            0.0f),
        .medium_interface = make_uint2(inside_medium_tag, outside_medium_tag),
        .light_tag = light_tag,
        .has_light = (properties & Shape::property_flag_has_light) != 0u,
        .opaque = !maybe_non_opaque,
    };
}

void Geometry::append_instance(
    const PreparedInstance& instance,
    float4x4 transform) noexcept
{
    auto instance_id = static_cast<uint>(m_accel.size());
    m_accel.emplace_back(*instance.mesh.resource, transform, 0xffu, instance.opaque);
    m_instances.emplace_back(instance.encoded);
    m_medium_interfaces.emplace_back(instance.medium_interface);
    if (instance.has_light)
    {
        m_instanced_lights.emplace_back(Light::Handle{
            .instance_id = instance_id,
            .light_tag = instance.light_tag,
        });
    }
    m_instanced_triangle_count += instance.mesh.resource->triangle_count();
}

bool Geometry::prepare_external_updates(
    CommandBuffer& command_buffer,
    luisa::span<const uint64_t> initial_instance_ids,
    const Surface* default_surface) noexcept
{
    if (m_external_updates_enabled || default_surface == nullptr ||
        default_surface->maybe_non_opaque() ||
        initial_instance_ids.size() != m_instances.size() ||
        initial_instance_ids.size() > external_instance_capacity ||
        !m_instanced_lights.empty())
    {
        return false;
    }
    luisa::unordered_map<uint64_t, uint> slots;
    slots.reserve(initial_instance_ids.size());
    for (auto i = 0u; i < initial_instance_ids.size(); i++)
    {
        if (!slots.emplace(initial_instance_ids[i], i).second)
        {
            return false;
        }
    }

    m_external_updates_enabled = true;
    m_external_surface = default_surface;
    m_external_high_water = static_cast<uint>(initial_instance_ids.size());
    m_external_slots = std::move(slots);
    m_external_ids.assign(external_instance_capacity, 0u);
    m_external_active.assign(external_instance_capacity, 0u);
    for (auto i = 0u; i < initial_instance_ids.size(); i++)
    {
        m_external_ids[i] = initial_instance_ids[i];
        m_external_active[i] = 1u;
    }
    m_instances.resize(external_instance_capacity, make_uint4(0u));
    m_medium_interfaces.resize(external_instance_capacity, make_uint2(0u));
    m_instance_buffer = m_renderer.device().create_buffer<uint4>(external_instance_capacity);
    m_medium_interface_buffer = m_renderer.device().create_buffer<uint2>(external_instance_capacity);
    command_buffer
        << m_instance_buffer.copy_from(m_instances.data())
        << m_medium_interface_buffer.copy_from(m_medium_interfaces.data());
    return true;
}

void Geometry::upload_external_slot(CommandBuffer& command_buffer, uint slot) noexcept
{
    command_buffer
        << m_instance_buffer.view().subview(slot, 1u).copy_from(luisa::span{&m_instances[slot], 1u})
        << m_medium_interface_buffer.view().subview(slot, 1u).copy_from(luisa::span{&m_medium_interfaces[slot], 1u});
}

bool Geometry::update_external(
    CommandBuffer& command_buffer,
    luisa::span<const ExternalMeshUpdate> updates) noexcept
{
    if (!m_external_updates_enabled)
    {
        return false;
    }

    luisa::unordered_set<uint64_t> changed_ids;
    auto new_instance_count = 0u;
    auto removed_instance_count = 0u;
    for (auto& update : updates)
    {
        if (!changed_ids.emplace(update.id).second)
        {
            return false;
        }
        auto existing = m_external_slots.find(update.id);
        switch (update.operation)
        {
            case ExternalMeshOp::add_or_replace:
                if (update.shape == nullptr || !update.shape->is_mesh() ||
                    update.shape->mesh().vertices.empty() || update.shape->mesh().triangles.empty() ||
                    validate_camera_to_world(update.local_to_world))
                {
                    return false;
                }
                new_instance_count += existing == m_external_slots.end() ? 1u : 0u;
                break;
            case ExternalMeshOp::transform:
                if (existing == m_external_slots.end() ||
                    validate_camera_to_world(update.local_to_world))
                {
                    return false;
                }
                break;
            case ExternalMeshOp::remove:
                if (existing == m_external_slots.end())
                {
                    return false;
                }
                removed_instance_count++;
                break;
            default:
                return false;
        }
    }
    auto available_slots = static_cast<uint64_t>(m_external_free_slots.size()) +
                           external_instance_capacity - m_external_high_water +
                           removed_instance_count;
    if (new_instance_count > available_slots)
    {
        return false;
    }

    auto accel_dirty = false;
    for (auto& update : updates)
    {
        if (update.operation != ExternalMeshOp::remove)
        {
            continue;
        }
        auto slot = m_external_slots.at(update.id);
        m_accel.set_visibility_on_update(slot, 0u);
        m_external_slots.erase(update.id);
        m_external_ids[slot] = 0u;
        m_external_active[slot] = 0u;
        m_external_free_slots.emplace_back(slot);
        m_instances[slot] = make_uint4(0u);
        m_medium_interfaces[slot] = make_uint2(0u);
        upload_external_slot(command_buffer, slot);
        accel_dirty = true;
    }

    for (auto& update : updates)
    {
        if (update.operation != ExternalMeshOp::add_or_replace)
        {
            continue;
        }
        auto prepared = prepare_instance(
            command_buffer,
            ShapeInstance{
                .shape = update.shape,
                .surface = m_external_surface,
                .transform = update.local_to_world,
            });
        auto existing = m_external_slots.find(update.id);
        uint slot;
        if (existing != m_external_slots.end())
        {
            slot = existing->second;
            m_accel.set(slot, *prepared.mesh.resource, update.local_to_world, 0xffu, prepared.opaque);
        }
        else if (!m_external_free_slots.empty())
        {
            slot = m_external_free_slots.back();
            m_external_free_slots.pop_back();
            m_accel.set(slot, *prepared.mesh.resource, update.local_to_world, 0xffu, prepared.opaque);
            m_external_slots.emplace(update.id, slot);
        }
        else
        {
            slot = m_external_high_water++;
            m_accel.emplace_back(*prepared.mesh.resource, update.local_to_world, 0xffu, prepared.opaque);
            m_external_slots.emplace(update.id, slot);
        }
        m_external_ids[slot] = update.id;
        m_external_active[slot] = 1u;
        m_instances[slot] = prepared.encoded;
        m_medium_interfaces[slot] = prepared.medium_interface;
        upload_external_slot(command_buffer, slot);
        accel_dirty = true;
    }

    for (auto& update : updates)
    {
        if (update.operation == ExternalMeshOp::transform)
        {
            m_accel.set_transform_on_update(m_external_slots.at(update.id), update.local_to_world);
            accel_dirty = true;
        }
    }

    if (m_renderer.bindless_array().dirty())
    {
        command_buffer << m_renderer.bindless_array().update();
    }
    if (accel_dirty)
    {
        command_buffer << m_accel.build(AccelBuildRequest::PREFER_UPDATE);
    }
    return true;
}

Shape::Handle Geometry::instance(Expr<uint> index) const noexcept
{
    return Shape::Handle::decode(m_instance_buffer->read(index));
}

Float4x4 Geometry::instance_to_world(Expr<uint> index) const noexcept
{
    return m_accel->instance_transform(index);
}

Var<Triangle> Geometry::triangle(const Shape::Handle& instance, Expr<uint> index) const noexcept
{
    return m_renderer.buffer<Triangle>(instance.triangle_buffer_id()).read(index);
}

Var<TriangleHit> Geometry::trace_closest(const Var<Ray>& ray_in) const noexcept
{
    if (!m_any_non_opaque)
    {
        return m_accel->intersect(ray_in, {});
    }

    Callable trace = [this](Var<Ray> ray) noexcept
    {
        auto hit = m_accel->traverse(ray, {})
                       .on_surface_candidate([&](SurfaceCandidate& candidate) noexcept
                       {
                           $if(!alpha_skip(candidate.ray(), candidate.hit()))
                           {
                               candidate.commit();
                           };
                       })
                       .trace();
        return Var<TriangleHit>{hit.inst, hit.prim, hit.bary, hit.committed_ray_t};
    };
    return trace(ray_in);
}

luisa::shared_ptr<Interaction> Geometry::intersect(const Var<Ray>& ray) const noexcept
{
    return interaction(ray, trace_closest(ray));
}

Bool Geometry::intersect_any(const Var<Ray>& ray_in) const noexcept
{
    if (!m_any_non_opaque)
    {
        return m_accel->intersect_any(ray_in, {});
    }
    Callable trace = [this](Var<Ray> ray) noexcept
    {
        auto hit = m_accel->traverse_any(ray, {})
                       .on_surface_candidate([&](SurfaceCandidate& candidate) noexcept
                       {
                           $if(!alpha_skip(candidate.ray(), candidate.hit()))
                           {
                               candidate.commit();
                           };
                       })
                       .trace();
        return !hit->miss();
    };
    return trace(ray_in);
}

Float Geometry::evaluate_opacity(const Interaction& interaction, Expr<float> time) const noexcept
{
    if (!m_any_non_opaque)
    {
        return 1.0f;
    }
    auto opacity = def(1.0f);
    $if(interaction.shape.maybe_non_opaque())
    {
        m_renderer.surfaces().dispatch_group(
            interaction.shape.surface_tag(), m_non_opaque_surface_tags,
            [&](const Surface::Instance* surface) noexcept
            {
                opacity = surface->evaluate_opacity(interaction, time).value_or(1.0f);
            });
    };
    return opacity;
}

Bool Geometry::alpha_skip(const Var<Ray>& ray, const Var<TriangleHit>& hit) const noexcept
{
    auto it      = interaction(ray, hit);
    auto alpha   = evaluate_opacity(*it, 0.0f);
    auto sample  = pbrt_hash_float(ray->origin(), ray->direction());
    return (alpha <= 0.0f) | ((alpha < 1.0f) & (sample > alpha));
}

UInt2 Geometry::medium_interface(Expr<uint> instance_id) const noexcept
{
    return m_medium_interface_buffer->read(instance_id);
}

UInt Geometry::next_medium(const Interaction& it, Expr<float3> wi) const noexcept
{
    return select_medium_interface(medium_interface(it.inst_id), it.n_g, wi);
}

luisa::shared_ptr<Interaction> Geometry::interaction(const Var<Ray> ray, const Var<TriangleHit> hit) const noexcept
{
    auto encoded_shape = def(make_uint4(0u));
    auto p_g           = def(make_float3(0.0f));
    auto n_g           = def(make_float3(0.0f, 1.0f, 0.0f));
    auto uv            = def(make_float2(0.0f));
    auto p_s           = def(make_float3(0.0f));
    auto area          = def(0.0f);
    auto front_face    = def(false);
    auto inst_id       = def(~0u);
    auto prim_id       = def(~0u);
    Frame shading      = Frame::make(n_g);

    $if(!hit->miss())
    {
        encoded_shape       = m_instance_buffer->read(hit.inst);
        auto shape_hit      = Shape::Handle::decode(encoded_shape);
        auto local_to_world = m_accel->instance_transform(hit.inst);
        auto tri            = m_renderer.buffer<Triangle>(shape_hit.triangle_buffer_id()).read(hit.prim);
        auto attr           = shading_point(shape_hit, tri, hit.bary, local_to_world);
        p_g                 = attr.pg;
        n_g                 = attr.ng;
        uv                  = attr.uv;
        p_s                 = attr.ps;
        area                = attr.area;
        shading             = Frame::make(attr.ns, attr.dpdu, attr.dpdv);
        front_face          = dot(-ray->direction(), n_g) > 0.0f;
        inst_id             = hit.inst;
        prim_id             = hit.prim;
    };

    auto shape = Shape::Handle::decode(encoded_shape);
    auto it = Interaction::from_surface(
        std::move(shape), p_g, n_g, uv,
        p_s, // TODO: apply normal offset
        std::move(shading), inst_id, prim_id, area, front_face);
    return luisa::make_shared<Interaction>(std::move(it));
}

ShadingAttribute Geometry::shading_point(const Shape::Handle& instance, const Var<Triangle>& triangle, const Var<float2>& bary, const Var<float4x4>& shape_to_world) const noexcept
{
    auto v_buffer = instance.vertex_buffer_id();
    auto v0       = m_renderer.buffer<Vertex>(v_buffer).read(triangle.i0);
    auto v1       = m_renderer.buffer<Vertex>(v_buffer).read(triangle.i1);
    auto v2       = m_renderer.buffer<Vertex>(v_buffer).read(triangle.i2);
    // object space
    auto p0_local = v0->position();
    auto p1_local = v1->position();
    auto p2_local = v2->position();
    auto ns_local = triangle_interpolate(bary, v0->normal(), v1->normal(), v2->normal());

    // compute dpdu and dpdv
    auto uv0        = v0->uv();
    auto uv1        = v1->uv();
    auto uv2        = v2->uv();
    auto duv0       = uv1 - uv0;
    auto duv1       = uv2 - uv0;
    auto det        = duv0.x * duv1.y - duv0.y * duv1.x;
    auto inv_det    = 1.f / det;
    auto dp0_local  = p1_local - p0_local;
    auto dp1_local  = p2_local - p0_local;
    auto dpdu_local = (dp0_local * duv1.y - dp1_local * duv0.y) * inv_det;
    auto dpdv_local = (dp1_local * duv0.x - dp0_local * duv1.x) * inv_det;
    // world space
    auto m              = make_float3x3(shape_to_world);
    auto t              = make_float3(shape_to_world[3]);
    auto p              = m * triangle_interpolate(bary, p0_local, p1_local, p2_local) + t;
    auto c              = cross(m * dp0_local, m * dp1_local);
    auto area           = length(c) * .5f;
    auto ng             = normalize(c);
    auto fallback_frame = Frame::make(ng);
    auto dpdu           = ite(det == 0.f, fallback_frame.s(), m * dpdu_local);
    auto dpdv           = ite(det == 0.f, fallback_frame.t(), m * dpdv_local);
    auto mn             = transpose(inverse(m));
    auto ns             = ite(instance.has_vertex_normal(), normalize(mn * ns_local), ng);
    // Match PBRT: interpolated shading normals are authoritative.
    ng      = ite(dot(ns, ng) < 0.f, -ng, ng);
    auto uv = ite(instance.has_vertex_uv(), triangle_interpolate(bary, uv0, uv1, uv2), bary);
    return {.pg   = p,
            .ng   = ng,
            .area = area,
            .ps   = p,
            .ns   = ns,
            .dpdu = dpdu,
            .dpdv = dpdv,
            .uv   = uv};
}

} // namespace Yutrel
