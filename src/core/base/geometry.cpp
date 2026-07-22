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
    m_accel = m_renderer.device().create_accel({});

    for (auto& instance : instances)
    {
        process_instance(command_buffer, instance);
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

void Geometry::process_instance(CommandBuffer& command_buffer, const ShapeInstance& instance) noexcept
{
    auto shape   = instance.shape;
    auto surface = instance.surface;
    auto light   = instance.light;

    if (shape->is_mesh())
    {
        auto mesh = [&]
        {
            if (auto it = m_meshes.find(shape); it != m_meshes.end())
            {
                return it->second;
            }
            auto mesh_gemo = [&]
            {
                auto [vertices, triangles] = shape->mesh();
                LUISA_ASSERT(!vertices.empty() && !triangles.empty(), "Empty mesh.");
                auto hash = luisa::hash64(vertices.data(), vertices.size_bytes(), luisa::hash64_default_seed);
                hash      = luisa::hash64(triangles.data(), triangles.size_bytes(), hash);
                if (auto mesh_it = m_mesh_cache.find(hash); mesh_it != m_mesh_cache.end())
                {
                    return mesh_it->second;
                }
                // create mesh
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
                // compute alisa table
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
                auto [alisa_table_buffer_view, alias_buffer_id] = m_renderer.bindless_arena_buffer<AliasEntry>(alias_table.size());
                auto [pdf_buffer_view, pdf_buffer_id]           = m_renderer.bindless_arena_buffer<float>(pdf.size());
                LUISA_ASSERT(triangle_buffer_id - vertex_buffer_id == Shape::Handle::triangle_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(alias_buffer_id - vertex_buffer_id == Shape::Handle::alias_table_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(pdf_buffer_id - vertex_buffer_id == Shape::Handle::pdf_buffer_id_offset, "Invalid.");
                command_buffer
                    << alisa_table_buffer_view.copy_from(alias_table.data())
                    << pdf_buffer_view.copy_from(pdf.data())
                    << commit();

                auto geom = MeshGeometry{
                    .resource       = mesh,
                    .buffer_id_base = vertex_buffer_id};
                m_mesh_cache.emplace(hash, geom);
                return geom;
            }();

            auto encode_fixed_point = [](float x) noexcept
            {
                return static_cast<uint16_t>(std::clamp(
                    std::round(x * 65535.f),
                    0.f,
                    65535.f));
            };

            MeshData data{
                .resource                = mesh_gemo.resource,
                .geometry_buffer_id_base = mesh_gemo.buffer_id_base,
                .vertex_properties       = shape->vertex_properties()};
            m_meshes.emplace(shape, data);
            return data;
        }();

        auto instance_id = static_cast<uint>(m_accel.size());

        // surfaces
        auto surface_tag = 0u;
        auto properties  = mesh.vertex_properties;
        auto maybe_non_opaque = false;
        if (surface && (!surface->is_null() || surface->maybe_non_opaque()))
        {
            surface_tag = m_renderer.register_surface(command_buffer, surface);
            if (!surface->is_null())
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

        auto inside_medium_tag  = instance.inside_medium ? m_renderer.register_medium(command_buffer, instance.inside_medium) : Medium::vacuum_tag;
        auto outside_medium_tag = instance.outside_medium ? m_renderer.register_medium(command_buffer, instance.outside_medium) : Medium::vacuum_tag;
        if (inside_medium_tag != Medium::vacuum_tag || outside_medium_tag != Medium::vacuum_tag)
        {
            properties |= Shape::property_flag_has_medium;
        }

        m_accel.emplace_back(*mesh.resource, instance.transform, 0xffu, !maybe_non_opaque);

        // lights
        auto light_tag = 0u;
        if (light && !light->is_null())
        {
            light_tag = m_renderer.register_light(command_buffer, light);
            properties |= Shape::property_flag_has_light;
        }

        m_instances.emplace_back(
            Shape::Handle::encode(
                mesh.geometry_buffer_id_base,
                properties,
                surface_tag,
                light_tag,
                0,
                mesh.resource->triangle_count(),
                0,
                0));
        m_medium_interfaces.emplace_back(make_uint2(inside_medium_tag, outside_medium_tag));

        if (properties & Shape::property_flag_has_light)
        {
            m_instanced_lights.emplace_back(
                Light::Handle{
                    .instance_id = instance_id,
                    .light_tag   = light_tag});
        }

        m_instanced_triangle_count += mesh.resource->triangle_count();
    }
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
        shading             = Frame::make(attr.ns, attr.dpdu);
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
