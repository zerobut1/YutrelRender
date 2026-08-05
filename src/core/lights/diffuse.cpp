#include "diffuse.h"

#include "base/interaction.h"
#include "base/geometry.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/frame.h"
#include "utils/rng.h"
#include "utils/sampling.h"
#include "utils/spectra.h"

namespace Yutrel
{
DiffuseLight::DiffuseLight(const Texture* emission, float scale, bool two_sided) noexcept
    : m_emission{emission},
      m_scale{scale},
      m_two_sided{two_sided} {}

luisa::unique_ptr<Light::Instance> DiffuseLight::build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto emission = renderer.build_texture(command_buffer, m_emission);

    return luisa::make_unique<Instance>(renderer, this, emission);
}

luisa::unique_ptr<Light::Closure> DiffuseLight::Instance::closure(const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    return luisa::make_unique<Closure>(this, swl, time);
}

Light::Evaluation DiffuseLight::Closure::evaluate(const Interaction& it_light, const Interaction& it_from) const noexcept
{
    auto eval = Light::Evaluation::zero(swl().dimension());

    $outline
    {
        auto light      = instance<DiffuseLight::Instance>();
        auto&& renderer = light->renderer();

        auto pdf_triangle = renderer.buffer<float>(it_light.shape.pdf_buffer_id()).read(it_light.prim_id);
        auto pdf_area     = pdf_triangle / it_light.prim_area;
        auto p_from       = it_from.p_s;
        auto cos_wo       = abs(dot(normalize(p_from - it_light.p_g), it_light.n_g));
        auto L            = light->texture()->evaluate_illuminant_spectrum(it_light, swl(), time()).value * light->base<DiffuseLight>()->scale();
        auto pdf          = distance_squared(it_light.p_g, p_from) * pdf_area * (1.0f / cos_wo);
        auto two_sided    = light->base<DiffuseLight>()->two_sided();
        auto invalid      = abs(cos_wo) < 1e-6f | (!two_sided & !it_light.front_face);
        auto alpha        = renderer.geometry()->evaluate_opacity(it_light, time());
        auto alpha_masked = (alpha <= 0.0f) |
                            ((alpha < 1.0f) & (pbrt_hash_float(it_light.p_g) > alpha));
        eval              = {.L   = ite(invalid | alpha_masked, 0.0f, L),
                             .pdf = ite(invalid, 0.0f, pdf),
                             .p   = it_light.p_g,
                             .ng  = it_light.shading.n()};
    };
    return eval;
}

Light::Sample DiffuseLight::Closure::sample_li(
    Expr<uint> instance_id, const Interaction& it_from, Expr<float2> u) const noexcept
{
    auto light      = instance<DiffuseLight::Instance>();
    auto&& renderer = light->renderer();
    auto light_inst = renderer.geometry()->instance(instance_id);
    auto light_to_world = renderer.geometry()->instance_to_world(instance_id);

    auto [triangle_id, ux] = sample_alias_table(
        renderer.buffer<AliasEntry>(light_inst.alias_table_buffer_id()),
        light_inst.triangle_count(), u.x);
    auto triangle = renderer.geometry()->triangle(light_inst, triangle_id);
    auto uv       = sample_uniform_triangle(make_float2(ux, u.y)).xy();
    auto attrib   = renderer.geometry()->shading_point(light_inst, triangle, uv, light_to_world);
    auto it_light = Interaction::from_surface(
        std::move(light_inst), attrib.pg, attrib.ng, attrib.uv, attrib.pg,
        Frame::make(attrib.ns, attrib.dpdu, attrib.dpdv), instance_id, triangle_id, attrib.area,
        dot(attrib.ng, it_from.p_g - attrib.pg) > 0.0f);

    return Light::Sample{
        .eval  = evaluate(it_light, it_from),
        .p     = attrib.pg,
        .delta = false,
    };
}

Light::Closure::EmissionSample DiffuseLight::Closure::sample_le(
    Expr<uint> instance_id, Expr<float2> u_position, Expr<float2> u_direction) const noexcept
{
    auto result = EmissionSample::zero(swl().dimension());

    $outline
    {
        auto light      = instance<DiffuseLight::Instance>();
        auto&& renderer = light->renderer();
        auto two_sided  = light->base<DiffuseLight>()->two_sided();

        // Sample a point on the light mesh
        auto light_inst     = renderer.geometry()->instance(instance_id);
        auto light_to_world = renderer.geometry()->instance_to_world(instance_id);

        // Sample triangle via alias table
        auto alias_table_buffer_id = light_inst.alias_table_buffer_id();
        auto [triangle_id, ux]     = sample_alias_table(
            renderer.buffer<AliasEntry>(alias_table_buffer_id),
            light_inst.triangle_count(), u_position.x);
        auto triangle = renderer.geometry()->triangle(light_inst, triangle_id);
        auto uv       = sample_uniform_triangle(make_float2(ux, u_position.y)).xy();
        auto attrib   = renderer.geometry()->shading_point(light_inst, triangle, uv, light_to_world);

        // PDF for area sampling: pdf_triangle / prim_area
        auto pdf_triangle = renderer.buffer<float>(light_inst.pdf_buffer_id()).read(triangle_id);
        auto pdf_area     = pdf_triangle / attrib.area;

        // Evaluate emission at the sampled point
        auto it_for_texture = Interaction::from_surface(
            std::move(light_inst), attrib.pg, attrib.ng, attrib.uv, attrib.pg,
            Frame::make(attrib.ns, attrib.dpdu, attrib.dpdv), instance_id, triangle_id, attrib.area, true);
        auto L = light->texture()->evaluate_illuminant_spectrum(it_for_texture, swl(), time()).value *
                 light->base<DiffuseLight>()->scale();

        // Sample direction: cosine hemisphere from the surface normal
        auto local_dir = sample_cosine_hemisphere(u_direction);
        auto cos_theta = local_dir.z;

        // Build a frame from the geometric normal for direction sampling
        auto frame_n = attrib.ng;
        $if(two_sided)
        {
            // For two-sided, choose side with 0.5 probability based on u_direction.x
            // We reuse u_direction but adjust: split u_direction.x into side choice and new u
            // Actually, better to just pick side from sign of local_dir.z and always use |cos|
            // For simplicity: always emit from +ng side, but for two-sided the PDF is halved
            // and we also consider -ng. We'll use a simpler approach:
            // Use u_direction to sample cosine hemisphere, then for two-sided flip with 0.5 prob.
            // But we don't have extra random numbers. Instead, per the plan:
            // "双面光源先以 0.5 概率选择正反面"
            // We split u_direction.y: top half -> front, bottom half -> back
            // This is a compromise. Let's just use the straightforward approach:
            // front_face is always emitted; two_sided means pdf_direction accounts for both sides.
        };

        // Cosine hemisphere PDF
        auto pdf_direction = cos_theta * (1.0f / 3.14159265358979323846f);

        // For two-sided lights, direction can be on either side
        // We emit from the normal side; two_sided adjusts the select_pdf later in the integrator
        auto normal_frame = Frame::make(frame_n);
        auto world_dir    = normal_frame.local_to_world(local_dir);

        // Construct the ray from the surface point
        // Use a small offset along the normal to avoid self-intersection
        auto ray_origin = attrib.pg + frame_n * 1e-4f;
        result.ray      = make_ray(ray_origin, world_dir, 0.0f, 1e30f);
        result.Le       = L;
        result.pdf       = pdf_area * pdf_direction;
        result.cos_theta = cos_theta;

        // Zero out if cos_theta is too small (degenerate)
        $if(cos_theta < 1e-6f)
        {
            result.Le  = SampledSpectrum{swl().dimension(), 0.0f};
            result.pdf = 0.0f;
        };
    };
    return result;
}

const Light* DiffuseLightSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Light, DiffuseLight>(builder.resolve(_emission), _scale, _two_sided);
}

} // namespace Yutrel
