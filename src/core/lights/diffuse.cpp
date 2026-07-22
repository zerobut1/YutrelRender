#include "diffuse.h"

#include "base/interaction.h"
#include "base/geometry.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/rng.h"
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

const Light* DiffuseLightSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Light, DiffuseLight>(builder.resolve(_emission), _scale, _two_sided);
}

} // namespace Yutrel
