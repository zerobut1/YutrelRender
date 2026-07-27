#include "point.h"

#include "base/interaction.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/sampling.h"

namespace Yutrel
{
namespace
{
constexpr auto inv_four_pi = 0.25f / 3.14159265358979323846f;
}

PointLight::PointLight(const Texture* intensity, float3 position, float scale) noexcept
    : _intensity{intensity}, _position{position}, _scale{scale} {}

bool PointLight::is_null() const noexcept
{
    if (_scale == 0.0f) { return true; }
    if (auto value = _intensity->evaluate_static())
    {
        return value->x <= 0.0f && value->y <= 0.0f && value->z <= 0.0f;
    }
    return false;
}

luisa::unique_ptr<Light::Instance> PointLight::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto intensity = renderer.build_texture(command_buffer, _intensity);
    return luisa::make_unique<Instance>(renderer, this, intensity);
}

luisa::unique_ptr<Light::Closure> PointLight::Instance::closure(
    const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    return luisa::make_unique<Closure>(this, swl, time);
}

Light::Evaluation PointLight::Closure::evaluate(
    const Interaction& /*it_light*/, const Interaction& /*it_from*/) const noexcept
{
    return Evaluation::zero(swl().dimension());
}

Light::Sample PointLight::Closure::sample_li(
    Expr<uint> /*instance_id*/, const Interaction& it_from, Expr<float2> /*u*/) const noexcept
{
    auto light     = instance<PointLight::Instance>();
    auto position  = light->base<PointLight>()->position();
    auto d2        = distance_squared(position, it_from.p_g);
    auto valid     = d2 > 1e-12f;
    auto intensity = light->intensity()->evaluate_illuminant_spectrum(
        Interaction::from_point(position), swl(), time()).value *
                     light->base<PointLight>()->scale();
    return Sample{
        .eval = {
            .L   = ite(valid, intensity / max(d2, 1e-12f), 0.0f),
            .pdf = ite(valid, 1.0f, 0.0f),
            .p   = position,
            .ng  = make_float3(0.0f),
        },
        .p     = position,
        .delta = true,
    };
}

Light::Closure::EmissionSample PointLight::Closure::sample_le(
    Expr<uint> /*instance_id*/, Expr<float2> /*u_position*/, Expr<float2> u_direction) const noexcept
{
    auto light     = instance<PointLight::Instance>();
    auto position  = light->base<PointLight>()->position();
    auto intensity = light->intensity()->evaluate_illuminant_spectrum(
        Interaction::from_point(position), swl(), time()).value *
                     light->base<PointLight>()->scale();
    return EmissionSample{
        .Le        = intensity,
        .ray       = make_ray(position, sample_uniform_sphere(u_direction), 0.0f, 1e30f),
        .pdf       = inv_four_pi,
        .cos_theta = 1.0f,
    };
}

luisa::optional<luisa::string> PointLightSpec::validate() const noexcept
{
    auto finite_position = std::isfinite(_position.x) && std::isfinite(_position.y) && std::isfinite(_position.z);
    if (!finite_position)
    {
        return spec_validation_error("Point light position must be finite.");
    }
    if (!std::isfinite(_scale) || _scale < 0.0f)
    {
        return spec_validation_error("Point light scale must be finite and non-negative.");
    }
    return luisa::nullopt;
}

const Light* PointLightSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Light, PointLight>(builder.resolve(_intensity), _position, _scale);
}

} // namespace Yutrel
