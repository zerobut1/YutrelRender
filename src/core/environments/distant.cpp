#include "distant.h"

#include "base/interaction.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{

DistantEnvironment::Instance::Instance(
    Renderer& renderer,
    CommandBuffer& command_buffer,
    const DistantEnvironment* environment,
    const Texture::Instance* emission) noexcept
    : Environment::Instance{renderer, environment},
      _emission{emission},
      _external_state{renderer.device().create_buffer<float4>(3u)}
{
    auto initial_state = std::array{
        make_float4(1.0f, 1.0f, 1.0f, 1.0f),
        make_float4(environment->direction(), 1.0f),
        make_float4(0.0f),
    };
    command_buffer
        << _external_state.copy_from(luisa::span{initial_state})
        << commit();
}

void DistantEnvironment::Instance::update_external_directional_light(
    CommandBuffer& command_buffer,
    const ExternalDirectionalLightState& state) noexcept
{
    auto data = std::array{
        make_float4(state.color, state.illuminance_lux),
        make_float4(state.direction, state.enabled != 0u ? 1.0f : 0.0f),
        make_float4(1.0f),
    };
    command_buffer
        << _external_state.copy_from(luisa::span{data})
        << commit();
}

bool DistantEnvironment::is_black() const noexcept
{
    if (_scale == 0.0f)
    {
        return true;
    }
    if (auto value = _emission->evaluate_static())
    {
        return value->x <= 0.0f && value->y <= 0.0f && value->z <= 0.0f;
    }
    return false;
}

Environment::Evaluation DistantEnvironment::Instance::evaluate(
    Expr<float3> wi, const SampledWavelengths& swl, Expr<float> time,
    bool allow_incomplete_pdf) const noexcept
{
    static_cast<void>(wi);
    static_cast<void>(time);
    static_cast<void>(allow_incomplete_pdf);
    return Evaluation::zero(swl.dimension());
}

Environment::Sample DistantEnvironment::Instance::sample(
    const SampledWavelengths& swl, Expr<float> time, Expr<float2> u,
    bool allow_incomplete_pdf) const noexcept
{
    static_cast<void>(u);
    static_cast<void>(allow_incomplete_pdf);
    Interaction it{};
    auto environment = base<DistantEnvironment>();
    auto color_and_illuminance = _external_state->read(0u);
    auto direction_and_enabled = _external_state->read(1u);
    auto external_override = _external_state->read(2u).x > 0.5f;
    auto encoded_color = renderer().spectrum()->encode_srgb_illuminant(
        max(color_and_illuminance.xyz(), 0.0f));
    auto color = renderer().spectrum()->decode_illuminant(swl, encoded_color).value;
    auto standalone = _emission->evaluate_illuminant_spectrum(it, swl, time).value *
                      environment->scale();
    auto external = color * color_and_illuminance.w;
    auto L = ite(external_override, external, standalone) * direction_and_enabled.w;
    return Sample{
        .eval = {
            .L = std::move(L),
            .pdf = 1.0f,
            .p = make_float3(0.0f),
            .ng = make_float3(0.0f),
        },
        .wi = direction_and_enabled.xyz(),
        .delta = true,
    };
}

luisa::unique_ptr<Environment::Instance> DistantEnvironment::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto emission = renderer.build_texture(command_buffer, _emission);
    return luisa::make_unique<Instance>(renderer, command_buffer, this, emission);
}

luisa::optional<luisa::string> DistantEnvironmentSpec::validate() const noexcept
{
    if (!std::isfinite(_scale) || _scale < 0.0f)
    {
        return spec_validation_error("Distant environment scale must be finite and non-negative.");
    }
    auto length_squared = dot(_direction, _direction);
    if (!std::isfinite(_direction.x) || !std::isfinite(_direction.y) ||
        !std::isfinite(_direction.z) || !std::isfinite(length_squared) ||
        std::abs(length_squared - 1.0f) > 1e-4f)
    {
        return spec_validation_error("Distant environment direction must be finite and normalized.");
    }
    return luisa::nullopt;
}

const Environment* DistantEnvironmentSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Environment, DistantEnvironment>(
        builder.resolve(_emission), _scale, _direction);
}

} // namespace Yutrel
