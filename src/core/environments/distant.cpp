#include "distant.h"

#include "base/interaction.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{

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
    auto L = _emission->evaluate_illuminant_spectrum(it, swl, time).value * environment->scale();
    return Sample{
        .eval = {
            .L = std::move(L),
            .pdf = 1.0f,
            .p = make_float3(0.0f),
            .ng = make_float3(0.0f),
        },
        .wi = environment->direction(),
        .delta = true,
    };
}

luisa::unique_ptr<Environment::Instance> DistantEnvironment::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto emission = renderer.build_texture(command_buffer, _emission);
    return luisa::make_unique<Instance>(renderer, this, emission);
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
