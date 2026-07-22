#include "uniform.h"

#include "base/interaction.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/sampling.h"

namespace Yutrel
{
namespace
{
constexpr auto inv_four_pi = 0.25f * inv_pi;
}

bool UniformEnvironment::is_black() const noexcept
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

SampledSpectrum UniformEnvironment::Instance::_evaluate_radiance(
    const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    Interaction it{};
    return _emission->evaluate_illuminant_spectrum(it, swl, time).value *
           base<UniformEnvironment>()->scale();
}

Environment::Evaluation UniformEnvironment::Instance::evaluate(
    Expr<float3> wi, const SampledWavelengths& swl, Expr<float> time,
    bool allow_incomplete_pdf) const noexcept
{
    static_cast<void>(wi);
    static_cast<void>(allow_incomplete_pdf);
    return {
        .L = _evaluate_radiance(swl, time),
        .pdf = inv_four_pi,
        .p = make_float3(0.0f),
        .ng = make_float3(0.0f),
    };
}

Environment::Sample UniformEnvironment::Instance::sample(
    const SampledWavelengths& swl, Expr<float> time, Expr<float2> u,
    bool allow_incomplete_pdf) const noexcept
{
    static_cast<void>(allow_incomplete_pdf);
    return {
        .eval = {
            .L = _evaluate_radiance(swl, time),
            .pdf = inv_four_pi,
            .p = make_float3(0.0f),
            .ng = make_float3(0.0f),
        },
        .wi = sample_uniform_sphere(u),
        .delta = false,
    };
}

luisa::unique_ptr<Environment::Instance> UniformEnvironment::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto emission = renderer.build_texture(command_buffer, _emission);
    return luisa::make_unique<Instance>(renderer, this, emission);
}

luisa::optional<luisa::string> UniformEnvironmentSpec::validate() const noexcept
{
    if (!std::isfinite(_scale) || _scale < 0.0f)
    {
        return spec_validation_error("Uniform environment scale must be finite and non-negative.");
    }
    return luisa::nullopt;
}

const Environment* UniformEnvironmentSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Environment, UniformEnvironment>(
        builder.resolve(_emission), _scale);
}

} // namespace Yutrel
