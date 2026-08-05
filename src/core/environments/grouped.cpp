#include "grouped.h"

#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{

bool GroupedEnvironment::is_black() const noexcept
{
    for (auto environment : _environments)
    {
        if (!environment->is_black())
        {
            return false;
        }
    }
    return true;
}

Environment::Evaluation GroupedEnvironment::Instance::evaluate(
    Expr<float3> wi, const SampledWavelengths& swl, Expr<float> time,
    bool allow_incomplete_pdf) const noexcept
{
    auto result = Evaluation::zero(swl.dimension());
    auto probability = 1.0f / static_cast<float>(_environments.size());
    for (auto&& environment : _environments)
    {
        auto evaluation = environment->evaluate(wi, swl, time, allow_incomplete_pdf);
        result.L += evaluation.L;
        result.pdf += probability * evaluation.pdf;
    }
    return result;
}

Environment::Sample GroupedEnvironment::Instance::sample(
    const SampledWavelengths& swl, Expr<float> time, Expr<float2> u_in,
    bool allow_incomplete_pdf) const noexcept
{
    auto u = make_float2(u_in);
    auto count = static_cast<float>(_environments.size());
    auto sample_id = clamp(
        cast<int>(floor(u.x * count)),
        0,
        static_cast<int>(_environments.size() - 1u));
    u.x = fract(u.x * count);

    auto result = Sample::zero(swl.dimension());
    $switch (sample_id)
    {
        for (auto i = 0u; i < _environments.size(); i++)
        {
            $case (i)
            {
                result = _environments[i]->sample(swl, time, u, allow_incomplete_pdf);
            };
        }
        $default { unreachable(); };
    };

    $if (!result.delta)
    {
        for (auto i = 0u; i < _environments.size(); i++)
        {
            $if (sample_id != static_cast<int>(i))
            {
                auto evaluation = _environments[i]->evaluate(
                    result.wi, swl, time, allow_incomplete_pdf);
                result.eval.L += evaluation.L;
                result.eval.pdf += evaluation.pdf;
            };
        }
    };
    result.eval.pdf *= 1.0f / count;
    return result;
}

bool GroupedEnvironment::Instance::supports_external_directional_light() const noexcept
{
    for (auto&& environment : _environments)
    {
        if (environment->supports_external_directional_light())
        {
            return true;
        }
    }
    return false;
}

void GroupedEnvironment::Instance::update_external_directional_light(
    CommandBuffer& command_buffer,
    const ExternalDirectionalLightState& state) noexcept
{
    for (auto&& environment : _environments)
    {
        if (environment->supports_external_directional_light())
        {
            environment->update_external_directional_light(command_buffer, state);
            return;
        }
    }
}

luisa::unique_ptr<Environment::Instance> GroupedEnvironment::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    luisa::vector<luisa::unique_ptr<Environment::Instance>> environments;
    environments.reserve(_environments.size());
    for (auto environment : _environments)
    {
        if (!environment->is_black())
        {
            environments.emplace_back(environment->build(renderer, command_buffer));
        }
    }
    LUISA_ASSERT(!environments.empty(), "Cannot build a black grouped environment.");
    return luisa::make_unique<Instance>(renderer, this, std::move(environments));
}

luisa::optional<luisa::string> GroupedEnvironmentSpec::validate() const noexcept
{
    if (_environments.empty())
    {
        return spec_validation_error("Grouped environment must contain at least one child environment.");
    }
    return luisa::nullopt;
}

void GroupedEnvironmentSpec::visit_dependencies(SpecDependencyVisitor& visitor) const noexcept
{
    for (auto environment : _environments)
    {
        visitor.visit(environment);
    }
}

const Environment* GroupedEnvironmentSpec::build(SceneBuilder& builder) const noexcept
{
    luisa::vector<const Environment*> environments;
    environments.reserve(_environments.size());
    for (auto environment : _environments)
    {
        environments.emplace_back(builder.resolve(environment));
    }
    return builder.emplace<Environment, GroupedEnvironment>(std::move(environments));
}

} // namespace Yutrel
