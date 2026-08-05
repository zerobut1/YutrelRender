#pragma once

#include <utility>

#include <luisa/core/stl/vector.h>

#include "base/environment.h"
#include "scene/spec_base.h"

namespace Yutrel
{

class GroupedEnvironment final : public Environment
{
public:
    class Instance;

private:
    luisa::vector<const Environment*> _environments;

public:
    explicit GroupedEnvironment(luisa::vector<const Environment*> environments) noexcept
        : _environments{std::move(environments)} {}

    [[nodiscard]] auto& environments() const noexcept { return _environments; }
    [[nodiscard]] bool is_black() const noexcept override;
    [[nodiscard]] luisa::unique_ptr<Environment::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class GroupedEnvironment::Instance final : public Environment::Instance
{
private:
    luisa::vector<luisa::unique_ptr<Environment::Instance>> _environments;

public:
    Instance(const Renderer& renderer, const GroupedEnvironment* environment,
             luisa::vector<luisa::unique_ptr<Environment::Instance>> environments) noexcept
        : Environment::Instance{renderer, environment}, _environments{std::move(environments)} {}

    [[nodiscard]] Evaluation evaluate(
        Expr<float3> wi, const SampledWavelengths& swl,
        Expr<float> time, bool allow_incomplete_pdf) const noexcept override;
    [[nodiscard]] Sample sample(
        const SampledWavelengths& swl, Expr<float> time,
        Expr<float2> u, bool allow_incomplete_pdf) const noexcept override;
    [[nodiscard]] bool supports_external_directional_light() const noexcept override;
    void update_external_directional_light(
        CommandBuffer& command_buffer,
        const ExternalDirectionalLightState& state) noexcept override;
};

class GroupedEnvironmentSpec final : public EnvironmentSpec
{
private:
    luisa::vector<EnvironmentRef> _environments;

public:
    explicit GroupedEnvironmentSpec(luisa::vector<EnvironmentRef> environments) noexcept
        : _environments{std::move(environments)} {}

    [[nodiscard]] auto& environments() const noexcept { return _environments; }
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override;
    [[nodiscard]] const Environment* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
