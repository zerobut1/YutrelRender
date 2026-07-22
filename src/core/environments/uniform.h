#pragma once

#include <cmath>

#include "base/environment.h"
#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{

class UniformEnvironment final : public Environment
{
public:
    class Instance;

private:
    const Texture* _emission;
    float _scale;

public:
    UniformEnvironment(const Texture* emission, float scale) noexcept
        : _emission{emission}, _scale{scale} {}

    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] bool is_black() const noexcept override;
    [[nodiscard]] luisa::unique_ptr<Environment::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class UniformEnvironment::Instance final : public Environment::Instance
{
private:
    const Texture::Instance* _emission;

private:
    [[nodiscard]] SampledSpectrum _evaluate_radiance(
        const SampledWavelengths& swl, Expr<float> time) const noexcept;

public:
    Instance(const Renderer& renderer, const UniformEnvironment* environment,
             const Texture::Instance* emission) noexcept
        : Environment::Instance{renderer, environment}, _emission{emission} {}

    [[nodiscard]] Evaluation evaluate(
        Expr<float3> wi, const SampledWavelengths& swl,
        Expr<float> time, bool allow_incomplete_pdf) const noexcept override;
    [[nodiscard]] Sample sample(
        const SampledWavelengths& swl, Expr<float> time,
        Expr<float2> u, bool allow_incomplete_pdf) const noexcept override;
};

class UniformEnvironmentSpec final : public EnvironmentSpec
{
private:
    TextureRef _emission;
    float _scale;

public:
    UniformEnvironmentSpec(TextureRef emission, float scale) noexcept
        : _emission{emission}, _scale{scale} {}

    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override { visitor.visit(_emission); }
    [[nodiscard]] const Environment* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
