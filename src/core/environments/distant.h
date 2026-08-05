#pragma once

#include <cmath>

#include "base/environment.h"
#include "base/external_scene.h"
#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{

class DistantEnvironment final : public Environment
{
public:
    class Instance;

private:
    const Texture* _emission;
    float _scale;
    float3 _direction;

public:
    DistantEnvironment(const Texture* emission, float scale, float3 direction) noexcept
        : _emission{emission}, _scale{scale}, _direction{direction} {}

    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto direction() const noexcept { return _direction; }
    [[nodiscard]] bool is_black() const noexcept override;
    [[nodiscard]] luisa::unique_ptr<Environment::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class DistantEnvironment::Instance final : public Environment::Instance
{
private:
    const Texture::Instance* _emission;
    Buffer<float4> _external_state;

public:
    Instance(Renderer& renderer, CommandBuffer& command_buffer,
             const DistantEnvironment* environment,
             const Texture::Instance* emission) noexcept;

    [[nodiscard]] bool supports_external_directional_light() const noexcept override
    {
        return true;
    }

    void update_external_directional_light(
        CommandBuffer& command_buffer,
        const ExternalDirectionalLightState& state) noexcept override;

    [[nodiscard]] Evaluation evaluate(
        Expr<float3> wi, const SampledWavelengths& swl,
        Expr<float> time, bool allow_incomplete_pdf) const noexcept override;
    [[nodiscard]] Sample sample(
        const SampledWavelengths& swl, Expr<float> time,
        Expr<float2> u, bool allow_incomplete_pdf) const noexcept override;
};

class DistantEnvironmentSpec final : public EnvironmentSpec
{
private:
    TextureRef _emission;
    float _scale;
    float3 _direction;

public:
    DistantEnvironmentSpec(TextureRef emission, float scale, float3 direction) noexcept
        : _emission{emission}, _scale{scale}, _direction{direction} {}

    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto direction() const noexcept { return _direction; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override { visitor.visit(_emission); }
    [[nodiscard]] const Environment* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
