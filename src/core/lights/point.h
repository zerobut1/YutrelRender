#pragma once

#include <cmath>

#include "base/light.h"
#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class PointLight final : public Light
{
public:
    class Instance;
    class Closure;

private:
    const Texture* _intensity;
    float3 _position;
    float _scale;

public:
    PointLight(const Texture* intensity, float3 position, float scale) noexcept;

    [[nodiscard]] auto intensity() const noexcept { return _intensity; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] bool is_null() const noexcept override;
    [[nodiscard]] luisa::unique_ptr<Light::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class PointLight::Instance final : public Light::Instance
{
private:
    const Texture::Instance* _intensity;

public:
    Instance(const Renderer& renderer, const PointLight* light, const Texture::Instance* intensity) noexcept
        : Light::Instance(renderer, light), _intensity{intensity} {}

    [[nodiscard]] auto intensity() const noexcept { return _intensity; }
    [[nodiscard]] luisa::unique_ptr<Light::Closure> closure(
        const SampledWavelengths& swl, Expr<float> time) const noexcept override;
};

class PointLight::Closure final : public Light::Closure
{
public:
    Closure(const Light::Instance* instance, const SampledWavelengths& swl, Expr<float> time) noexcept
        : Light::Closure(instance, swl, time) {}

    [[nodiscard]] Evaluation evaluate(
        const Interaction& it_light, const Interaction& it_from) const noexcept override;
    [[nodiscard]] Sample sample_li(
        Expr<uint> instance_id, const Interaction& it_from, Expr<float2> u) const noexcept override;
    [[nodiscard]] EmissionSample sample_le(
        Expr<uint> instance_id, Expr<float2> u_position, Expr<float2> u_direction) const noexcept override;
};

class PointLightSpec final : public LightSpec
{
private:
    TextureRef _intensity;
    float3 _position;
    float _scale;

public:
    PointLightSpec(TextureRef intensity, float3 position, float scale) noexcept
        : _intensity{intensity}, _position{position}, _scale{scale} {}

    [[nodiscard]] auto intensity() const noexcept { return _intensity; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override { visitor.visit(_intensity); }
    [[nodiscard]] const Light* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
