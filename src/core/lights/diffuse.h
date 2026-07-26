#pragma once

#include <cmath>

#include "base/light.h"
#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class DiffuseLight : public Light
{
public:
    class Instance;
    class Closure;

private:
    const Texture* m_emission;
    float m_scale;
    bool m_two_sided;

public:
    DiffuseLight(const Texture* emission, float scale, bool two_sided) noexcept;

    [[nodiscard]] auto scale() const noexcept { return m_scale; }
    [[nodiscard]] auto two_sided() const noexcept { return m_two_sided; }
    [[nodiscard]] bool is_null() const noexcept override { return m_scale == 0.0f; }
    [[nodiscard]] luisa::unique_ptr<Light::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class DiffuseLight::Instance : public Light::Instance
{
private:
    const Texture::Instance* m_emission;

public:
    explicit Instance(const Renderer& renderer, const DiffuseLight* light, const Texture::Instance* emission) noexcept
        : Light::Instance(renderer, light), m_emission(emission) {}

    [[nodiscard]] auto texture() const noexcept { return m_emission; }
    [[nodiscard]] luisa::unique_ptr<Light::Closure> closure(const SampledWavelengths& swl, Expr<float> time) const noexcept override;
};

class DiffuseLight::Closure : public Light::Closure
{
public:
    Closure(const Light::Instance* instance, const SampledWavelengths& swl, Expr<float> time) noexcept
        : Light::Closure(instance, swl, time) {}

    [[nodiscard]] Evaluation evaluate(const Interaction& it_light, const Interaction& it_from) const noexcept override;
    [[nodiscard]] EmissionSample sample_le(
        Expr<uint> instance_id, Expr<float2> u_position, Expr<float2> u_direction) const noexcept override;
};

class DiffuseLightSpec final : public LightSpec
{
private:
    TextureRef _emission;
    float _scale;
    bool _two_sided;

public:
    DiffuseLightSpec(TextureRef emission, float scale, bool two_sided) noexcept
        : _emission{emission}, _scale{scale}, _two_sided{two_sided} {}

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        return std::isfinite(_scale) && _scale >= 0.0f
                   ? luisa::nullopt
                   : spec_validation_error("Diffuse light scale must be finite and non-negative.");
    }
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override
    {
        visitor.visit(_emission);
    }

    [[nodiscard]] const Light* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
