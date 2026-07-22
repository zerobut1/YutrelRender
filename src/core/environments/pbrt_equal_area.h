#pragma once

#include <cmath>

#include "base/environment.h"
#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{

class PBRTEqualAreaEnvironment final : public Environment
{
public:
    class Instance;

private:
    const Texture* _emission;
    float _scale;
    float3x3 _transform_to_world;

public:
    PBRTEqualAreaEnvironment(const Texture* emission, float scale, float3x3 transform_to_world) noexcept
        : _emission{emission}, _scale{scale}, _transform_to_world{transform_to_world} {}

    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto transform_to_world() const noexcept { return _transform_to_world; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f; }
    [[nodiscard]] luisa::unique_ptr<Environment::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class PBRTEqualAreaEnvironment::Instance final : public Environment::Instance
{
private:
    const Texture::Instance* _emission;
    uint2 _resolution;
    uint _alias_buffer_id;
    uint _pdf_buffer_id;
    uint _alias_distribution_stride;
    uint _pdf_distribution_stride;

private:
    [[nodiscard]] SampledSpectrum _evaluate_radiance(
        Expr<float2> uv, const SampledWavelengths& swl, Expr<float> time) const noexcept;

public:
    Instance(const Renderer& renderer, const PBRTEqualAreaEnvironment* environment,
             const Texture::Instance* emission, uint2 resolution,
             uint alias_buffer_id, uint pdf_buffer_id,
             uint alias_distribution_stride, uint pdf_distribution_stride) noexcept
        : Environment::Instance{renderer, environment},
          _emission{emission}, _resolution{resolution},
          _alias_buffer_id{alias_buffer_id}, _pdf_buffer_id{pdf_buffer_id},
          _alias_distribution_stride{alias_distribution_stride},
          _pdf_distribution_stride{pdf_distribution_stride} {}

    [[nodiscard]] Evaluation evaluate(
        Expr<float3> wi, const SampledWavelengths& swl, Expr<float> time,
        bool allow_incomplete_pdf) const noexcept override;
    [[nodiscard]] Sample sample(
        const SampledWavelengths& swl, Expr<float> time, Expr<float2> u,
        bool allow_incomplete_pdf) const noexcept override;
};

class PBRTEqualAreaEnvironmentSpec final : public EnvironmentSpec
{
private:
    TextureRef _emission;
    float _scale;
    float3x3 _transform_to_world;

public:
    PBRTEqualAreaEnvironmentSpec(TextureRef emission, float scale, float3x3 transform_to_world) noexcept
        : _emission{emission}, _scale{scale}, _transform_to_world{transform_to_world} {}

    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto transform_to_world() const noexcept { return _transform_to_world; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override { visitor.visit(_emission); }
    [[nodiscard]] const Environment* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
