#pragma once

#include <utility>

#include "base/medium.h"
#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{

struct HomogeneousMediumParams
{
    TextureRef sigma_a;
    TextureRef sigma_s;
    float scale{1.0f};
    float g{0.0f};
};

class HomogeneousMedium final : public Medium
{
public:
    class Instance final : public Medium::Instance
    {
    private:
        const Texture::Instance* _sigma_a;
        const Texture::Instance* _sigma_s;

    public:
        Instance(const Renderer& renderer, const HomogeneousMedium* medium,
                 const Texture::Instance* sigma_a, const Texture::Instance* sigma_s) noexcept;

        [[nodiscard]] Properties properties(const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept override;
    };

private:
    const Texture* _sigma_a;
    const Texture* _sigma_s;
    float _scale;
    float _g;

public:
    HomogeneousMedium(const Texture* sigma_a, const Texture* sigma_s, float scale, float g) noexcept;

    [[nodiscard]] float scale() const noexcept { return _scale; }
    [[nodiscard]] float g() const noexcept { return _g; }
    [[nodiscard]] luisa::unique_ptr<Medium::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class HomogeneousMediumSpec final : public MediumSpec
{
private:
    HomogeneousMediumParams _params;

public:
    explicit HomogeneousMediumSpec(HomogeneousMediumParams params) noexcept
        : _params{std::move(params)} {}

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override;
    [[nodiscard]] const Medium* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
