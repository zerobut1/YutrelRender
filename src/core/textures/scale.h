#pragma once

#include <cmath>

#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class ScaleTexture final : public Texture
{
public:
    class Instance final : public Texture::Instance
    {
    private:
        const Texture::Instance* _base;

    public:
        Instance(const Renderer& renderer, const Texture* texture, const Texture::Instance* base) noexcept
            : Texture::Instance{renderer, texture}, _base{base} {}

        [[nodiscard]] Float4 evaluate(const Interaction& it, Expr<float> time) const noexcept override;
    };

private:
    const Texture* _base;
    float4 _scale;
    float4 _offset;

public:
    ScaleTexture(const Texture* base, float4 scale, float4 offset) noexcept
        : _base{base}, _scale{scale}, _offset{offset} {}

    [[nodiscard]] luisa::unique_ptr<Texture::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
    [[nodiscard]] luisa::optional<float4> evaluate_static() const noexcept override;
    [[nodiscard]] uint channels() const noexcept override { return _base->channels(); }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
};

class ScaleTextureSpec final : public TextureSpec
{
private:
    TextureRef _base;
    float4 _scale;
    float4 _offset;

public:
    ScaleTextureSpec(TextureRef base, float4 scale, float4 offset) noexcept
        : _base{base}, _scale{scale}, _offset{offset} {}

    [[nodiscard]] auto base() const noexcept { return _base; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override { visitor.visit(_base); }
    [[nodiscard]] const Texture* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
