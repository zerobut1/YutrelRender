#pragma once

#include <cmath>

#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class CheckerBoard final : public Texture
{
public:
    class Instance final : public Texture::Instance
    {
    private:
        const Texture::Instance* m_tex1;
        const Texture::Instance* m_tex2;

    public:
        explicit Instance(const Renderer& renderer, const Texture* texture,
                          const Texture::Instance* tex1, const Texture::Instance* tex2) noexcept
            : Texture::Instance(renderer, texture), m_tex1(tex1), m_tex2(tex2) {}
        ~Instance() noexcept override = default;

        Float4 evaluate(const Interaction& it, Expr<float> time) const noexcept override;
    };

private:
    float2 m_uv_scale;
    const Texture* m_tex1;
    const Texture* m_tex2;

public:
    CheckerBoard(float2 uv_scale, const Texture* tex1, const Texture* tex2) noexcept;
    ~CheckerBoard() noexcept override = default;

public:
    [[nodiscard]] luisa::unique_ptr<Texture::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
    [[nodiscard]] auto uv_scale() const noexcept { return m_uv_scale; }
};

class CheckerBoardTextureSpec final : public TextureSpec
{
private:
    float2 _uv_scale;
    TextureRef _tex1;
    TextureRef _tex2;

public:
    CheckerBoardTextureSpec(float2 uv_scale, TextureRef tex1, TextureRef tex2) noexcept
        : _uv_scale{uv_scale}, _tex1{tex1}, _tex2{tex2} {}

    [[nodiscard]] auto uv_scale() const noexcept { return _uv_scale; }
    [[nodiscard]] auto tex1() const noexcept { return _tex1; }
    [[nodiscard]] auto tex2() const noexcept { return _tex2; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        return std::isfinite(_uv_scale.x) && std::isfinite(_uv_scale.y)
                   ? luisa::nullopt
                   : spec_validation_error("Checkerboard texture UV scale must be finite.");
    }
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override
    {
        visitor.visit(_tex1);
        visitor.visit(_tex2);
    }

    [[nodiscard]] const Texture* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
