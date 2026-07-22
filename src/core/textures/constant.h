#pragma once

#include <cmath>

#include "base/texture.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class ConstantTexture final : public Texture
{
public:
    class Instance final : public Texture::Instance
    {
    public:
        explicit Instance(const Renderer& renderer, const Texture* texture) noexcept
            : Texture::Instance(renderer, texture) {}
        ~Instance() noexcept override = default;

        Float4 evaluate(const Interaction& it, Expr<float> time) const noexcept override;
    };

private:
    float4 m_v;

public:
    explicit ConstantTexture(float4 value) noexcept;
    ~ConstantTexture() noexcept override = default;

public:
    [[nodiscard]] luisa::unique_ptr<Texture::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;

    luisa::optional<float4> evaluate_static() const noexcept override { return m_v; }
};

class ConstantTextureSpec final : public TextureSpec
{
private:
    luisa::float4 _value;

public:
    explicit ConstantTextureSpec(luisa::float4 value) noexcept
        : _value{value} {}

    [[nodiscard]] auto value() const noexcept { return _value; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        return std::isfinite(_value.x) && std::isfinite(_value.y) && std::isfinite(_value.z) && std::isfinite(_value.w)
                   ? luisa::nullopt
                   : spec_validation_error("Constant texture value must be finite.");
    }
    [[nodiscard]] const Texture* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
