#include "constant.h"

#include "scene/scene_builder.h"

namespace Yutrel
{
ConstantTexture::ConstantTexture(float4 value) noexcept
    : m_v{value} {}

luisa::unique_ptr<Texture::Instance> ConstantTexture::build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    return luisa::make_unique<ConstantTexture::Instance>(renderer, this);
}

Float4 ConstantTexture::Instance::evaluate(const Interaction& it, Expr<float> time) const noexcept
{
    return base<ConstantTexture>()->m_v;
}

const Texture* ConstantTextureSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Texture, ConstantTexture>(_value);
}
} // namespace Yutrel
