#include "checker_board.h"

#include "base/interaction.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
CheckerBoard::CheckerBoard(float2 uv_scale, const Texture* tex1, const Texture* tex2) noexcept
    : m_uv_scale{uv_scale},
      m_tex1{tex1},
      m_tex2{tex2} {}

luisa::unique_ptr<Texture::Instance> CheckerBoard::build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto tex1 = renderer.build_texture(command_buffer, m_tex1);
    auto tex2 = renderer.build_texture(command_buffer, m_tex2);
    return luisa::make_unique<CheckerBoard::Instance>(renderer, this, tex1, tex2);
}

Float4 CheckerBoard::Instance::evaluate(const Interaction& it, Expr<float> time) const noexcept
{
    auto uv       = it.uv * base<CheckerBoard>()->uv_scale();
    auto u        = static_cast<Int>(floor(uv.x));
    auto v        = static_cast<Int>(floor(uv.y));
    auto use_tex2 = (u + v) % 2 != 0;
    return ite(use_tex2, m_tex2->evaluate(it, time), m_tex1->evaluate(it, time));
}

const Texture* CheckerBoardTextureSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Texture, CheckerBoard>(_uv_scale, builder.resolve(_tex1), builder.resolve(_tex2));
}
} // namespace Yutrel
