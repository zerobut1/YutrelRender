#include "scale.h"

#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
luisa::unique_ptr<Texture::Instance> ScaleTexture::build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto base = renderer.build_texture(command_buffer, _base);
    return luisa::make_unique<ScaleTexture::Instance>(renderer, this, base);
}

Float4 ScaleTexture::Instance::evaluate(const Interaction& it, Expr<float> time) const noexcept
{
    auto texture = base<ScaleTexture>();
    return _base->evaluate(it, time) * texture->scale() + texture->offset();
}

luisa::optional<float4> ScaleTexture::evaluate_static() const noexcept
{
    if (auto value = _base->evaluate_static())
    {
        return *value * _scale + _offset;
    }
    return luisa::nullopt;
}

luisa::optional<luisa::string> ScaleTextureSpec::validate() const noexcept
{
    auto finite = [](float4 v) noexcept
    { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w); };
    return finite(_scale) && finite(_offset)
               ? luisa::nullopt
               : spec_validation_error("Scale texture scale and offset must be finite.");
}

const Texture* ScaleTextureSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Texture, ScaleTexture>(builder.resolve(_base), _scale, _offset);
}
} // namespace Yutrel
