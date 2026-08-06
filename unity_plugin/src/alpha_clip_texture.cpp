#include "alpha_clip_texture.h"

#include <cmath>

#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace yutrel::unity {

using namespace luisa;
using namespace luisa::compute;

bool valid_alpha_clip(const AlphaClipData &data) noexcept {
    return data.enabled <= 1u &&
           std::isfinite(data.base_color_alpha) && data.base_color_alpha >= 0.0f &&
           std::isfinite(data.cutoff) && data.cutoff >= 0.0f && data.cutoff <= 1.0f;
}

UnityAlphaClipTexture::Instance::Instance(
    const Yutrel::Renderer &renderer,
    const Yutrel::Texture *texture,
    const Yutrel::Texture::Instance *base_color) noexcept
    : Yutrel::Texture::Instance{renderer, texture},
      _base_color{base_color} {}

Float4 UnityAlphaClipTexture::Instance::evaluate(
    const Yutrel::Interaction &it, Expr<float> time) const noexcept {
    auto texture = base<UnityAlphaClipTexture>();
    // The BaseColor instance applies its own UV transform and encoding
    // (sRGB decode only touches RGB, alpha is unaffected), so the mask
    // inherits the BaseColor sampling exactly.
    auto alpha = _base_color->evaluate(it, time).w * texture->base_color_alpha();
    auto mask = ite(alpha >= texture->cutoff(), 1.0f, 0.0f);
    return make_float4(mask);
}

UnityAlphaClipTexture::UnityAlphaClipTexture(
    const Yutrel::Texture *base_color,
    float base_color_alpha, float cutoff) noexcept
    : _base_color{base_color},
      _base_color_alpha{base_color_alpha},
      _cutoff{cutoff} {}

luisa::unique_ptr<Yutrel::Texture::Instance> UnityAlphaClipTexture::build(
    Yutrel::Renderer &renderer,
    Yutrel::CommandBuffer &command_buffer) const noexcept {
    // renderer.build_texture caches instances per Texture object, so the
    // BaseColor texture is uploaded exactly once even though both the OpenPBR
    // surface and the mask reference it.
    auto base_instance = renderer.build_texture(command_buffer, _base_color);
    return luisa::make_unique<Instance>(renderer, this, base_instance);
}

luisa::optional<float4> UnityAlphaClipTexture::evaluate_static() const noexcept {
    auto base = _base_color->evaluate_static();
    if (!base) {
        return luisa::nullopt;
    }
    auto alpha = base->w * _base_color_alpha;
    return make_float4(alpha >= _cutoff ? 1.0f : 0.0f);
}

UnityAlphaClipTextureSpec::UnityAlphaClipTextureSpec(
    Yutrel::TextureRef base_color,
    float base_color_alpha, float cutoff) noexcept
    : _base_color{base_color},
      _base_color_alpha{base_color_alpha},
      _cutoff{cutoff} {}

luisa::optional<luisa::string> UnityAlphaClipTextureSpec::validate() const noexcept {
    if (!std::isfinite(_base_color_alpha) || _base_color_alpha < 0.0f ||
        !std::isfinite(_cutoff) || _cutoff < 0.0f || _cutoff > 1.0f) {
        return Yutrel::spec_validation_error(
            "Unity Alpha Clip mask requires a finite non-negative base color alpha "
            "and a finite cutoff in [0, 1].");
    }
    return luisa::nullopt;
}

const Yutrel::Texture *UnityAlphaClipTextureSpec::build(
    Yutrel::SceneBuilder &builder) const noexcept {
    return builder.emplace<Yutrel::Texture, UnityAlphaClipTexture>(
        builder.resolve(_base_color), _base_color_alpha, _cutoff);
}

} // namespace yutrel::unity
