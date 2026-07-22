#include "image.h"

#include "base/interaction.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
ImageTexture::ImageTexture(luisa::filesystem::path path, TextureSampler sampler, Encoding encoding,
                           float2 uv_scale, float2 uv_offset) noexcept
    : m_sampler{sampler},
      m_encoding{encoding},
      m_uv_scale{uv_scale},
      m_uv_offset{uv_offset}
{
    m_image = LoadedImage::load(path);
    if (!m_image) [[unlikely]]
    {
        LUISA_ERROR("Failed to load image texture from '{}'.", path.string());
    }
}

luisa::unique_ptr<Texture::Instance> ImageTexture::build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto&& image      = m_image;
    auto device_image = renderer.create<Image<float>>(image.pixel_storage(), image.size());
    auto tex_id       = renderer.register_bindless(*device_image, m_sampler);
    command_buffer << device_image->copy_from(image.pixels()) << commit();

    return luisa::make_unique<ImageTexture::Instance>(renderer, this, tex_id);
}

Float4 ImageTexture::Instance::evaluate(const Interaction& it, Expr<float> time) const noexcept
{
    auto texture = base<ImageTexture>();
    auto uv      = it.uv * texture->uv_scale() + texture->uv_offset();
    auto v       = renderer().tex2d(m_texture_id).sample(uv);

    return decode(v);
}

Float4 ImageTexture::Instance::decode(Expr<float4> rgba) const noexcept
{
    auto texture  = base<ImageTexture>();
    auto encoding = texture->encoding();

    if (encoding == Encoding::SRGB)
    {
        auto rgb    = rgba.xyz();
        auto linear = ite(
            rgb <= 0.04045f,
            rgb * (1.0f / 12.92f),
            pow((rgb + 0.055f) * (1.0f / 1.055f), 2.4f));
        return make_float4(linear, rgba.w);
    }

    return rgba;
}

const Texture* ImageTextureSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Texture, ImageTexture>(_path, _sampler, _encoding, _uv_scale, _uv_offset);
}

} // namespace Yutrel
