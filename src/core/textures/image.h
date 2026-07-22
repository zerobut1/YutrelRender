#pragma once

#include <utility>

#include "base/texture.h"
#include "scene/spec_base.h"
#include "utils/image_io.h"

namespace Yutrel
{
class ImageTexture final : public Texture
{
public:
    class Instance final : public Texture::Instance
    {
    private:
        uint m_texture_id;

    public:
        explicit Instance(const Renderer& renderer, const Texture* texture, uint texture_id) noexcept
            : Texture::Instance(renderer, texture), m_texture_id(texture_id) {}
        ~Instance() noexcept override = default;

        [[nodiscard]] Float4 evaluate(const Interaction& it, Expr<float> time) const noexcept override;

        [[nodiscard]] Float4 decode(Expr<float4> rgba) const noexcept;
    };

private:
    LoadedImage m_image;
    TextureSampler m_sampler;
    Encoding m_encoding;
    float2 m_uv_scale;
    float2 m_uv_offset;

public:
    ImageTexture(luisa::filesystem::path path, TextureSampler sampler, Encoding encoding,
                 float2 uv_scale = make_float2(1.0f), float2 uv_offset = make_float2(0.0f)) noexcept;
    ~ImageTexture() noexcept override = default;

public:
    [[nodiscard]] luisa::unique_ptr<Texture::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
    [[nodiscard]] uint channels() const noexcept override { return m_image.channels(); }
    [[nodiscard]] uint2 resolution() const noexcept override { return m_image.size(); }

    [[nodiscard]] auto encoding() const noexcept { return m_encoding; }
    [[nodiscard]] auto uv_scale() const noexcept { return m_uv_scale; }
    [[nodiscard]] auto uv_offset() const noexcept { return m_uv_offset; }
};

class ImageTextureSpec final : public TextureSpec
{
private:
    luisa::filesystem::path _path;
    TextureSampler _sampler;
    Texture::Encoding _encoding;
    float2 _uv_scale;
    float2 _uv_offset;

public:
    ImageTextureSpec(luisa::filesystem::path path, TextureSampler sampler, Texture::Encoding encoding,
                     float2 uv_scale = make_float2(1.0f), float2 uv_offset = make_float2(0.0f)) noexcept
        : _path{std::move(path)}, _sampler{sampler}, _encoding{encoding},
          _uv_scale{uv_scale}, _uv_offset{uv_offset} {}

    [[nodiscard]] const auto& path() const noexcept { return _path; }
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto encoding() const noexcept { return _encoding; }
    [[nodiscard]] auto uv_scale() const noexcept { return _uv_scale; }
    [[nodiscard]] auto uv_offset() const noexcept { return _uv_offset; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        if (_path.empty())
        {
            return spec_validation_error("Image texture path cannot be empty.");
        }
        std::error_code error;
        if (!std::filesystem::is_regular_file(_path, error))
        {
            return spec_validation_error("Image texture path must reference a regular file.");
        }
        return luisa::nullopt;
    }
    [[nodiscard]] const Texture* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
