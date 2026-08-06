#pragma once

#include <cstdint>

#include <luisa/core/stl/optional.h>

#include "base/texture.h"
#include "scene/spec_base.h"

namespace yutrel::unity {

using namespace luisa;
using namespace luisa::compute;

// Unity Alpha Clip payload, appended to SceneSubMeshData (ABI).
// The plugin converts the BaseColor texture alpha into a binary mask
// (tex.a * base_color_alpha >= cutoff) and wraps the OpenPBR surface in an
// OpacitySurface, so existing candidate filtering (`alpha_skip`) handles it.
struct AlphaClipData {
    float base_color_alpha;
    float cutoff;
    uint32_t enabled;
};

static_assert(sizeof(AlphaClipData) == 12u);

// Native validation: `enabled` must be 0/1, `base_color_alpha` finite and
// non-negative, `cutoff` finite and inside [0, 1]. Invalid data rejects the
// whole scene delta.
[[nodiscard]] bool valid_alpha_clip(const AlphaClipData &data) noexcept;

// Binary opacity mask texture: 1.0 where base_color.a * base_color_alpha >=
// cutoff, 0.0 otherwise. Reuses the BaseColor texture instance (same UV
// sampling, encoding and upload), so no duplicate texture upload happens.
class UnityAlphaClipTexture final : public Yutrel::Texture {
public:
    class Instance final : public Yutrel::Texture::Instance {
    private:
        const Yutrel::Texture::Instance *_base_color;

    public:
        Instance(const Yutrel::Renderer &renderer,
                 const Yutrel::Texture *texture,
                 const Yutrel::Texture::Instance *base_color) noexcept;
        [[nodiscard]] Float4 evaluate(const Yutrel::Interaction &it,
                                      Expr<float> time) const noexcept override;
    };

private:
    const Yutrel::Texture *_base_color;
    float _base_color_alpha;
    float _cutoff;

public:
    UnityAlphaClipTexture(const Yutrel::Texture *base_color,
                          float base_color_alpha, float cutoff) noexcept;
    [[nodiscard]] luisa::unique_ptr<Yutrel::Texture::Instance> build(
        Yutrel::Renderer &renderer,
        Yutrel::CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::optional<float4> evaluate_static() const noexcept override;
    [[nodiscard]] uint channels() const noexcept override { return 1u; }
    [[nodiscard]] auto base_color() const noexcept { return _base_color; }
    [[nodiscard]] auto base_color_alpha() const noexcept { return _base_color_alpha; }
    [[nodiscard]] auto cutoff() const noexcept { return _cutoff; }
};

class UnityAlphaClipTextureSpec final : public Yutrel::TextureSpec {
private:
    Yutrel::TextureRef _base_color;
    float _base_color_alpha;
    float _cutoff;

public:
    UnityAlphaClipTextureSpec(Yutrel::TextureRef base_color,
                              float base_color_alpha, float cutoff) noexcept;
    [[nodiscard]] auto base_color() const noexcept { return _base_color; }
    [[nodiscard]] auto base_color_alpha() const noexcept { return _base_color_alpha; }
    [[nodiscard]] auto cutoff() const noexcept { return _cutoff; }
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(Yutrel::SpecDependencyVisitor &visitor) const noexcept override
    {
        visitor.visit(_base_color);
    }
    [[nodiscard]] const Yutrel::Texture *build(Yutrel::SceneBuilder &builder) const noexcept override;
};

} // namespace yutrel::unity
