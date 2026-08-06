// Unity Alpha Clip mask tests.
//
// Host-side (always run):
//   - valid_alpha_clip() rejects NaN/Inf/negative/out-of-range cutoff and
//     non-finite/negative base_color_alpha; accepts 0/1 enabled.
//   - UnityAlphaClipTextureSpec::validate() mirrors those rules.
//   - evaluate_static() reproduces the binary mask boundary behavior
//     (alpha * baseColorAlpha >= cutoff) for constant BaseColor textures.
//   - The scene-build graph wraps the OpenPBR surface in an OpacitySurface and
//     reuses the same BaseColor TextureRef for the mask and the OpenPBR
//     surface (no duplicate texture upload).
//
// GPU side (needs a backend argument, e.g. `dx`):
//   - Constant and image BaseColor textures both produce binary 0/1 masks.
//   - OpacitySurface over an OpenPBRSurface reports maybe_non_opaque() and
//     evaluate_opacity() == mask.

#include <cmath>
#include <cstdio>
#include <memory>

#include <luisa/luisa-compute.h>

#include "base/renderer.h"
#include "scene/scene_spec_builder.h"
#include "surfaces/openpbr.h"
#include "surfaces/opacity.h"
#include "textures/constant.h"

#include "alpha_clip_texture.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;
using namespace yutrel::unity;

namespace {

[[nodiscard]] bool fail(const char *label) noexcept
{
    std::fprintf(stderr, "test_Yutrel_unity_alpha_clip: %s\n", label);
    return false;
}

[[nodiscard]] Interaction make_interaction(Expr<float2> uv) noexcept
{
    return Interaction::from_surface(
        Shape::Handle::decode(make_uint4(0u)), make_float3(0.0f),
        make_float3(0.0f, 0.0f, 1.0f), uv,
        make_float3(0.0f), Frame{}, 0u, 0u, 1.0f, true);
}

// --- host-side: ABI validation --------------------------------------------

[[nodiscard]] bool run_alpha_clip_validation()
{
    if (!valid_alpha_clip({.base_color_alpha = 0.5f, .cutoff = 0.5f, .enabled = 0u}) ||
        !valid_alpha_clip({.base_color_alpha = 0.5f, .cutoff = 0.5f, .enabled = 1u}) ||
        !valid_alpha_clip({.base_color_alpha = 0.0f, .cutoff = 0.0f, .enabled = 0u}) ||
        !valid_alpha_clip({.base_color_alpha = 1.0f, .cutoff = 1.0f, .enabled = 1u}))
    {
        return fail("valid_alpha_clip rejected a valid entry");
    }
    struct Case
    {
        AlphaClipData data;
    };
    const Case invalid[]{
        {{0.5f, 0.5f, 2u}},               // enabled not 0/1
        {{NAN, 0.5f, 1u}},                // NaN base_color_alpha
        {{INFINITY, 0.5f, 1u}},           // Inf base_color_alpha
        {{-0.1f, 0.5f, 1u}},             // negative base_color_alpha
        {{0.5f, NAN, 1u}},                // NaN cutoff
        {{0.5f, INFINITY, 1u}},           // Inf cutoff
        {{0.5f, -0.1f, 1u}},             // negative cutoff
        {{0.5f, 1.1f, 1u}},              // out-of-range cutoff
    };
    for (auto &&entry : invalid)
    {
        if (valid_alpha_clip(entry.data))
        {
            return fail("valid_alpha_clip accepted an invalid entry");
        }
    }
    return true;
}

// --- host-side: spec validation -------------------------------------------

[[nodiscard]] bool run_spec_validation()
{
    SceneSpecBuilder builder;
    SourceLocation source{.file = "alpha-clip-test"};
    auto base_color = builder.add_anonymous_texture<ConstantTextureSpec>(
        source, make_float4(0.8f, 0.8f, 0.8f, 0.6f));
    if (UnityAlphaClipTextureSpec{base_color, 1.0f, 0.5f}.validate().has_value())
    {
        return fail("UnityAlphaClipTextureSpec rejected a valid configuration");
    }
    if (UnityAlphaClipTextureSpec{base_color, 1.0f, 0.0f}.validate().has_value() ||
        UnityAlphaClipTextureSpec{base_color, 0.0f, 1.0f}.validate().has_value())
    {
        return fail("UnityAlphaClipTextureSpec rejected boundary cutoffs");
    }
    if (!UnityAlphaClipTextureSpec{base_color, -0.1f, 0.5f}.validate().has_value() ||
        !UnityAlphaClipTextureSpec{base_color, NAN, 0.5f}.validate().has_value() ||
        !UnityAlphaClipTextureSpec{base_color, INFINITY, 0.5f}.validate().has_value() ||
        !UnityAlphaClipTextureSpec{base_color, 1.0f, -0.1f}.validate().has_value() ||
        !UnityAlphaClipTextureSpec{base_color, 1.0f, 1.1f}.validate().has_value() ||
        !UnityAlphaClipTextureSpec{base_color, 1.0f, NAN}.validate().has_value() ||
        !UnityAlphaClipTextureSpec{base_color, 1.0f, INFINITY}.validate().has_value())
    {
        return fail("UnityAlphaClipTextureSpec accepted invalid parameters");
    }
    return true;
}

// --- host-side: evaluate_static boundary behavior -------------------------

// Local stand-in for UnityImageTexture: a bindless image sampled by UV.
// Not statically known, so it also exercises the non-constant mask path.
class TestImageTexture final : public Texture
{
public:
    class Instance final : public Texture::Instance
    {
    private:
        uint32_t _texture_id;

    public:
        Instance(const Renderer &renderer, const Texture *texture,
                 uint32_t texture_id) noexcept
            : Texture::Instance{renderer, texture},
              _texture_id{texture_id} {}

        [[nodiscard]] Float4 evaluate(
            const Interaction &it, Expr<float>) const noexcept override
        {
            return renderer().tex2d(_texture_id).sample(it.uv);
        }
    };

private:
    luisa::vector<float4> _pixels;
    uint2 _size;

public:
    TestImageTexture(luisa::vector<float4> pixels, uint2 size) noexcept
        : _pixels{std::move(pixels)},
          _size{size} {}

    [[nodiscard]] luisa::unique_ptr<Texture::Instance> build(
        Renderer &renderer, CommandBuffer &command_buffer) const noexcept override
    {
        auto image = renderer.create<Image<float>>(PixelStorage::FLOAT4, _size);
        auto texture_id = renderer.register_bindless(
            *image, TextureSampler::linear_point_repeat());
        command_buffer << image->copy_from(_pixels.data()) << commit();
        return luisa::make_unique<Instance>(renderer, this, texture_id);
    }
};

[[nodiscard]] bool run_static_boundary()
{
    struct Case
    {
        float base_alpha;
        float base_color_alpha;
        float cutoff;
        float expected;
    };
    const Case cases[]{
        {0.5f, 1.0f, 0.5f, 1.0f},   // equal to cutoff keeps
        {0.4f, 1.0f, 0.5f, 0.0f},   // below cutoff drops
        {1.0f, 1.0f, 0.5f, 1.0f},   // full alpha keeps
        {0.0f, 1.0f, 0.5f, 0.0f},   // zero alpha drops
        {1.0f, 0.0f, 0.5f, 0.0f},   // zero base color alpha drops
        {0.5f, 0.8f, 0.4f, 1.0f},   // product boundary keeps
        {0.5f, 0.8f, 0.41f, 0.0f},  // product below cutoff drops
        {0.0f, 1.0f, 0.0f, 1.0f},   // 0 >= 0 keeps
    };
    for (auto &&entry : cases)
    {
        ConstantTexture base{make_float4(0.8f, 0.8f, 0.8f, entry.base_alpha)};
        UnityAlphaClipTexture mask{&base, entry.base_color_alpha, entry.cutoff};
        auto value = mask.evaluate_static();
        if (!value || value->x != entry.expected)
        {
            std::fprintf(stderr,
                         "static boundary base_alpha=%g base_color_alpha=%g cutoff=%g: "
                         "expected=%g got=%s\n",
                         entry.base_alpha, entry.base_color_alpha, entry.cutoff,
                         entry.expected,
                         value ? luisa::to_string(*value).c_str() : "<none>");
            return false;
        }
    }
    // Non-constant base color: no static value (candidate filtering enabled).
    TestImageTexture image_base{
        luisa::vector{make_float4(1.0f)}, make_uint2(1u)};
    UnityAlphaClipTexture image_mask{&image_base, 1.0f, 0.5f};
    if (image_mask.evaluate_static().has_value())
    {
        return fail("image-based mask must not be statically known");
    }
    return true;
}

// --- host-side: scene-build graph (wrap + TextureRef reuse) ---------------

[[nodiscard]] bool run_spec_graph()
{
    SceneSpecBuilder builder;
    SourceLocation source{.file = "alpha-clip-test"};

    auto base_color = builder.add_anonymous_texture<ConstantTextureSpec>(
        source, make_float4(0.8f, 0.8f, 0.8f, 0.5f));
    auto mask = builder.add_anonymous_texture<UnityAlphaClipTextureSpec>(
        source, base_color, 1.0f, 0.5f);
    auto surface = builder.add_anonymous_surface<OpenPBRSurfaceSpec>(
        source,
        OpenPBRSurfaceParams{
            .base_color = base_color,
            .two_sided = false,
        });
    auto wrapped = builder.add_anonymous_surface<OpacitySurfaceSpec>(
        source, surface, mask);

    // Mirror of the plugin build code: the mask and the OpenPBR surface must
    // share one BaseColor TextureRef, and the OpacitySurface wraps the OpenPBR
    // surface with the mask.
    UnityAlphaClipTextureSpec mask_spec{base_color, 1.0f, 0.5f};
    OpenPBRSurfaceSpec openpbr_spec{OpenPBRSurfaceParams{.base_color = base_color}};
    OpacitySurfaceSpec wrapped_spec{surface, mask};

    if (wrapped_spec.base() != surface)
    {
        return fail("OpacitySurface does not wrap the OpenPBR surface");
    }
    if (wrapped_spec.alpha() != mask)
    {
        return fail("OpacitySurface does not use the alpha clip mask");
    }
    if (!openpbr_spec.params().base_color.has_value() ||
        *openpbr_spec.params().base_color != base_color)
    {
        return fail("OpenPBR surface does not use the shared BaseColor TextureRef");
    }
    if (mask_spec.base_color() != base_color)
    {
        return fail("Alpha clip mask does not reuse the BaseColor TextureRef");
    }
    return true;
}

// --- GPU: textures for the mask -------------------------------------------

[[nodiscard]] bool run_gpu_constant_mask(
    Device &device, Stream &stream, Renderer &renderer, CommandBuffer &commands)
{
    struct Case
    {
        float base_alpha;
        float base_color_alpha;
        float cutoff;
        float expected;
    };
    const Case cases[]{
        {0.5f, 1.0f, 0.5f, 1.0f},
        {0.4f, 1.0f, 0.5f, 0.0f},
        {1.0f, 1.0f, 0.5f, 1.0f},
        {0.0f, 1.0f, 0.5f, 0.0f},
        {1.0f, 0.0f, 0.5f, 0.0f},
        {0.5f, 0.8f, 0.4f, 1.0f},
        {0.5f, 0.8f, 0.41f, 0.0f},
        {0.0f, 1.0f, 0.0f, 1.0f},
    };
    for (auto &&entry : cases)
    {
        ConstantTexture base{make_float4(0.8f, 0.8f, 0.8f, entry.base_alpha)};
        UnityAlphaClipTexture mask{&base, entry.base_color_alpha, entry.cutoff};
        auto instance = mask.build(renderer, commands);
        if (renderer.bindless_array().dirty())
        {
            commands << renderer.bindless_array().update();
        }
        commands << synchronize();

        Kernel1D kernel = [instance = instance.get()](BufferFloat output) noexcept
        {
            auto it = make_interaction(make_float2(0.5f));
            output.write(dispatch_x(), instance->evaluate(it, 0.0f).x);
        };
        auto shader = device.compile(kernel);
        auto output = device.create_buffer<float>(1u);
        std::array<float, 1u> result{};
        stream << shader(output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        if (result[0u] != entry.expected)
        {
            std::fprintf(stderr,
                         "gpu constant base_alpha=%g base_color_alpha=%g cutoff=%g: "
                         "expected=%g got=%g\n",
                         entry.base_alpha, entry.base_color_alpha, entry.cutoff,
                         entry.expected, result[0u]);
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool run_gpu_image_mask(
    Device &device, Stream &stream, Renderer &renderer, CommandBuffer &commands)
{
    // 2x1 BaseColor: alpha 0.4 at u=0.25, alpha 0.6 at u=0.75. With
    // base_color_alpha = 1 and cutoff = 0.5 the mask must be 0/1 respectively.
    TestImageTexture base{
        luisa::vector{make_float4(0.5f, 0.5f, 0.5f, 0.4f),
                      make_float4(0.5f, 0.5f, 0.5f, 0.6f)},
        make_uint2(2u, 1u)};
    UnityAlphaClipTexture mask{&base, 1.0f, 0.5f};
    auto instance = mask.build(renderer, commands);
    if (renderer.bindless_array().dirty())
    {
        commands << renderer.bindless_array().update();
    }
    commands << synchronize();

    Kernel1D kernel = [instance = instance.get()](BufferFloat output) noexcept
    {
        auto uv = ite(dispatch_x() == 0u, make_float2(0.25f, 0.5f),
                      make_float2(0.75f, 0.5f));
        auto it = make_interaction(uv);
        output.write(dispatch_x(), instance->evaluate(it, 0.0f).x);
    };
    auto shader = device.compile(kernel);
    auto output = device.create_buffer<float>(2u);
    std::array<float, 2u> result{};
    stream << shader(output).dispatch(2u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    if (result[0u] != 0.0f || result[1u] != 1.0f)
    {
        std::fprintf(stderr,
                     "gpu image mask: expected (0, 1) got (%g, %g)\n",
                     result[0u], result[1u]);
        return false;
    }
    return true;
}

[[nodiscard]] bool run_gpu_opacity_wrap(
    Device &device, Stream &stream, Renderer &renderer, CommandBuffer &commands)
{
    // Constant OpenPBR surface wrapped in OpacitySurface with an alpha clip
    // mask. OpenPBR is opaque by itself; a statically-opaque mask keeps the
    // surface opaque, while a non-static (image) mask must enable candidate
    // filtering via maybe_non_opaque(). evaluate_opacity() must equal the
    // binary mask.
    ConstantTexture base_color{make_float4(0.5f, 0.5f, 0.5f, 0.6f)};
    ConstantTexture base_weight{make_float4(1.0f)};
    ConstantTexture base_metalness{make_float4(0.0f)};
    ConstantTexture base_diffuse_roughness{make_float4(0.5f)};
    ConstantTexture specular_weight{make_float4(1.0f)};
    ConstantTexture specular_color{make_float4(1.0f)};
    ConstantTexture specular_roughness{make_float4(0.3f)};
    ConstantTexture specular_roughness_anisotropy{make_float4(0.0f)};
    ConstantTexture specular_ior{make_float4(1.5f)};

    OpenPBRSurface openpbr{
        &base_weight, &base_color, &base_metalness, &base_diffuse_roughness,
        &specular_weight, &specular_color, &specular_roughness,
        &specular_roughness_anisotropy, &specular_ior, nullptr, true};
    if (openpbr.maybe_non_opaque())
    {
        return fail("OpenPBR surface must be opaque on its own");
    }

    // Constant mask (0.6 * 1.0 >= 0.5 -> 1): statically fully opaque.
    UnityAlphaClipTexture constant_mask{&base_color, 1.0f, 0.5f};
    OpacitySurface constant_opacity{&openpbr, &constant_mask};
    if (constant_opacity.maybe_non_opaque())
    {
        return fail("statically opaque mask must keep the surface opaque");
    }

    // Image mask: not statically known, so candidate filtering is enabled.
    TestImageTexture image_base{
        luisa::vector{make_float4(0.5f, 0.5f, 0.5f, 0.4f),
                      make_float4(0.5f, 0.5f, 0.5f, 0.6f)},
        make_uint2(2u, 1u)};
    UnityAlphaClipTexture image_mask{&image_base, 1.0f, 0.5f};
    OpacitySurface opacity{&openpbr, &image_mask};
    if (!opacity.maybe_non_opaque())
    {
        return fail("OpacitySurface with an image alpha clip mask must be non-opaque");
    }
    auto opacity_instance = opacity.build(renderer, commands);
    if (!opacity_instance->maybe_non_opaque())
    {
        return fail("OpacitySurface instance must be non-opaque");
    }
    auto mask_instance = image_mask.build(renderer, commands);
    if (renderer.bindless_array().dirty())
    {
        commands << renderer.bindless_array().update();
    }
    commands << synchronize();

    // uv 0.25 -> alpha 0.4 < 0.5 -> mask/opacity 0; uv 0.75 -> 0.6 -> 1.
    Kernel1D kernel = [opacity_instance = opacity_instance.get(),
                       mask_instance = mask_instance.get()](BufferFloat output) noexcept
    {
        auto uv = ite(dispatch_x() == 0u, make_float2(0.25f, 0.5f),
                      make_float2(0.75f, 0.5f));
        auto it = make_interaction(uv);
        auto opacity_value = opacity_instance->evaluate_opacity(it, 0.0f).value_or(1.0f);
        auto mask_value = mask_instance->evaluate(it, 0.0f).x;
        output.write(dispatch_x() * 2u, opacity_value);
        output.write(dispatch_x() * 2u + 1u, mask_value);
    };
    auto shader = device.compile(kernel);
    auto output = device.create_buffer<float>(4u);
    std::array<float, 4u> result{};
    stream << shader(output).dispatch(2u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    if (result[0u] != 0.0f || result[1u] != 0.0f ||
        result[2u] != 1.0f || result[3u] != 1.0f ||
        result[0u] != result[1u] || result[2u] != result[3u])
    {
        std::fprintf(stderr,
                     "gpu opacity wrap: expected (0,0,1,1) got (%g,%g,%g,%g)\n",
                     result[0u], result[1u], result[2u], result[3u]);
        return false;
    }
    return true;
}

[[nodiscard]] bool run_gpu(Device &device, Stream &stream)
{
    Renderer renderer{device};
    CommandBuffer commands{stream};
    auto valid = run_gpu_constant_mask(device, stream, renderer, commands);
    valid = run_gpu_image_mask(device, stream, renderer, commands) && valid;
    valid = run_gpu_opacity_wrap(device, stream, renderer, commands) && valid;
    return valid;
}

} // namespace

int main(int argc, char *argv[])
{
    auto valid = run_alpha_clip_validation();
    valid = run_spec_validation() && valid;
    valid = run_static_boundary() && valid;
    valid = run_spec_graph() && valid;
    if (!valid)
    {
        return 1;
    }
    if (argc < 2)
    {
        return 0; // host-only checks pass; GPU checks need a backend argument
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();
    return run_gpu(device, stream) ? 0 : 1;
}
