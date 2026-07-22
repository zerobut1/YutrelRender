#include "spectrum/hero.h"

#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/color_space.h"
#include "utils/rgb2spec.h"

namespace Yutrel
{

const Spectrum* HeroWavelengthSpectrumSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Spectrum, HeroWavelengthSpectrum>();
}

HeroWavelengthSpectrum::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const Spectrum* spectrum, uint t0) noexcept
    : Spectrum::Instance(renderer, command_buffer, spectrum),
      m_illum_d65(SPD::create_cie_d65(renderer, command_buffer)),
      m_rgb2spec_t0(t0) {}

SampledWavelengths HeroWavelengthSpectrum::Instance::sample(Expr<float> u) const noexcept
{
    constexpr auto sample_visible = [](auto u) noexcept
    {
        return clamp(
            538.0f - 138.888889f * atanh(0.85691062f - 1.82750197f * u),
            visible_wavelength_min,
            visible_wavelength_max);
    };
    constexpr auto visible_pdf = [](auto lambda) noexcept
    {
        auto c = cosh(0.0072f * (lambda - 538.0f));
        return 0.0039398042f / (c * c);
    };

    SampledWavelengths swl{base()->dimension()};
    auto inv_n = 1.0f / static_cast<float>(swl.dimension());
    for (auto i = 0u; i < swl.dimension(); i++)
    {
        auto up     = fract(u + static_cast<float>(i) * inv_n);
        auto lambda = sample_visible(up);
        swl.set_lambda(i, lambda);
        swl.set_pdf(i, visible_pdf(lambda));
    }
    return swl;
}

Spectrum::Decode HeroWavelengthSpectrum::Instance::decode_albedo(
    const SampledWavelengths& swl, Expr<float4> v) const noexcept
{
    auto spec = RGBAlbedoSpectrum{RGBSigmoidPolynomial{v.xyz()}};
    SampledSpectrum s{base()->dimension()};
    for (auto i = 0u; i < s.dimension(); i++)
    {
        s[i] = spec.sample(swl.lambda(i));
    }
    return {.value = s, .strength = v.w};
}

Spectrum::Decode HeroWavelengthSpectrum::Instance::decode_unbounded(
    const SampledWavelengths& swl, Expr<float4> v) const noexcept
{
    auto spec = RGBAlbedoSpectrum{RGBSigmoidPolynomial{v.xyz()}};
    SampledSpectrum s{base()->dimension()};
    for (auto i = 0u; i < s.dimension(); i++)
    {
        s[i] = spec.sample(swl.lambda(i));
    }
    return {.value = s, .strength = v.w};
}

Spectrum::Decode HeroWavelengthSpectrum::Instance::decode_illuminant(
    const SampledWavelengths& swl, Expr<float4> v) const noexcept
{
    auto spec = RGBIlluminantSpectrum{RGBSigmoidPolynomial{v.xyz()}, v.w, m_illum_d65};
    SampledSpectrum s{base()->dimension()};
    for (auto i = 0u; i < s.dimension(); i++)
    {
        s[i] = spec.sample(swl.lambda(i));
    }
    return {.value = s, .strength = v.w};
}

Float4 HeroWavelengthSpectrum::Instance::encode_srgb_albedo(Expr<float3> rgb) const noexcept
{
    return RGB2SpectrumTable::srgb().decode_albedo(renderer().bindless_array(), m_rgb2spec_t0, rgb);
}

Float4 HeroWavelengthSpectrum::Instance::encode_srgb_unbounded(Expr<float3> rgb) const noexcept
{
    return RGB2SpectrumTable::srgb().decode_unbounded(renderer().bindless_array(), m_rgb2spec_t0, rgb);
}

Float4 HeroWavelengthSpectrum::Instance::encode_srgb_illuminant(Expr<float3> rgb) const noexcept
{
    return RGB2SpectrumTable::srgb().decode_unbounded(renderer().bindless_array(), m_rgb2spec_t0, rgb);
}

luisa::unique_ptr<Spectrum::Instance> HeroWavelengthSpectrum::build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto rgb2spec_t0 = renderer.create<Volume<float>>(PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::s_resolution));
    auto rgb2spec_t1 = renderer.create<Volume<float>>(PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::s_resolution));
    auto rgb2spec_t2 = renderer.create<Volume<float>>(PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::s_resolution));
    RGB2SpectrumTable::srgb().encode(command_buffer, *rgb2spec_t0, *rgb2spec_t1, *rgb2spec_t2);
    auto t0 = renderer.register_bindless(*rgb2spec_t0, TextureSampler::linear_point_zero());
    auto t1 = renderer.register_bindless(*rgb2spec_t1, TextureSampler::linear_point_zero());
    auto t2 = renderer.register_bindless(*rgb2spec_t2, TextureSampler::linear_point_zero());
    LUISA_ASSERT(
        t1 == t0 + 1u && t2 == t0 + 2u,
        "Invalid RGB2Spec texture indices: "
        "{}, {}, and {}.",
        t0,
        t1,
        t2);
    return luisa::make_unique<Instance>(renderer, command_buffer, this, t0);
}

float4 HeroWavelengthSpectrum::encode_static_srgb_albedo(float3 rgb) const noexcept
{
    return RGB2SpectrumTable::srgb().decode_albedo(rgb);
}

float4 HeroWavelengthSpectrum::encode_static_srgb_unbounded(float3 rgb) const noexcept
{
    return RGB2SpectrumTable::srgb().decode_unbounded(rgb);
}

float4 HeroWavelengthSpectrum::encode_static_srgb_illuminant(float3 rgb) const noexcept
{
    return RGB2SpectrumTable::srgb().decode_unbounded(rgb);
}
} // namespace Yutrel
