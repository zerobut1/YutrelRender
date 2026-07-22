#include "texture.h"

#include "base/renderer.h"

namespace Yutrel
{
[[nodiscard]] inline auto extend_color_to_rgb(auto color, uint n) noexcept
{
    if (n == 1u)
    {
        return color.xxx();
    }
    if (n == 2u)
    {
        return make_float3(color.xy(), 1.f);
    }
    return color;
}

Spectrum::Decode Texture::Instance::evaluate_albedo_spectrum(
    const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    // skip the expensive encoding/decoding if the texture is static
    if (auto v = base()->evaluate_static())
    {
        return evaluate_static_albedo_spectrum_impl(swl, *v);
    }
    // we have got no luck, do the expensive encoding/decoding
    auto v = evaluate(it, time);
    v      = renderer().spectrum()->encode_srgb_albedo(
        extend_color_to_rgb(v.xyz(), base()->channels()));
    return renderer().spectrum()->decode_albedo(swl, v);
}

Spectrum::Decode Texture::Instance::evaluate_unbounded_spectrum(
    const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    // skip the expensive encoding/decoding if the texture is static
    if (auto v = base()->evaluate_static())
    {
        return evaluate_static_unbounded_spectrum_impl(swl, *v);
    }
    // we have got no luck, do the expensive encoding/decoding
    auto v = evaluate(it, time);
    v      = renderer().spectrum()->encode_srgb_unbounded(
        extend_color_to_rgb(v.xyz(), base()->channels()));
    return renderer().spectrum()->decode_unbounded(swl, v);
}

Spectrum::Decode Texture::Instance::evaluate_illuminant_spectrum(
    const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    // skip the expensive encoding/decoding if the texture is static
    if (auto v = base()->evaluate_static())
    {
        return evaluate_static_illuminant_spectrum_impl(swl, *v);
    }
    // we have got no luck, do the expensive encoding/decoding
    auto v = evaluate(it, time);
    v      = renderer().spectrum()->encode_srgb_illuminant(
        extend_color_to_rgb(v.xyz(), base()->channels()));
    return renderer().spectrum()->decode_illuminant(swl, v);
}

Spectrum::Decode Texture::Instance::evaluate_static_albedo_spectrum_impl(
    const SampledWavelengths& swl, float4 v) const noexcept
{
    auto enc = renderer().spectrum()->base()->encode_static_srgb_albedo(
        extend_color_to_rgb(v.xyz(), base()->channels()));
    return renderer().spectrum()->decode_albedo(swl, enc);
}

Spectrum::Decode Texture::Instance::evaluate_static_unbounded_spectrum_impl(
    const SampledWavelengths& swl, float4 v) const noexcept
{
    auto enc = renderer().spectrum()->base()->encode_static_srgb_unbounded(
        extend_color_to_rgb(v.xyz(), base()->channels()));
    return renderer().spectrum()->decode_unbounded(swl, enc);
}

Spectrum::Decode Texture::Instance::evaluate_static_illuminant_spectrum_impl(
    const SampledWavelengths& swl, float4 v) const noexcept
{
    auto enc = renderer().spectrum()->base()->encode_static_srgb_illuminant(
        extend_color_to_rgb(v.xyz(), base()->channels()));
    return renderer().spectrum()->decode_illuminant(swl, enc);
}

} // namespace Yutrel
