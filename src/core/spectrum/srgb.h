#pragma once

#include "base/spectrum.h"
#include "scene/scene_builder.h"
#include "utils/color_space.h"

namespace Yutrel
{
class SRGBSpectrum : public Spectrum
{
public:
    class Instance : public Spectrum::Instance
    {
    public:
        explicit Instance(Renderer& renderer, CommandBuffer& command_buffer, const Spectrum* spectrum) noexcept
            : Spectrum::Instance(renderer, command_buffer, spectrum) {}

        [[nodiscard]] SampledWavelengths sample(Expr<float>) const noexcept override
        {
            SampledWavelengths swl{3u};
            auto lambdas = make_float3(602.785f, 539.285f, 445.772f); // rgb spectrum peak wavelengths
            for (auto i = 0u; i < 3u; i++)
            {
                swl.set_lambda(i, lambdas[i]);
                swl.set_pdf(i, 1.f);
            }

            return swl;
        }
        [[nodiscard]] Spectrum::Decode decode_albedo(
            const SampledWavelengths& swl, Expr<float4> v) const noexcept override
        {
            SampledSpectrum s{base()->dimension()};
            auto sv = saturate(v.xyz());
            for (auto i = 0u; i < 3u; i++)
            {
                s[i] = sv[i];
            }
            return {.value = s, .strength = linear_srgb_to_cie_y(sv)};
        }
        [[nodiscard]] Spectrum::Decode decode_unbounded(
            const SampledWavelengths& swl, Expr<float4> v) const noexcept override
        {
            SampledSpectrum s{base()->dimension()};
            auto sv = v.xyz();
            for (auto i = 0u; i < 3u; i++)
            {
                s[i] = sv[i];
            }
            return {.value = s, .strength = linear_srgb_to_cie_y(sv)};
        }
        [[nodiscard]] Spectrum::Decode decode_illuminant(
            const SampledWavelengths& swl, Expr<float4> v) const noexcept override
        {
            auto sv = max(v.xyz(), 0.f);
            SampledSpectrum s{base()->dimension()};
            for (auto i = 0u; i < 3u; i++)
            {
                s[i] = sv[i];
            }
            return {.value = s, .strength = linear_srgb_to_cie_y(sv)};
        }
        [[nodiscard]] Float cie_y(
            const SampledWavelengths& swl,
            const SampledSpectrum& sp) const noexcept override
        {
            return linear_srgb_to_cie_y(srgb(swl, sp));
        }
        [[nodiscard]] Float3 cie_xyz(
            const SampledWavelengths& swl,
            const SampledSpectrum& sp) const noexcept override
        {
            return linear_srgb_to_cie_xyz(srgb(swl, sp));
        }
        [[nodiscard]] Float3 srgb(
            const SampledWavelengths& swl,
            const SampledSpectrum& sp) const noexcept override
        {
            return make_float3(sp[0u], sp[1u], sp[2u]);
        }
        [[nodiscard]] Float4 encode_srgb_albedo(Expr<float3> rgb) const noexcept override
        {
            return make_float4(clamp(rgb, 0.f, 1.f), 1.f);
        }
        [[nodiscard]] Float4 encode_srgb_unbounded(Expr<float3> rgb) const noexcept override
        {
            return make_float4(rgb, 1.f);
        }
        [[nodiscard]] Float4 encode_srgb_illuminant(Expr<float3> rgb) const noexcept override
        {
            return make_float4(max(rgb, 0.f), 1.f);
        }
    };

public:
    SRGBSpectrum() noexcept = default;

    [[nodiscard]] luisa::unique_ptr<Spectrum::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override
    {
        return luisa::make_unique<Instance>(renderer, command_buffer, this);
    }

    [[nodiscard]] bool is_fixed() const noexcept override { return true; }
    [[nodiscard]] uint dimension() const noexcept override { return 3u; }
    [[nodiscard]] float4 encode_static_srgb_albedo(float3 rgb) const noexcept override { return make_float4(clamp(rgb, 0.f, 1.f), 1.f); }
    [[nodiscard]] float4 encode_static_srgb_unbounded(float3 rgb) const noexcept override { return make_float4(rgb, 1.f); }
    [[nodiscard]] float4 encode_static_srgb_illuminant(float3 rgb) const noexcept override { return make_float4(max(rgb, 0.f), 1.f); }
};

class SRGBSpectrumSpec final : public SpectrumSpec
{
public:
    [[nodiscard]] const Spectrum* build(SceneBuilder& builder) const noexcept override
    {
        return builder.emplace<Spectrum, SRGBSpectrum>();
    }
};

} // namespace Yutrel
