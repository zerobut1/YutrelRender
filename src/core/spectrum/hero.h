#pragma once

#include "base/spectrum.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class HeroWavelengthSpectrum : public Spectrum
{
private:
    static constexpr uint s_dimension = 4u;

public:
    class Instance : public Spectrum::Instance
    {
    private:
        SPD m_illum_d65;
        uint m_rgb2spec_t0;

    public:
        explicit Instance(Renderer& renderer, CommandBuffer& command_buffer, const Spectrum* spectrum, uint t0) noexcept;

        [[nodiscard]] SampledWavelengths sample(Expr<float> u) const noexcept override;
        [[nodiscard]] Spectrum::Decode decode_albedo(const SampledWavelengths& swl, Expr<float4> v) const noexcept override;
        [[nodiscard]] Spectrum::Decode decode_unbounded(const SampledWavelengths& swl, Expr<float4> v) const noexcept override;
        [[nodiscard]] Spectrum::Decode decode_illuminant(const SampledWavelengths& swl, Expr<float4> v) const noexcept override;
        [[nodiscard]] Float4 encode_srgb_albedo(Expr<float3> rgb) const noexcept override;
        [[nodiscard]] Float4 encode_srgb_unbounded(Expr<float3> rgb) const noexcept override;
        [[nodiscard]] Float4 encode_srgb_illuminant(Expr<float3> rgb) const noexcept override;
    };

public:
    HeroWavelengthSpectrum() noexcept = default;

    [[nodiscard]] luisa::unique_ptr<Spectrum::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;

    [[nodiscard]] bool is_fixed() const noexcept override { return false; }
    [[nodiscard]] uint dimension() const noexcept override { return s_dimension; }
    [[nodiscard]] float4 encode_static_srgb_albedo(float3 rgb) const noexcept override;
    [[nodiscard]] float4 encode_static_srgb_unbounded(float3 rgb) const noexcept override;
    [[nodiscard]] float4 encode_static_srgb_illuminant(float3 rgb) const noexcept override;
};

class HeroWavelengthSpectrumSpec final : public SpectrumSpec
{
public:
    [[nodiscard]] const Spectrum* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
