#pragma once

#include <luisa/dsl/syntax.h>

#include "frame.h"
#include "spectra.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

enum TransportMode
{
    RADIANCE,
    IMPORTANCE
};

class MicrofacetDistribution
{
private:
    Float2 m_alpha;

public:
    explicit MicrofacetDistribution(Expr<float2> alpha) noexcept : m_alpha{alpha} {}
    virtual ~MicrofacetDistribution() noexcept = default;

    [[nodiscard]] Float G1(Expr<float3> w) const noexcept;
    [[nodiscard]] Float G(Expr<float3> wo, Expr<float3> wi) const noexcept;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wh) const noexcept;
    [[nodiscard]] virtual Float D(Expr<float3> wh) const noexcept = 0;
    [[nodiscard]] virtual Float Lambda(Expr<float3> w) const noexcept = 0;
    [[nodiscard]] virtual Float3 sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept = 0;
    [[nodiscard]] auto alpha() const noexcept { return m_alpha; }
};

class TrowbridgeReitzDistribution final : public MicrofacetDistribution
{
public:
    explicit TrowbridgeReitzDistribution(Expr<float2> alpha) noexcept;

    [[nodiscard]] Float D(Expr<float3> wh) const noexcept override;
    [[nodiscard]] Float Lambda(Expr<float3> w) const noexcept override;
    [[nodiscard]] Float3 sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept override;
    [[nodiscard]] Bool effectively_smooth() const noexcept;

    [[nodiscard]] static Float roughness_to_alpha(Expr<float> roughness) noexcept;
    [[nodiscard]] static Float2 roughness_to_alpha(Expr<float2> roughness) noexcept;
};

[[nodiscard]] Float fresnel_dielectric(Expr<float> cos_theta_i, Expr<float> eta_i, Expr<float> eta_t) noexcept;

class FresnelDielectric
{
private:
    Float m_eta_i;
    Float m_eta_t;

public:
    FresnelDielectric(Expr<float> eta_i, Expr<float> eta_t) noexcept
        : m_eta_i{eta_i}, m_eta_t{eta_t} {}

    [[nodiscard]] Float evaluate(Expr<float> cos_theta_i) const noexcept;
    [[nodiscard]] auto eta_i() const noexcept { return m_eta_i; }
    [[nodiscard]] auto eta_t() const noexcept { return m_eta_t; }
};

class BxDF
{
public:
    struct SampledDirection
    {
        Float3 wi;
        Bool valid;
    };

public:
    virtual ~BxDF() noexcept = default;

    [[nodiscard]] virtual SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept = 0;
    [[nodiscard]] virtual SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept;
    [[nodiscard]] virtual SampledSpectrum sample(Expr<float3> wo, Float3* wi, Expr<float2> u, Float* pdf, TransportMode mode) const noexcept;
    [[nodiscard]] virtual Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept;
    [[nodiscard]] virtual SampledSpectrum albedo() const noexcept = 0;
};

class LambertianReflection final : public BxDF
{
private:
    SampledSpectrum m_reflectance;

public:
    explicit LambertianReflection(const SampledSpectrum& reflectance) noexcept
        : m_reflectance{reflectance} {}

    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return m_reflectance; }
};

class SpecularReflection final : public BxDF
{
private:
    SampledSpectrum m_reflectance;
    const FresnelDielectric* m_fresnel;

public:
    SpecularReflection(const SampledSpectrum& reflectance, const FresnelDielectric* fresnel) noexcept
        : m_reflectance{reflectance}, m_fresnel{fresnel} {}

    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledSpectrum sample(Expr<float3> wo, Float3* wi, Expr<float2> u,
                                         Float* pdf, TransportMode mode) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return m_reflectance; }
};

class SpecularTransmission final : public BxDF
{
private:
    SampledSpectrum m_transmittance;
    Float m_eta_a;
    Float m_eta_b;

public:
    SpecularTransmission(const SampledSpectrum& transmittance,
                         Expr<float> eta_a, Expr<float> eta_b) noexcept
        : m_transmittance{transmittance}, m_eta_a{eta_a}, m_eta_b{eta_b} {}

    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledSpectrum sample(Expr<float3> wo, Float3* wi, Expr<float2> u,
                                         Float* pdf, TransportMode mode) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return SampledSpectrum{m_transmittance.dimension()}; }
};

class MicrofacetReflection final : public BxDF
{
private:
    SampledSpectrum m_reflectance;
    const MicrofacetDistribution* m_distribution;
    const FresnelDielectric* m_fresnel;

public:
    MicrofacetReflection(const SampledSpectrum& reflectance,
                         const MicrofacetDistribution* distribution,
                         const FresnelDielectric* fresnel) noexcept
        : m_reflectance{reflectance}, m_distribution{distribution}, m_fresnel{fresnel} {}

    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return m_reflectance; }
};

class MicrofacetTransmission final : public BxDF
{
private:
    SampledSpectrum m_transmittance;
    const MicrofacetDistribution* m_distribution;
    Float m_eta_a;
    Float m_eta_b;

public:
    MicrofacetTransmission(const SampledSpectrum& transmittance,
                           const MicrofacetDistribution* distribution,
                           Expr<float> eta_a, Expr<float> eta_b) noexcept
        : m_transmittance{transmittance}, m_distribution{distribution}, m_eta_a{eta_a}, m_eta_b{eta_b} {}

    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return SampledSpectrum{m_transmittance.dimension()}; }
};

[[nodiscard]] Bool refract(Expr<float3> wi, Expr<float3> n, Expr<float> eta, Float3* wt) noexcept;

} // namespace Yutrel

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::BxDF)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::BxDF::SampledDirection)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::LambertianReflection)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::SpecularReflection)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::SpecularTransmission)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::MicrofacetDistribution)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::TrowbridgeReitzDistribution)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::FresnelDielectric)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::MicrofacetReflection)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::MicrofacetTransmission)
