#include "scattering.h"

#include "sampling.h"

namespace Yutrel
{
Float MicrofacetDistribution::G1(Expr<float3> w) const noexcept
{
    return 1.0f / (1.0f + Lambda(w));
}

Float MicrofacetDistribution::G(Expr<float3> wo, Expr<float3> wi) const noexcept
{
    return 1.0f / (1.0f + Lambda(wo) + Lambda(wi));
}

Float MicrofacetDistribution::pdf(Expr<float3> wo, Expr<float3> wh) const noexcept
{
    return D(wh) * G1(wo) * abs_dot(wo, wh) / abs_cos_theta(wo);
}

TrowbridgeReitzDistribution::TrowbridgeReitzDistribution(Expr<float2> alpha) noexcept
    : MicrofacetDistribution{ite(max(alpha.x, alpha.y) < 1e-3f, alpha, max(alpha, 1e-4f))} {}

Bool TrowbridgeReitzDistribution::effectively_smooth() const noexcept
{
    return max(alpha().x, alpha().y) < 1e-3f;
}

Float TrowbridgeReitzDistribution::roughness_to_alpha(Expr<float> roughness) noexcept
{
    return sqrt(roughness);
}

Float2 TrowbridgeReitzDistribution::roughness_to_alpha(Expr<float2> roughness) noexcept
{
    return sqrt(roughness);
}

Float TrowbridgeReitzDistribution::D(Expr<float3> wh) const noexcept
{
    auto tan2_theta_h = tan2_theta(wh);
    auto cos4_theta_h = sqr(cos2_theta(wh));
    auto e            = tan2_theta_h * (sqr(cos_phi(wh) / alpha().x) + sqr(sin_phi(wh) / alpha().y));
    auto d            = 1.0f / (pi * alpha().x * alpha().y * cos4_theta_h * sqr(1.0f + e));
    return ite(luisa::compute::isinf(tan2_theta_h) | cos4_theta_h < 1e-16f, 0.0f, d);
}

Float TrowbridgeReitzDistribution::Lambda(Expr<float3> w) const noexcept
{
    auto tan2_theta_w = tan2_theta(w);
    auto alpha2       = sqr(cos_phi(w) * alpha().x) + sqr(sin_phi(w) * alpha().y);
    auto lambda       = 0.5f * (sqrt(1.0f + alpha2 * tan2_theta_w) - 1.0f);
    return ite(luisa::compute::isinf(tan2_theta_w), 0.0f, lambda);
}

Float3 TrowbridgeReitzDistribution::sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept
{
    auto wh = normalize(make_float3(alpha().x * wo.x, alpha().y * wo.y, wo.z));
    wh      = ite(wh.z < 0.0f, -wh, wh);

    auto t1 = ite(wh.z < 0.99999f,
                  normalize(cross(make_float3(0.0f, 0.0f, 1.0f), wh)),
                  make_float3(1.0f, 0.0f, 0.0f));
    auto t2 = cross(wh, t1);
    auto p  = sample_uniform_disk_polar(u);
    auto h  = sqrt(max(0.0f, 1.0f - sqr(p.x)));
    p.y     = lerp(h, p.y, 0.5f * (1.0f + wh.z));
    auto pz = sqrt(max(0.0f, 1.0f - dot(p, p)));
    auto nh = p.x * t1 + p.y * t2 + pz * wh;
    auto wm = normalize(make_float3(alpha().x * nh.x, alpha().y * nh.y, max(1e-6f, nh.z)));
    return ite(dot(wm, wo) < 0.0f, -wm, wm);
}

Float fresnel_dielectric(Expr<float> cos_theta_i_in, Expr<float> eta_i_in, Expr<float> eta_t_in) noexcept
{
    static Callable impl = [](Float cos_theta_i_in, Float eta_i_in, Float eta_t_in) noexcept
    {
        auto cos_theta_i = clamp(cos_theta_i_in, -1.0f, 1.0f);
        auto entering    = cos_theta_i > 0.0f;
        auto eta_i       = ite(entering, eta_i_in, eta_t_in);
        auto eta_t       = ite(entering, eta_t_in, eta_i_in);
        cos_theta_i      = abs(cos_theta_i);
        auto sin_theta_i = sqrt(max(0.0f, 1.0f - sqr(cos_theta_i)));
        auto sin_theta_t = eta_i / eta_t * sin_theta_i;
        auto cos_theta_t = sqrt(max(0.0f, 1.0f - sqr(sin_theta_t)));
        auto r_parallel  = (eta_t * cos_theta_i - eta_i * cos_theta_t) /
                          (eta_t * cos_theta_i + eta_i * cos_theta_t);
        auto r_perp = (eta_i * cos_theta_i - eta_t * cos_theta_t) /
                      (eta_i * cos_theta_i + eta_t * cos_theta_t);
        auto f = 0.5f * (sqr(r_parallel) + sqr(r_perp));
        return ite(sin_theta_t < 1.0f, f, 1.0f);
    };
    return impl(cos_theta_i_in, eta_i_in, eta_t_in);
}

Float FresnelDielectric::evaluate(Expr<float> cos_theta_i) const noexcept
{
    return fresnel_dielectric(cos_theta_i, m_eta_i, m_eta_t);
}

SampledSpectrum BxDF::sample(Expr<float3> wo, Float3* wi, Expr<float2> u, Float* pdf, TransportMode mode) const noexcept
{
    auto wi_sample = sample_wi(wo, u, mode);
    auto valid     = wi_sample.valid;
    *wi            = wi_sample.wi;
    *pdf           = ite(valid, this->pdf(wo, *wi, mode), 0.0f);
    return ite(valid, evaluate(wo, *wi, mode), 0.0f);
}

Float BxDF::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    return ite(same_hemisphere(wo, wi), abs_cos_theta(wi) * inv_pi, 0.0f);
}

BxDF::SampledDirection BxDF::sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept
{
    auto wi = sample_cosine_hemisphere(u);
    wi.z *= sign(cos_theta(wo));
    return {.wi = wi, .valid = true};
}

SampledSpectrum LambertianReflection::evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    return m_reflectance * ite(same_hemisphere(wo, wi), inv_pi, 0.0f);
}

SampledSpectrum SpecularReflection::evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    return SampledSpectrum{m_reflectance.dimension()};
}

SampledSpectrum SpecularReflection::sample(Expr<float3> wo, Float3* wi, Expr<float2> u,
                                           Float* pdf, TransportMode mode) const noexcept
{
    *wi  = make_float3(-wo.x, -wo.y, wo.z);
    *pdf = ite(abs_cos_theta(*wi) > 0.0f, 1.0f, 0.0f);
    return m_reflectance * m_fresnel->evaluate(cos_theta(wo)) /
           max(abs_cos_theta(*wi), 1e-30f);
}

Float SpecularReflection::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    return 0.0f;
}

SampledSpectrum SpecularTransmission::evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    return SampledSpectrum{m_transmittance.dimension()};
}

SampledSpectrum SpecularTransmission::sample(Expr<float3> wo, Float3* wi, Expr<float2> u,
                                             Float* pdf, TransportMode mode) const noexcept
{
    auto entering = cos_theta(wo) > 0.0f;
    auto eta_i    = ite(entering, m_eta_a, m_eta_b);
    auto eta_t    = ite(entering, m_eta_b, m_eta_a);
    auto n        = make_float3(0.0f, 0.0f, ite(entering, 1.0f, -1.0f));
    auto eta      = eta_i / eta_t;
    auto valid    = refract(wo, n, eta, wi);
    auto F        = fresnel_dielectric(cos_theta(wo), m_eta_a, m_eta_b);
    auto ft       = m_transmittance * (1.0f - F) / max(abs_cos_theta(*wi), 1e-30f);
    if (mode == TransportMode::RADIANCE)
    {
        ft *= sqr(eta);
    }
    *pdf = ite(valid & (abs_cos_theta(*wi) > 0.0f), 1.0f, 0.0f);
    return ite(valid, ft, 0.0f);
}

Float SpecularTransmission::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    return 0.0f;
}

SampledSpectrum MicrofacetReflection::evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    SampledSpectrum f{m_reflectance.dimension()};
    auto wh = wo + wi;
    $if(same_hemisphere(wo, wi) & any(wh != 0.0f) &
        (abs_cos_theta(wo) > 0.0f) & (abs_cos_theta(wi) > 0.0f))
    {
        wh     = normalize(wh);
        auto F = m_fresnel->evaluate(dot(wi, ite(wh.z < 0.0f, -wh, wh)));
        f      = m_reflectance * F * abs(0.25f * m_distribution->D(wh) * m_distribution->G(wo, wi) /
                                         (cos_theta(wi) * cos_theta(wo)));
    };
    return f;
}

BxDF::SampledDirection MicrofacetReflection::sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept
{
    auto wh = m_distribution->sample_wh(wo, u);
    auto wi = reflect(-wo, wh);
    return {.wi = wi, .valid = same_hemisphere(wo, wi)};
}

Float MicrofacetReflection::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    auto p  = def(0.0f);
    auto wh = wo + wi;
    $if(same_hemisphere(wo, wi) & any(wh != 0.0f))
    {
        wh = normalize(wh);
        p  = m_distribution->pdf(wo, wh) / (4.0f * abs_dot(wo, wh));
    };
    return p;
}

SampledSpectrum MicrofacetTransmission::evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    auto cos_o = cos_theta(wo);
    auto cos_i = cos_theta(wi);
    auto eta   = ite(cos_o > 0.0f, m_eta_b / m_eta_a, m_eta_a / m_eta_b);
    auto wh_v  = wo + wi * eta;
    auto valid_wh = any(wh_v != 0.0f);
    auto wh    = normalize(ite(valid_wh, wh_v, make_float3(0.0f, 0.0f, 1.0f)));
    wh         = ite(wh.z < 0.0f, -wh, wh);
    SampledSpectrum f{m_transmittance.dimension()};
    $if(!same_hemisphere(wo, wi) & valid_wh & cos_o != 0.0f & cos_i != 0.0f &
        dot(wo, wh) * dot(wi, wh) < 0.0f)
    {
        auto sqrt_denom = dot(wo, wh) + eta * dot(wi, wh);
        auto F          = fresnel_dielectric(dot(wo, wh), m_eta_a, m_eta_b);
        f = (1.0f - F) * m_transmittance * m_distribution->D(wh) * m_distribution->G(wo, wi) *
            abs(dot(wi, wh) * dot(wo, wh) / (cos_i * cos_o * sqr(sqrt_denom)));
        if (mode == TransportMode::IMPORTANCE)
        {
            f *= sqr(eta);
        }
    };
    return f;
}

BxDF::SampledDirection MicrofacetTransmission::sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept
{
    auto eta = ite(cos_theta(wo) > 0.0f, m_eta_a / m_eta_b, m_eta_b / m_eta_a);
    auto wh  = m_distribution->sample_wh(wo, u);
    auto wi  = def(make_float3(0.0f));
    auto ok  = refract(wo, wh, eta, std::addressof(wi));
    return {.wi = wi, .valid = ok & !same_hemisphere(wo, wi)};
}

Float MicrofacetTransmission::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept
{
    auto p        = def(0.0f);
    auto entering = cos_theta(wo) > 0.0f;
    auto eta      = ite(entering, m_eta_b / m_eta_a, m_eta_a / m_eta_b);
    auto wh_v     = wo + wi * eta;
    auto valid_wh = any(wh_v != 0.0f);
    auto wh       = normalize(ite(valid_wh, wh_v, make_float3(0.0f, 0.0f, 1.0f)));
    wh            = ite(wh.z < 0.0f, -wh, wh);
    $if(!same_hemisphere(wo, wi) & valid_wh & dot(wo, wh) * dot(wi, wh) < 0.0f)
    {
        auto sqrt_denom = dot(wo, wh) + eta * dot(wi, wh);
        auto dwh_dwi    = sqr(eta / sqrt_denom) * abs_dot(wi, wh);
        p               = m_distribution->pdf(wo, wh) * dwh_dwi;
    };
    return p;
}

[[nodiscard]] Bool refract(Expr<float3> wi, Expr<float3> n, Expr<float> eta, Float3* wt) noexcept
{
    static Callable impl = [](Float3 wi, Float3 n, Float eta) noexcept
    {
        // Compute $\cos \theta_\roman{t}$ using Snell's law
        auto cosThetaI  = dot(n, wi);
        auto sin2ThetaI = max(0.0f, one_minus_sqr(cosThetaI));
        auto sin2ThetaT = sqr(eta) * sin2ThetaI;
        auto cosThetaT  = sqrt(max(0.0f, 1.f - sin2ThetaT));
        // Handle total internal reflection for transmission
        auto wt = (eta * cosThetaI - cosThetaT) * n - eta * wi;
        return make_float4(wt, sin2ThetaT);
    };
    auto v = impl(wi, n, eta);
    *wt    = v.xyz();
    return v.w < 1.0f;
}

} // namespace Yutrel
