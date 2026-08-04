#include "openpbr.h"

#include <array>
#include <cstdint>

#include <luisa/dsl/sugar.h>

#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/sampling.h"
#include "utils/scattering.h"

namespace Yutrel
{
namespace
{
constexpr uint energy_table_size = 32u;

// LUT data is copied from Adobe openpbr-bsdf commit 8a20d6f9 under Apache-2.0.
static constexpr std::array<luisa::ushort, energy_table_size * energy_table_size * energy_table_size>
    opaque_dielectric_energy_data{
#include "openpbr_data/openpbr_opaque_dielectric_energy_complement_data.h"
    };
static constexpr std::array<luisa::ushort, energy_table_size * energy_table_size>
    opaque_dielectric_average_data{
#include "openpbr_data/openpbr_opaque_dielectric_avg_energy_complement_data.h"
    };
static constexpr std::array<luisa::ushort, energy_table_size * energy_table_size>
    ideal_metal_energy_data{
#include "openpbr_data/openpbr_ideal_metal_energy_complement_data.h"
    };
static constexpr std::array<luisa::ushort, energy_table_size>
    ideal_metal_average_data{
#include "openpbr_data/openpbr_ideal_metal_avg_energy_complement_data.h"
    };

template <typename Texture, typename Size>
[[nodiscard]] uint register_lut_texture(Renderer& renderer,
                                        CommandBuffer& command_buffer,
                                        luisa::string_view identifier,
                                        Size size,
                                        luisa::span<const luisa::ushort> data) noexcept
{
    return renderer.register_named_id(identifier, [&]() noexcept
    {
        auto texture = renderer.create<Texture>(PixelStorage::SHORT1, size);
        auto id = renderer.register_bindless(*texture, TextureSampler::linear_point_edge());
        command_buffer << texture->copy_from(data) << compute::commit();
        return id;
    });
}

[[nodiscard]] Float remap_lut_index(Expr<float> exact_index) noexcept
{
    constexpr auto inv_size = 1.0f / static_cast<float>(energy_table_size);
    return clamp((exact_index + 0.5f) * inv_size,
                 0.5f * inv_size, 1.0f - 0.5f * inv_size);
}

[[nodiscard]] Float lut_lookup_2d(const Renderer& renderer, uint lut_id,
                                  Expr<float> exact_x, Expr<float> exact_y) noexcept
{
    return renderer.tex2d(lut_id)
        .sample(make_float2(remap_lut_index(exact_y), remap_lut_index(exact_x)))
        .x;
}

[[nodiscard]] Float lut_lookup_1d(const Renderer& renderer, uint lut_id,
                                  Expr<float> exact_x) noexcept
{
    return renderer.tex2d(lut_id)
        .sample(make_float2(remap_lut_index(exact_x), 0.5f))
        .x;
}

[[nodiscard]] Float lut_lookup_3d(const Renderer& renderer, uint lut_id,
                                  Expr<float> exact_x, Expr<float> exact_y,
                                  Expr<float> exact_z) noexcept
{
    return renderer.tex3d(lut_id)
        .sample(make_float3(remap_lut_index(exact_z),
                            remap_lut_index(exact_y),
                            remap_lut_index(exact_x)))
        .x;
}

[[nodiscard]] Float ior_exact_index(Expr<float> ior) noexcept
{
    constexpr auto half_size = static_cast<float>(energy_table_size / 2u);
    constexpr auto half_size_minus_one = half_size - 1.0f;
    auto below_one = half_size_minus_one - (1.0f / ior - 1.0f) * (half_size_minus_one / 1.5f);
    auto above_one = half_size + (ior - 1.0f) * (half_size_minus_one / 1.5f);
    return ite(ior < 1.0f, below_one, above_one);
}

[[nodiscard]] Float alpha_exact_index(Expr<float> alpha) noexcept
{
    return sqrt(alpha) * 31.0f;
}

[[nodiscard]] Float extrapolate_opaque_ior(Expr<float> table_value,
                                           Expr<float> ior) noexcept
{
    constexpr auto ior_max = 2.5f;
    constexpr auto inverse_ior_max = 1.0f / ior_max;
    constexpr auto f0_max = ((ior_max - 1.0f) / (ior_max + 1.0f)) *
                            ((ior_max - 1.0f) / (ior_max + 1.0f));
    auto f0 = sqr((ior - 1.0f) / (ior + 1.0f));
    auto progress = (f0 - f0_max) / (1.0f - f0_max);
    return ite((ior > ior_max) | (ior < inverse_ior_max),
               (1.0f - progress) * table_value, table_value);
}

[[nodiscard]] Float opaque_dielectric_energy(const Renderer& renderer, uint lut_id,
                                             Expr<float> ior, Expr<float> alpha,
                                             Expr<float> cosine) noexcept
{
    auto value = lut_lookup_3d(renderer, lut_id,
                               ior_exact_index(ior), alpha_exact_index(alpha), cosine * 31.0f);
    return extrapolate_opaque_ior(value, ior);
}

[[nodiscard]] Float opaque_dielectric_average(const Renderer& renderer, uint lut_id,
                                              Expr<float> ior, Expr<float> alpha) noexcept
{
    auto value = lut_lookup_2d(renderer, lut_id,
                               ior_exact_index(ior), alpha_exact_index(alpha));
    return extrapolate_opaque_ior(value, ior);
}

[[nodiscard]] Float ideal_metal_energy(const Renderer& renderer, uint lut_id,
                                      Expr<float> alpha, Expr<float> cosine) noexcept
{
    return lut_lookup_2d(renderer, lut_id,
                         alpha_exact_index(alpha), cosine * 31.0f);
}

[[nodiscard]] Float ideal_metal_average(const Renderer& renderer, uint lut_id,
                                       Expr<float> alpha) noexcept
{
    return lut_lookup_1d(renderer, lut_id, alpha_exact_index(alpha));
}

[[nodiscard]] auto fifth_power(auto x) noexcept
{
    auto x2 = x * x;
    return x2 * x2 * x;
}

[[nodiscard]] SampledSpectrum metal_schlick_b(const SampledSpectrum& f0,
                                              const SampledSpectrum& f82_tint) noexcept
{
    constexpr auto cos_theta_max = 1.0f / 7.0f;
    constexpr auto one_minus_cos = 1.0f - cos_theta_max;
    constexpr auto one_minus_cos_5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;
    constexpr auto one_minus_cos_6 = one_minus_cos_5 * one_minus_cos;
    return (f0 + (1.0f - f0) * one_minus_cos_5) * (1.0f - f82_tint) /
           (cos_theta_max * one_minus_cos_6);
}

[[nodiscard]] SampledSpectrum metal_f82(const SampledSpectrum& f0,
                                       const SampledSpectrum& f82_tint,
                                       Expr<float> cosine) noexcept
{
    auto one_minus_cos = 1.0f - cosine;
    auto b = metal_schlick_b(f0, f82_tint);
    return saturate(f0 + ((1.0f - f0) - b * cosine * one_minus_cos) *
                             fifth_power(one_minus_cos));
}

[[nodiscard]] SampledSpectrum metal_average_fresnel(const SampledSpectrum& f0,
                                                    const SampledSpectrum& f82_tint) noexcept
{
    auto b = metal_schlick_b(f0, f82_tint);
    return saturate(f0 + (1.0f - f0) * (1.0f / 21.0f) - b * (1.0f / 126.0f));
}

[[nodiscard]] SampledSpectrum eon_brdf(const SampledSpectrum& rho,
                                      Expr<float> roughness,
                                      Expr<float3> wo,
                                      Expr<float3> wi) noexcept
{
    constexpr auto fon_a = 0.5f - 2.0f / (3.0f * pi);
    constexpr auto fon_b = 2.0f / 3.0f - 28.0f / (15.0f * pi);
    constexpr auto g1 = 0.0571085289f;
    constexpr auto g2 = 0.491881867f;
    constexpr auto g3 = -0.332181442f;
    constexpr auto g4 = 0.0714429953f;
    auto mu_o = wo.z;
    auto mu_i = wi.z;
    auto s = dot(wo, wi) - mu_o * mu_i;
    auto s_over_t = ite(s > 0.0f, s / max(mu_i, mu_o), s);
    auto a = 1.0f / (1.0f + fon_a * roughness);
    auto f_ss = rho * (1.0f / pi) * a * (1.0f + roughness * s_over_t);
    auto directional_albedo = [&](Expr<float> mu) noexcept
    {
        auto c = 1.0f - mu;
        auto g_over_pi = c * (g1 + c * (g2 + c * (g3 + c * g4)));
        return (1.0f + roughness * g_over_pi) / (1.0f + fon_a * roughness);
    };
    auto e_o = directional_albedo(mu_o);
    auto e_i = directional_albedo(mu_i);
    auto e_average = a * (1.0f + fon_b * roughness);
    auto rho_ms = rho * rho * e_average / (1.0f - rho * (1.0f - e_average));
    auto f_ms = rho_ms * (1.0f / pi) * max(1.0e-7f, 1.0f - e_o) *
                max(1.0e-7f, 1.0f - e_i) / max(1.0e-7f, 1.0f - e_average);
    return f_ss + f_ms;
}

[[nodiscard]] Float openpbr_ggx_d(Expr<float3> wh, Expr<float2> alpha) noexcept
{
    auto tan2_h = tan2_theta(wh);
    auto cos4_h = sqr(cos2_theta(wh));
    auto e = tan2_h * (sqr(cos_phi(wh) / alpha.x) + sqr(sin_phi(wh) / alpha.y));
    auto d = 1.0f / (pi * alpha.x * alpha.y * cos4_h * sqr(1.0f + e));
    return ite(compute::isinf(tan2_h) | (cos4_h < 1.0e-16f), 0.0f, d);
}

[[nodiscard]] Float openpbr_ggx_g1(Expr<float3> w, Expr<float2> alpha) noexcept
{
    auto wz2 = sqr(w.z);
    auto slope2 = (sqr(alpha.x * w.x) + sqr(alpha.y * w.y)) /
                  max(wz2, 1.0e-30f);
    return ite(wz2 == 0.0f, 0.0f, 2.0f / (1.0f + sqrt(1.0f + slope2)));
}

[[nodiscard]] Float3 openpbr_sample_ggx_wh(Expr<float3> wo,
                                            Expr<float2> alpha,
                                            Expr<float2> u) noexcept
{
    auto ellipsoid = make_float3(alpha.x, alpha.y, 1.0f);
    auto incoming = normalize(ellipsoid * wo);
    auto incoming_xy2 = sqr(incoming.x) + sqr(incoming.y);
    auto inv_incoming_xy = 1.0f / sqrt(max(incoming_xy2, 1.0e-20f));
    auto tangent = ite(incoming_xy2 > 0.0f,
                       make_float3(-incoming.y, incoming.x, 0.0f) * inv_incoming_xy,
                       make_float3(1.0f, 0.0f, 0.0f));
    auto bitangent = cross(incoming, tangent);
    auto radius = sqrt(u.x);
    auto angle = 2.0f * pi * u.y;
    auto tangent_component = radius * cos(angle);
    auto bitangent_unscaled = radius * sin(angle);
    auto scale = 0.5f * (1.0f + incoming.z);
    auto bitangent_component = (1.0f - scale) *
                                   sqrt(max(0.0f, 1.0f - sqr(tangent_component))) +
                               scale * bitangent_unscaled;
    auto normal_component2 = max(0.0f, 1.0f -
                                          (sqr(tangent_component) + sqr(bitangent_component)));
    auto normal = tangent * tangent_component + bitangent * bitangent_component +
                  incoming * sqrt(normal_component2);
    normal = make_float3(normal.x, normal.y, max(normal.z, 0.0f));
    return normalize(ellipsoid * normal);
}

[[nodiscard]] Float evaluate_scalar(const Texture::Instance* texture,
                                    const Interaction& it, Expr<float> time,
                                    float fallback) noexcept
{
    return texture ? texture->evaluate(it, time).x : def(fallback);
}

[[nodiscard]] SampledSpectrum evaluate_color(const Texture::Instance* texture,
                                             const Interaction& it,
                                             const SampledWavelengths& swl,
                                             Expr<float> time,
                                             float fallback) noexcept
{
    return texture ? texture->evaluate_albedo_spectrum(it, swl, time).value
                   : SampledSpectrum{swl.dimension(), fallback};
}
} // namespace

OpenPBRSurface::OpenPBRSurface(const Texture* base_weight,
                               const Texture* base_color,
                               const Texture* base_metalness,
                               const Texture* base_diffuse_roughness,
                               const Texture* specular_weight,
                               const Texture* specular_color,
                               const Texture* specular_roughness,
                               const Texture* specular_roughness_anisotropy,
                               const Texture* specular_ior,
                               bool two_sided) noexcept
    : Surface{two_sided},
      m_base_weight{base_weight},
      m_base_color{base_color},
      m_base_metalness{base_metalness},
      m_base_diffuse_roughness{base_diffuse_roughness},
      m_specular_weight{specular_weight},
      m_specular_color{specular_color},
      m_specular_roughness{specular_roughness},
      m_specular_roughness_anisotropy{specular_roughness_anisotropy},
      m_specular_ior{specular_ior} {}

luisa::unique_ptr<Surface::Instance> OpenPBRSurface::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    auto opaque_dielectric_energy_lut = register_lut_texture<Volume<float>>(
        renderer, command_buffer, "__internal_openpbr_opaque_dielectric_energy",
        make_uint3(energy_table_size), luisa::span{opaque_dielectric_energy_data});
    auto opaque_dielectric_average_lut = register_lut_texture<Image<float>>(
        renderer, command_buffer, "__internal_openpbr_opaque_dielectric_average",
        make_uint2(energy_table_size), luisa::span{opaque_dielectric_average_data});
    auto ideal_metal_energy_lut = register_lut_texture<Image<float>>(
        renderer, command_buffer, "__internal_openpbr_ideal_metal_energy",
        make_uint2(energy_table_size), luisa::span{ideal_metal_energy_data});
    auto ideal_metal_average_lut = register_lut_texture<Image<float>>(
        renderer, command_buffer, "__internal_openpbr_ideal_metal_average",
        make_uint2(energy_table_size, 1u), luisa::span{ideal_metal_average_data});
    return luisa::make_unique<Instance>(
        renderer, this,
        renderer.build_texture(command_buffer, m_base_weight),
        renderer.build_texture(command_buffer, m_base_color),
        renderer.build_texture(command_buffer, m_base_metalness),
        renderer.build_texture(command_buffer, m_base_diffuse_roughness),
        renderer.build_texture(command_buffer, m_specular_weight),
        renderer.build_texture(command_buffer, m_specular_color),
        renderer.build_texture(command_buffer, m_specular_roughness),
        renderer.build_texture(command_buffer, m_specular_roughness_anisotropy),
        renderer.build_texture(command_buffer, m_specular_ior),
        opaque_dielectric_energy_lut, opaque_dielectric_average_lut,
        ideal_metal_energy_lut, ideal_metal_average_lut);
}

OpenPBRSurface::Instance::Instance(
    const Renderer& renderer, const OpenPBRSurface* surface,
    const Texture::Instance* base_weight, const Texture::Instance* base_color,
    const Texture::Instance* base_metalness, const Texture::Instance* base_diffuse_roughness,
    const Texture::Instance* specular_weight, const Texture::Instance* specular_color,
    const Texture::Instance* specular_roughness,
    const Texture::Instance* specular_roughness_anisotropy,
    const Texture::Instance* specular_ior,
    uint opaque_dielectric_energy_lut,
    uint opaque_dielectric_average_lut,
    uint ideal_metal_energy_lut,
    uint ideal_metal_average_lut) noexcept
    : Surface::Instance{renderer, surface},
      m_base_weight{base_weight}, m_base_color{base_color},
      m_base_metalness{base_metalness}, m_base_diffuse_roughness{base_diffuse_roughness},
      m_specular_weight{specular_weight}, m_specular_color{specular_color},
      m_specular_roughness{specular_roughness},
      m_specular_roughness_anisotropy{specular_roughness_anisotropy},
      m_specular_ior{specular_ior},
      m_opaque_dielectric_energy_lut{opaque_dielectric_energy_lut},
      m_opaque_dielectric_average_lut{opaque_dielectric_average_lut},
      m_ideal_metal_energy_lut{ideal_metal_energy_lut},
      m_ideal_metal_average_lut{ideal_metal_average_lut} {}

luisa::unique_ptr<Surface::Closure> OpenPBRSurface::Instance::create_closure(
    SampledWavelengths& swl, Expr<float> time) const noexcept
{
    return luisa::make_unique<Closure>(
        renderer(), swl, time,
        m_opaque_dielectric_energy_lut,
        m_opaque_dielectric_average_lut,
        m_ideal_metal_energy_lut,
        m_ideal_metal_average_lut);
}

void OpenPBRSurface::Instance::populate_closure(Surface::Closure* closure,
                                                const Interaction& it,
                                                Expr<float3> wo,
                                                Expr<float> eta_i) const noexcept
{
    auto& swl = closure->swl();
    auto time = closure->time();
    auto base_weight = clamp(evaluate_scalar(m_base_weight, it, time, 1.0f), 0.0f, 1.0f);
    auto base_color = evaluate_color(m_base_color, it, swl, time, 0.8f);
    auto metalness = clamp(evaluate_scalar(m_base_metalness, it, time, 0.0f), 0.0f, 1.0f);
    auto diffuse_roughness = clamp(evaluate_scalar(m_base_diffuse_roughness, it, time, 0.0f), 0.0f, 1.0f);
    auto specular_weight = max(evaluate_scalar(m_specular_weight, it, time, 1.0f), 0.0f);
    auto specular_color = evaluate_color(m_specular_color, it, swl, time, 1.0f);
    auto roughness = clamp(evaluate_scalar(m_specular_roughness, it, time, 0.3f), 0.0f, 1.0f);
    auto anisotropy = clamp(evaluate_scalar(m_specular_roughness_anisotropy, it, time, 0.0f), 0.0f, 1.0f);
    auto specular_ior = max(evaluate_scalar(m_specular_ior, it, time, 1.5f), 1.0e-6f);

    auto alpha = max(sqr(roughness), 1.0e-6f);
    auto alpha_x = alpha * sqrt(2.0f / (1.0f + sqr(1.0f - anisotropy)));
    auto anisotropic_alpha = max(make_float2(alpha_x, (1.0f - anisotropy) * alpha_x), 1.0e-6f);

    auto relative_ior = specular_ior / max(eta_i, 1.0e-6f);
    auto f0 = sqr((relative_ior - 1.0f) / (relative_ior + 1.0f));
    auto weighted_f0 = min(specular_weight * f0, 0.9999f);
    auto external_weighted_ior = (1.0f + sqrt(weighted_f0)) / (1.0f - sqrt(weighted_f0));
    auto weighted_ior = ite(relative_ior < 1.0f, 1.0f / external_weighted_ior,
                            external_weighted_ior);

    auto weighted_base_color = base_color * base_weight;
    auto diffuse_albedo = weighted_base_color * (1.0f - metalness);
    auto metal_average = metal_average_fresnel(weighted_base_color, specular_color);
    auto metal_mms_scale = metal_average * metal_average * (metalness * specular_weight);

    auto wo_local = it.shading.world_to_local(wo);
    auto no_v = max(wo_local.z, 0.0f);
    auto dielectric_view = opaque_dielectric_energy(renderer(), m_opaque_dielectric_energy_lut,
                                                    weighted_ior, alpha, no_v);
    auto dielectric_average = opaque_dielectric_average(renderer(), m_opaque_dielectric_average_lut,
                                                        weighted_ior, alpha);
    auto dielectric_view_compensation = dielectric_view / max(dielectric_average, 1.0e-12f);
    auto metal_view = ideal_metal_energy(renderer(), m_ideal_metal_energy_lut, alpha, no_v);

    auto dielectric_f = specular_color * ((1.0f - metalness) *
                                           fresnel_dielectric(no_v, 1.0f, weighted_ior));
    auto metal_f = metal_f82(weighted_base_color, specular_color, no_v) *
                   (metalness * specular_weight);
    auto specular_lobe_weight = (dielectric_f + metal_f).max();
    auto metal_mms_lobe_weight = ite(alpha >= 0.0016f,
                                     metal_mms_scale.max() * metal_view, 0.0f);
    auto diffuse_lobe_weight = (diffuse_albedo * dielectric_view_compensation).max();
    auto total_lobe_weight = specular_lobe_weight + metal_mms_lobe_weight + diffuse_lobe_weight;

    Closure::Context ctx{
        .it = it,
        .weighted_base_color = weighted_base_color,
        .specular_color = specular_color,
        .diffuse_albedo = diffuse_albedo,
        .metal_mms_scale = metal_mms_scale,
        .metalness = metalness,
        .specular_weight = specular_weight,
        .diffuse_roughness = diffuse_roughness,
        .alpha = alpha,
        .anisotropic_alpha = anisotropic_alpha,
        .weighted_ior = weighted_ior,
        .dielectric_view_compensation = dielectric_view_compensation,
        .metal_view_energy_complement = metal_view,
        .specular_lobe_weight = specular_lobe_weight,
        .metal_mms_lobe_weight = metal_mms_lobe_weight,
        .diffuse_lobe_weight = diffuse_lobe_weight,
        .total_lobe_weight = total_lobe_weight,
    };
    closure->bind(std::move(ctx));
}

OpenPBRSurface::Closure::Closure(const Renderer& renderer,
                                const SampledWavelengths& swl,
                                Expr<float> time,
                                uint opaque_dielectric_energy_lut,
                                uint opaque_dielectric_average_lut,
                                uint ideal_metal_energy_lut,
                                uint ideal_metal_average_lut) noexcept
    : Surface::Closure{renderer, swl, time},
      m_opaque_dielectric_energy_lut{opaque_dielectric_energy_lut},
      m_opaque_dielectric_average_lut{opaque_dielectric_average_lut},
      m_ideal_metal_energy_lut{ideal_metal_energy_lut},
      m_ideal_metal_average_lut{ideal_metal_average_lut} {}

UInt OpenPBRSurface::Closure::lobe_flags() const noexcept
{
    return Surface::lobe_reflection | Surface::lobe_glossy | Surface::lobe_diffuse;
}

Surface::Evaluation OpenPBRSurface::Closure::evaluate_impl(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode,
    ScatterFlags flags) const noexcept
{
    auto result = Surface::Evaluation::zero(swl().dimension());
    if (!has_scatter_flag(flags, ScatterFlags::Reflection)) { return result; }
    auto&& ctx = context<Context>();
    auto wo_local = ctx.it.shading.world_to_local(wo);
    auto wi_local = ctx.it.shading.world_to_local(wi);
    $if((wo_local.z > 0.0f) & (wi_local.z > 0.0f))
    {
        auto wh_v = wo_local + wi_local;
        auto valid_wh = any(wh_v != 0.0f);
        auto wh = normalize(ite(valid_wh, wh_v, make_float3(0.0f, 0.0f, 1.0f)));
        auto cos_h = abs(dot(wo_local, wh));
        auto fresnel = ctx.specular_color * ((1.0f - ctx.metalness) *
                                             fresnel_dielectric(cos_h, 1.0f, ctx.weighted_ior)) +
                       metal_f82(ctx.weighted_base_color, ctx.specular_color, cos_h) *
                           (ctx.metalness * ctx.specular_weight);
        auto specular_f_cos = fresnel * (openpbr_ggx_d(wh, ctx.anisotropic_alpha) *
                                         openpbr_ggx_g1(wo_local, ctx.anisotropic_alpha) *
                                         openpbr_ggx_g1(wi_local, ctx.anisotropic_alpha) /
                                         (4.0f * wo_local.z));
        specular_f_cos = ite(valid_wh, specular_f_cos, 0.0f);

        auto metal_light = ideal_metal_energy(renderer(), m_ideal_metal_energy_lut,
                                              ctx.alpha, wi_local.z);
        auto metal_average = ideal_metal_average(renderer(), m_ideal_metal_average_lut, ctx.alpha);
        auto metal_factors = ctx.metal_view_energy_complement * metal_light /
                             max(metal_average, 1.0e-12f);
        metal_factors = min(metal_factors, 1.0f / wi_local.z);
        auto metal_f_cos = ctx.metal_mms_scale *
                           (metal_factors * (1.0f / pi) * wi_local.z);
        metal_f_cos = ite(ctx.alpha >= 0.0016f, metal_f_cos, 0.0f);

        auto dielectric_light = opaque_dielectric_energy(renderer(), m_opaque_dielectric_energy_lut,
                                                         ctx.weighted_ior, ctx.alpha,
                                                         wi_local.z);
        auto diffuse_factor = ctx.dielectric_view_compensation * dielectric_light;
        auto diffuse_f_cos = eon_brdf(ctx.diffuse_albedo, ctx.diffuse_roughness,
                                      wo_local, wi_local) * diffuse_factor * wi_local.z;

        auto specular_pdf = openpbr_ggx_d(wh, ctx.anisotropic_alpha) *
                            openpbr_ggx_g1(wo_local, ctx.anisotropic_alpha) /
                            (4.0f * wo_local.z);
        specular_pdf = ite(valid_wh, specular_pdf, 0.0f);
        auto metal_pdf = ite(ctx.alpha >= 0.0016f, 1.0f / (2.0f * pi), 0.0f);
        auto diffuse_pdf = wi_local.z * (1.0f / pi);
        auto weighted_pdf = ctx.specular_lobe_weight * specular_pdf +
                            ctx.metal_mms_lobe_weight * metal_pdf +
                            ctx.diffuse_lobe_weight * diffuse_pdf;
        auto pdf = ite(ctx.total_lobe_weight > 0.0f,
                       weighted_pdf / ctx.total_lobe_weight, 0.0f);
        result = Surface::Evaluation{
            .f = specular_f_cos + metal_f_cos + diffuse_f_cos,
            .pdf = pdf,
            .f_diffuse = diffuse_f_cos,
            .pdf_diffuse = diffuse_pdf,
        };
    };
    return result;
}

Surface::Sample OpenPBRSurface::Closure::sample_impl(
    Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u,
    TransportMode mode, ScatterFlags flags) const noexcept
{
    auto result = Surface::Sample::zero(swl().dimension());
    result.wi = make_float3(0.0f);
    if (!has_scatter_flag(flags, ScatterFlags::Reflection)) { return result; }
    auto&& ctx = context<Context>();
    auto wo_local = ctx.it.shading.world_to_local(wo);
    $if((wo_local.z > 0.0f) & (ctx.total_lobe_weight > 1.17549435e-38f))
    {
        auto lobe_random = min(u_lobe, 0.99999994f) * ctx.total_lobe_weight;
        auto selected = ite(lobe_random < ctx.specular_lobe_weight, 0u,
                            ite(lobe_random < ctx.specular_lobe_weight +
                                                  ctx.metal_mms_lobe_weight,
                                1u, 2u));
        auto selected_offset = ite(selected == 0u, 0.0f,
                                   ite(selected == 1u, ctx.specular_lobe_weight,
                                       ctx.specular_lobe_weight + ctx.metal_mms_lobe_weight));
        auto selected_weight = ite(selected == 0u, ctx.specular_lobe_weight,
                                   ite(selected == 1u, ctx.metal_mms_lobe_weight,
                                       ctx.diffuse_lobe_weight));
        auto remapped_lobe = min((lobe_random - selected_offset) /
                                     max(selected_weight, 1.17549435e-38f),
                                 0.99999994f);
        auto lobe_u = make_float2(remapped_lobe, u.x);
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        $if(selected == 0u)
        {
            auto wh = openpbr_sample_ggx_wh(wo_local, ctx.anisotropic_alpha, lobe_u);
            wi_local = reflect(-wo_local, wh);
        }
        $elif(selected == 1u)
        {
            auto z = lobe_u.x;
            auto radius = sqrt(max(0.0f, 1.0f - sqr(z)));
            auto phi = 2.0f * pi * lobe_u.y;
            // Adobe's normal-only basis maps local X to shading Y and local Y to shading X.
            wi_local = make_float3(sin(phi) * radius, cos(phi) * radius, z);
        }
        $else
        {
            auto phi = 2.0f * pi * lobe_u.x;
            auto z = sqrt(lobe_u.y);
            auto radius = sqrt(max(0.0f, 1.0f - lobe_u.y));
            wi_local = make_float3(sin(phi) * radius, cos(phi) * radius, z);
        };
        $if(wi_local.z > 0.0f)
        {
            auto wi = ctx.it.shading.local_to_world(wi_local);
            auto eval = evaluate_impl(wo, wi, mode, flags);
            result = Surface::Sample{
                .eval = eval,
                .wi = wi,
                .event = Surface::event_reflect,
                .pdf_mis = eval.pdf,
                .delta = false,
                .eta = 1.0f,
            };
        };
    };
    return result;
}

void OpenPBRSurfaceSpec::visit_dependencies(SpecDependencyVisitor& visitor) const noexcept
{
    auto visit = [&visitor](const luisa::optional<TextureRef>& ref) noexcept
    {
        if (ref) { visitor.visit(*ref); }
    };
    visit(m_params.base_weight);
    visit(m_params.base_color);
    visit(m_params.base_metalness);
    visit(m_params.base_diffuse_roughness);
    visit(m_params.specular_weight);
    visit(m_params.specular_color);
    visit(m_params.specular_roughness);
    visit(m_params.specular_roughness_anisotropy);
    visit(m_params.specular_ior);
}

const Surface* OpenPBRSurfaceSpec::build(SceneBuilder& builder) const noexcept
{
    auto resolve = [&builder](const luisa::optional<TextureRef>& ref) noexcept -> const Texture*
    {
        return ref ? builder.resolve(*ref) : nullptr;
    };
    return builder.emplace<Surface, OpenPBRSurface>(
        resolve(m_params.base_weight), resolve(m_params.base_color),
        resolve(m_params.base_metalness), resolve(m_params.base_diffuse_roughness),
        resolve(m_params.specular_weight), resolve(m_params.specular_color),
        resolve(m_params.specular_roughness),
        resolve(m_params.specular_roughness_anisotropy),
        resolve(m_params.specular_ior), m_params.two_sided);
}
} // namespace Yutrel
