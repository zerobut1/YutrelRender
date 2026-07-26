#include "dielectric.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <luisa/dsl/sugar.h>

#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/scattering.h"

namespace Yutrel
{
CauchyEta CauchyEta::fit(float3 wavelengths_nm, float3 eta) noexcept
{
    std::array wavelengths{wavelengths_nm.x, wavelengths_nm.y, wavelengths_nm.z};
    std::array indices{eta.x, eta.y, eta.z};
    std::array<std::array<double, 4u>, 3u> m{};
    for (auto row = 0u; row < 3u; row++)
    {
        auto inv_l2 = 1.0 / (static_cast<double>(wavelengths[row]) * wavelengths[row]);
        m[row] = {1.0, inv_l2, inv_l2 * inv_l2, static_cast<double>(indices[row])};
    }
    for (auto column = 0u; column < 3u; column++)
    {
        auto pivot = column;
        for (auto row = column + 1u; row < 3u; row++)
        {
            if (std::abs(m[row][column]) > std::abs(m[pivot][column])) { pivot = row; }
        }
        LUISA_ASSERT(std::abs(m[pivot][column]) > 1e-30, "Degenerate Cauchy eta fit.");
        if (pivot != column) { std::swap(m[pivot], m[column]); }
        auto scale = m[column][column];
        for (auto j = column; j < 4u; j++) { m[column][j] /= scale; }
        for (auto row = 0u; row < 3u; row++)
        {
            if (row == column) { continue; }
            auto factor = m[row][column];
            for (auto j = column; j < 4u; j++) { m[row][j] -= factor * m[column][j]; }
        }
    }
    return CauchyEta{.coefficients = make_float3(
                         static_cast<float>(m[0u][3u]),
                         static_cast<float>(m[1u][3u]),
                         static_cast<float>(m[2u][3u]))};
}

Float CauchyEta::evaluate(Expr<float> lambda_nm) const noexcept
{
    auto inv_l2 = sqr(1.0f / lambda_nm);
    return coefficients.x + coefficients.y * inv_l2 + coefficients.z * sqr(inv_l2);
}

const CauchyEta& glass_f11_cauchy_eta() noexcept
{
    static const auto eta = CauchyEta::fit(
        make_float3(656.27f, 587.56f, 486.13f),
        make_float3(1.7754589288508518f, 1.7842240428294434f, 1.8065917880168352f));
    return eta;
}

class Dielectric::Closure::Impl
{
private:
    const Context& m_ctx;
    TrowbridgeReitzDistribution m_distribution;
    FresnelDielectric m_fresnel;
    SampledSpectrum m_unit;
    SpecularReflection m_specular_reflection;
    SpecularTransmission m_specular_transmission;
    MicrofacetReflection m_microfacet_reflection;
    MicrofacetTransmission m_microfacet_transmission;

    [[nodiscard]] Float selection_probability(Expr<float> F, ScatterFlags flags,
                                              bool reflection) const noexcept
    {
        auto pr = def(0.0f);
        auto pt = def(0.0f);
        if (has_scatter_flag(flags, ScatterFlags::Reflection)) { pr = F; }
        if (has_scatter_flag(flags, ScatterFlags::Transmission)) { pt = 1.0f - F; }
        auto sum = pr + pt;
        return ite(sum > 0.0f, (reflection ? pr : pt) / sum, 0.0f);
    }

public:
    Impl(const Context& ctx, uint dimension) noexcept
        : m_ctx{ctx},
          m_distribution{ctx.alpha},
          m_fresnel{ctx.eta_i, ctx.eta_t},
          m_unit{dimension, 1.0f},
          m_specular_reflection{m_unit, std::addressof(m_fresnel)},
          m_specular_transmission{m_unit, ctx.eta_i, ctx.eta_t},
          m_microfacet_reflection{m_unit, std::addressof(m_distribution), std::addressof(m_fresnel)},
          m_microfacet_transmission{m_unit, std::addressof(m_distribution), ctx.eta_i, ctx.eta_t} {}

    [[nodiscard]] Bool smooth() const noexcept
    {
        return (m_ctx.eta_i == m_ctx.eta_t) | m_distribution.effectively_smooth();
    }

    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode, ScatterFlags flags) const noexcept
    {
        auto result = Surface::Evaluation::zero(m_unit.dimension());
        $if(!smooth())
        {
            auto reflection = same_hemisphere(wo, wi);
            $if(reflection)
            {
                if (has_scatter_flag(flags, ScatterFlags::Reflection))
                {
                    auto wh_v = wo + wi;
                    auto valid = any(wh_v != 0.0f);
                    auto wh = normalize(ite(valid, wh_v, make_float3(0.0f, 0.0f, 1.0f)));
                    auto F = m_fresnel.evaluate(dot(wo, ite(wh.z < 0.0f, -wh, wh)));
                    auto p = selection_probability(F, flags, true);
                    auto f = m_microfacet_reflection.evaluate(wo, wi, mode);
                    auto pdf = m_microfacet_reflection.pdf(wo, wi, mode) * p;
                    auto f_cos = f * abs_cos_theta(wi);
                    result = {.f = f_cos, .pdf = pdf,
                              .f_diffuse = SampledSpectrum{m_unit.dimension()},
                              .pdf_diffuse = 0.0f};
                }
            }
            $else
            {
                if (has_scatter_flag(flags, ScatterFlags::Transmission))
                {
                    auto etap = ite(cos_theta(wo) > 0.0f,
                                    m_ctx.eta_t / m_ctx.eta_i,
                                    m_ctx.eta_i / m_ctx.eta_t);
                    auto wh_v = wo + wi * etap;
                    auto valid = any(wh_v != 0.0f);
                    auto wh = normalize(ite(valid, wh_v, make_float3(0.0f, 0.0f, 1.0f)));
                    wh = ite(wh.z < 0.0f, -wh, wh);
                    auto F = m_fresnel.evaluate(dot(wo, wh));
                    auto p = selection_probability(F, flags, false);
                    auto f = m_microfacet_transmission.evaluate(wo, wi, mode);
                    auto pdf = m_microfacet_transmission.pdf(wo, wi, mode) * p;
                    result = {.f = f * abs_cos_theta(wi), .pdf = pdf,
                              .f_diffuse = SampledSpectrum{m_unit.dimension()},
                              .pdf_diffuse = 0.0f};
                }
            };
        };
        return result;
    }

    [[nodiscard]] Surface::Sample sample(Expr<float3> wo, Expr<float> u_lobe,
                                         Expr<float2> u, TransportMode mode,
                                         ScatterFlags flags) const noexcept
    {
        auto result = Surface::Sample::zero(m_unit.dimension());
        auto f = SampledSpectrum{m_unit.dimension()};
        auto wi = def(make_float3(0.0f, 0.0f, 1.0f));
        auto pdf = def(0.0f);
        auto reflection = def(false);
        auto is_delta = def(false);

        $if(smooth())
        {
            auto F = m_fresnel.evaluate(cos_theta(wo));
            auto pr = selection_probability(F, flags, true);
            auto pt = selection_probability(F, flags, false);
            $if((pr + pt) > 0.0f)
            {
                reflection = u_lobe < pr;
                $if(reflection)
                {
                    f = m_specular_reflection.sample(wo, std::addressof(wi), u,
                                                     std::addressof(pdf), mode);
                    pdf *= pr;
                }
                $else
                {
                    f = m_specular_transmission.sample(wo, std::addressof(wi), u,
                                                       std::addressof(pdf), mode);
                    pdf *= pt;
                };
                is_delta = true;
            };
        }
        $else
        {
            auto wh = m_distribution.sample_wh(wo, u);
            // The distribution faces the sampled normal toward wo, while
            // Fresnel uses the canonical (+z) interface orientation so that
            // the sign selects the incident medium (and detects internal TIR).
            auto wm = ite(wh.z < 0.0f, -wh, wh);
            auto F = m_fresnel.evaluate(dot(wo, wm));
            auto pr = selection_probability(F, flags, true);
            auto pt = selection_probability(F, flags, false);
            $if((pr + pt) > 0.0f)
            {
                reflection = u_lobe < pr;
                $if(reflection)
                {
                    f = m_microfacet_reflection.sample(wo, std::addressof(wi), u,
                                                       std::addressof(pdf), mode);
                    pdf *= pr;
                }
                $else
                {
                    f = m_microfacet_transmission.sample(wo, std::addressof(wi), u,
                                                         std::addressof(pdf), mode);
                    pdf *= pt;
                };
            };
        };

        auto entering = cos_theta(wo) > 0.0f;
        auto eta = ite(reflection, 1.0f,
                       ite(entering, m_ctx.eta_t / m_ctx.eta_i,
                           m_ctx.eta_i / m_ctx.eta_t));
        auto event = ite(reflection, Surface::event_reflect,
                         ite(entering, Surface::event_enter, Surface::event_exit));
        auto valid = (pdf > 0.0f) & (wi.z != 0.0f);
        result = Surface::Sample{
            .eval = {
                .f = ite(valid, f * abs_cos_theta(wi), 0.0f),
                .pdf = ite(valid, pdf, 0.0f),
                .f_diffuse = SampledSpectrum{m_unit.dimension()},
                .pdf_diffuse = 0.0f,
            },
            .wi = wi,
            .event = event,
            .pdf_mis = ite(valid, pdf, 0.0f),
            .delta = is_delta,
            .eta = eta,
        };
        return result;
    }
};

Dielectric::Dielectric(const Texture* roughness, const Texture* u_roughness,
                       const Texture* v_roughness, const Texture* eta,
                       luisa::optional<CauchyEta> cauchy_eta,
                       bool remap_roughness, bool two_sided) noexcept
    : Surface{two_sided}, m_roughness{roughness}, m_u_roughness{u_roughness},
      m_v_roughness{v_roughness}, m_eta{eta}, m_cauchy_eta{std::move(cauchy_eta)},
      m_remap_roughness{remap_roughness} {}

luisa::unique_ptr<Surface::Instance> Dielectric::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    return luisa::make_unique<Instance>(
        renderer, this,
        renderer.build_texture(command_buffer, m_roughness),
        renderer.build_texture(command_buffer, m_u_roughness),
        renderer.build_texture(command_buffer, m_v_roughness),
        renderer.build_texture(command_buffer, m_eta));
}

Dielectric::Instance::Instance(const Renderer& renderer, const Dielectric* surface,
                               const Texture::Instance* roughness,
                               const Texture::Instance* u_roughness,
                               const Texture::Instance* v_roughness,
                               const Texture::Instance* eta) noexcept
    : Surface::Instance{renderer, surface}, m_roughness{roughness},
      m_u_roughness{u_roughness}, m_v_roughness{v_roughness}, m_eta{eta} {}

luisa::unique_ptr<Surface::Closure> Dielectric::Instance::create_closure(
    SampledWavelengths& swl, Expr<float> time) const noexcept
{
    return luisa::make_unique<Closure>(renderer(), swl, time);
}

void Dielectric::Instance::populate_closure(Surface::Closure* closure,
                                             const Interaction& it,
                                             Expr<float3> wo,
                                             Expr<float> eta_i) const noexcept
{
    auto time = closure->time();
    auto roughness = m_roughness ? m_roughness->evaluate(it, time).x : Float{0.0f};
    auto u_roughness = m_u_roughness ? m_u_roughness->evaluate(it, time).x : roughness;
    auto v_roughness = m_v_roughness ? m_v_roughness->evaluate(it, time).x : roughness;
    auto alpha = make_float2(u_roughness, v_roughness);
    if (base<Dielectric>()->remap_roughness())
    {
        alpha = TrowbridgeReitzDistribution::roughness_to_alpha(alpha);
    }
    auto eta_t = def(1.5f);
    auto dispersive = def(false);
    if (auto&& cauchy_eta = base<Dielectric>()->cauchy_eta())
    {
        eta_t = cauchy_eta->evaluate(closure->swl().lambda(0u));
        dispersive = true;
    }
    else if (m_eta)
    {
        eta_t = m_eta->evaluate(it, time).x;
    }
    eta_t = ite(eta_t == 0.0f, 1.0f, eta_t);
    closure->bind(Closure::Context{.it = it, .alpha = alpha,
                                   .eta_i = eta_i, .eta_t = eta_t,
                                   .dispersive = dispersive});
}

Dielectric::Closure::Closure(const Renderer& renderer, const SampledWavelengths& swl,
                             Expr<float> time) noexcept
    : Surface::Closure{renderer, swl, time} {}

Dielectric::Closure::~Closure() noexcept = default;

UInt Dielectric::Closure::lobe_flags() const noexcept
{
    auto&& ctx = context<Context>();
    TrowbridgeReitzDistribution distribution{ctx.alpha};
    auto smooth = (ctx.eta_i == ctx.eta_t) | distribution.effectively_smooth();
    auto directional = ite(ctx.eta_i == ctx.eta_t,
                           Surface::lobe_transmission,
                           Surface::lobe_reflection | Surface::lobe_transmission);
    return directional | ite(smooth, Surface::lobe_delta, Surface::lobe_glossy);
}

void Dielectric::Closure::pre_eval() noexcept
{
    m_impl = luisa::make_unique<Impl>(context<Context>(), swl().dimension());
}

void Dielectric::Closure::post_eval() noexcept
{
    m_impl = nullptr;
}

Surface::Sample Dielectric::Closure::sample_impl(Expr<float3> wo, Expr<float> u_lobe,
                                                  Expr<float2> u, TransportMode mode,
                                                  ScatterFlags flags) const noexcept
{
    auto&& ctx = context<Context>();
    auto wo_local = ctx.it.shading.world_to_local(wo);
    auto result = m_impl->sample(wo_local, u_lobe, u, mode, flags);
    result.wi = ctx.it.shading.local_to_world(result.wi);
    return result;
}

Surface::Evaluation Dielectric::Closure::evaluate_impl(Expr<float3> wo, Expr<float3> wi,
                                                        TransportMode mode,
                                                        ScatterFlags flags) const noexcept
{
    auto&& ctx = context<Context>();
    return m_impl->evaluate(ctx.it.shading.world_to_local(wo),
                            ctx.it.shading.world_to_local(wi), mode, flags);
}

luisa::optional<luisa::string> DielectricSurfaceSpec::validate() const noexcept
{
    if (m_params.eta && m_params.cauchy_eta)
    {
        return spec_validation_error("Dielectric eta texture and Cauchy eta are mutually exclusive.");
    }
    return luisa::nullopt;
}

void DielectricSurfaceSpec::visit_dependencies(SpecDependencyVisitor& visitor) const noexcept
{
    auto visit = [&visitor](const luisa::optional<TextureRef>& ref) noexcept
    {
        if (ref) { visitor.visit(*ref); }
    };
    visit(m_params.roughness);
    visit(m_params.u_roughness);
    visit(m_params.v_roughness);
    visit(m_params.eta);
}

const Surface* DielectricSurfaceSpec::build(SceneBuilder& builder) const noexcept
{
    auto resolve = [&builder](const luisa::optional<TextureRef>& ref) noexcept -> const Texture*
    {
        return ref ? builder.resolve(*ref) : nullptr;
    };
    return builder.emplace<Surface, Dielectric>(
        resolve(m_params.roughness), resolve(m_params.u_roughness),
        resolve(m_params.v_roughness), resolve(m_params.eta),
        m_params.cauchy_eta,
        m_params.remap_roughness, m_params.two_sided);
}
} // namespace Yutrel
