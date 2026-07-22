#include "layered.h"

#include <limits>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>

#include "base/phase_function.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/rng.h"
#include "utils/sampling.h"

namespace Yutrel
{
namespace
{
[[nodiscard]] inline Float transmittance(Expr<float> dz, Expr<float3> w) noexcept
{
    return ite(abs(dz) <= std::numeric_limits<float>::min(),
               1.0f,
               exp(-abs(dz / w.z)));
}

[[nodiscard]] inline Bool is_delta_lobe(Expr<uint> flags) noexcept
{
    return ((flags & Surface::lobe_delta) != 0u) &
           ((flags & (Surface::lobe_diffuse | Surface::lobe_glossy)) == 0u);
}

class TopOrBottom
{
private:
    const Surface::Closure* m_top;
    const Surface::Closure* m_bottom;
    Bool m_is_top;

public:
    TopOrBottom(const Surface::Closure* top, const Surface::Closure* bottom,
                Expr<bool> is_top) noexcept
        : m_top{top}, m_bottom{bottom}, m_is_top{is_top} {}

    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode,
                                               ScatterFlags flags = ScatterFlags::All) const noexcept
    {
        auto result = Surface::Evaluation::zero(m_top->swl().dimension());
        $if(m_is_top) { result = m_top->evaluate(wo, wi, mode, flags); }
        $else { result = m_bottom->evaluate(wo, wi, mode, flags); };
        return result;
    }

    [[nodiscard]] Surface::Sample sample(Expr<float3> wo, Expr<float> u_lobe,
                                         Expr<float2> u, TransportMode mode,
                                         ScatterFlags flags = ScatterFlags::All) const noexcept
    {
        auto result = Surface::Sample::zero(m_top->swl().dimension());
        $if(m_is_top) { result = m_top->sample(wo, u_lobe, u, mode, flags); }
        $else { result = m_bottom->sample(wo, u_lobe, u, mode, flags); };
        return result;
    }

    [[nodiscard]] UInt lobe_flags() const noexcept
    {
        return ite(m_is_top, m_top->lobe_flags(), m_bottom->lobe_flags());
    }

    [[nodiscard]] Float3 to_local(Expr<float3> w) const noexcept
    {
        return ite(m_is_top,
                   m_top->it().shading.world_to_local(w),
                   m_bottom->it().shading.world_to_local(w));
    }

    [[nodiscard]] Float3 to_world(Expr<float3> w) const noexcept
    {
        return ite(m_is_top,
                   m_top->it().shading.local_to_world(w),
                   m_bottom->it().shading.local_to_world(w));
    }

    [[nodiscard]] SampledSpectrum unprojected_f(const Surface::Evaluation& eval,
                                                Expr<float3> wi) const noexcept
    {
        return eval.f / max(abs_cos_theta(to_local(wi)), 1e-30f);
    }
};
} // namespace

class Layered::Closure::Impl
{
private:
    const Context& m_ctx;
    const Surface::Closure* m_top;
    const Surface::Closure* m_bottom;

    [[nodiscard]] static TransportMode reverse_mode(TransportMode mode) noexcept
    {
        return mode == TransportMode::RADIANCE ? TransportMode::IMPORTANCE : TransportMode::RADIANCE;
    }

public:
    Impl(const Context& ctx, const Surface::Closure* top,
         const Surface::Closure* bottom) noexcept
        : m_ctx{ctx}, m_top{top}, m_bottom{bottom} {}

    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi,
                            TransportMode mode) const noexcept
    {
        auto&& it        = m_ctx.it;
        auto wo_local    = it.shading.world_to_local(wo);
        auto wi_local    = it.shading.world_to_local(wi);
        auto entered_top = wo_local.z > 0.0f;
        auto same        = same_hemisphere(wo_local, wi_local);
        auto pdf_sum     = def(0.0f);
        $if(same)
        {
            auto enter = TopOrBottom{m_top, m_bottom, entered_top};
            pdf_sum    = Float{m_ctx.samples} *
                         enter.evaluate(wo, wi, mode, ScatterFlags::Reflection).pdf;
        };

        PBRTRNG rng{pbrt_hash64(make_uint4(0u, as<UInt3>(wi_local))),
                    pbrt_hash64(as<UInt3>(wo_local))};
        $for(sample_index, m_ctx.samples)
        {
            $if(same)
            {
                auto r_interface = TopOrBottom{m_top, m_bottom, !entered_top};
                auto t_interface = TopOrBottom{m_top, m_bottom, entered_top};
                auto wos         = t_interface.sample(wo, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), mode, ScatterFlags::Transmission);
                auto wis         = t_interface.sample(wi, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), reverse_mode(mode), ScatterFlags::Transmission);
                $if((wos.eval.pdf <= 0.0f) | (wis.eval.pdf <= 0.0f)) { $continue; };
                $if(is_delta_lobe(t_interface.lobe_flags()))
                {
                    pdf_sum += r_interface.evaluate(-wos.wi, -wis.wi, mode, ScatterFlags::Reflection).pdf;
                }
                $else
                {
                    auto rs = r_interface.sample(-wos.wi, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), mode, ScatterFlags::Reflection);
                    $if(rs.eval.pdf > 0.0f)
                    {
                        $if(is_delta_lobe(r_interface.lobe_flags()))
                        {
                            pdf_sum += t_interface.evaluate(-rs.wi, wi, mode, ScatterFlags::Transmission).pdf;
                        }
                        $else
                        {
                            auto r_pdf = r_interface.evaluate(-wos.wi, -wis.wi, mode, ScatterFlags::Reflection).pdf;
                            pdf_sum += power_heuristic(wis.eval.pdf, r_pdf) * r_pdf;
                            auto t_pdf = t_interface.evaluate(-rs.wi, wi, mode, ScatterFlags::Transmission).pdf;
                            pdf_sum += power_heuristic(rs.eval.pdf, t_pdf) * t_pdf;
                        };
                    };
                };
            }
            $else
            {
                auto to_interface = TopOrBottom{m_top, m_bottom, entered_top};
                auto ti_interface = TopOrBottom{m_top, m_bottom, !entered_top};
                auto wos          = to_interface.sample(wo, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), mode);
                $if((wos.eval.pdf <= 0.0f) | (to_interface.to_local(wos.wi).z == 0.0f) |
                    ((wos.event & event_reflect) != 0u)) { $continue; };
                auto wis = ti_interface.sample(wi, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), reverse_mode(mode));
                $if((wis.eval.pdf <= 0.0f) | (ti_interface.to_local(wis.wi).z == 0.0f) |
                    ((wis.event & event_reflect) != 0u)) { $continue; };
                $if(is_delta_lobe(to_interface.lobe_flags()))
                {
                    pdf_sum += ti_interface.evaluate(-wos.wi, wi, mode, ScatterFlags::Transmission).pdf;
                }
                $elif(is_delta_lobe(ti_interface.lobe_flags()))
                {
                    pdf_sum += to_interface.evaluate(wo, -wis.wi, mode, ScatterFlags::Transmission).pdf;
                }
                $else
                {
                    pdf_sum += 0.5f *
                               (to_interface.evaluate(wo, -wis.wi, mode, ScatterFlags::Transmission).pdf +
                                ti_interface.evaluate(-wos.wi, wi, mode, ScatterFlags::Transmission).pdf);
                };
            };
        };
        return lerp(0.25f * inv_pi, pdf_sum / Float{m_ctx.samples}, 0.9f);
    }

    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode,
                                               ScatterFlags requested_flags) const noexcept
    {
        auto result   = Surface::Evaluation::zero(m_ctx.albedo.dimension());
        auto&& it     = m_ctx.it;
        auto wo_local = it.shading.world_to_local(wo);
        auto wi_local = it.shading.world_to_local(wi);
        auto same     = same_hemisphere(wo_local, wi_local);

        auto entered_top       = wo_local.z > 0.0f;
        auto enter_interface   = TopOrBottom{m_top, m_bottom, entered_top};
        auto exit_interface    = TopOrBottom{m_bottom, m_top, same ^ entered_top};
        auto nonexit_interface = TopOrBottom{m_top, m_bottom, same ^ entered_top};
        auto exit_z            = ite(same ^ entered_top, 0.0f, m_ctx.thickness);
        auto enter_eval        = enter_interface.evaluate(wo, wi, mode);
        auto f                 = ite(same,
                                     Float{m_ctx.samples} * enter_interface.unprojected_f(enter_eval, wi),
                                     SampledSpectrum{m_ctx.albedo.dimension()});
        PBRTRNG rng{pbrt_hash64(make_uint4(0u, as<UInt3>(wo_local))),
                    pbrt_hash64(as<UInt3>(wi_local))};
        HGPhaseFunction phase{m_ctx.g};

        $for(sample_index, m_ctx.samples)
        {
            auto wos = enter_interface.sample(wo, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), mode, ScatterFlags::Transmission);
            auto wis = exit_interface.sample(wi, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), reverse_mode(mode), ScatterFlags::Transmission);
            $if((wos.eval.pdf <= 0.0f) | (wis.eval.pdf <= 0.0f)) { $continue; };

            auto beta      = wos.eval.f / wos.eval.pdf;
            auto z         = ite(entered_top, m_ctx.thickness, 0.0f);
            auto w         = def(wos.wi);
            auto w_local   = enter_interface.to_local(w);
            auto wis_local = exit_interface.to_local(wis.wi);
            auto wis_f     = exit_interface.unprojected_f(wis.eval, wis.wi);

            $for(depth, m_ctx.max_depth)
            {
                $if((depth > 3u) & (beta.max() < 0.25f))
                {
                    auto q = max(0.0f, 1.0f - beta.max());
                    $if(rng.uniform() < q) { $break; };
                    beta /= 1.0f - q;
                };
                $if(m_ctx.albedo.is_zero())
                {
                    z = ite(z == m_ctx.thickness, 0.0f, m_ctx.thickness);
                    beta *= transmittance(m_ctx.thickness, w_local);
                }
                $else
                {
                    auto dz = sample_exponential(rng.uniform(), 1.0f / abs(w_local.z));
                    auto zp = ite(w_local.z > 0.0f, z + dz, z - dz);
                    $if(zp == z) { $continue; };
                    $if((zp > 0.0f) & (zp < m_ctx.thickness))
                    {
                        auto phase_pdf = phase.p(-w_local, -wis_local);
                        auto wt        = ite(is_delta_lobe(exit_interface.lobe_flags()), 1.0f, power_heuristic(wis.eval.pdf, phase_pdf));
                        f += beta * m_ctx.albedo * phase_pdf * wt *
                             transmittance(zp - exit_z, wis_local) *
                             wis_f / wis.eval.pdf;
                        auto ps = phase.sample(-w_local, make_float2(rng.uniform(), rng.uniform()));
                        $if((ps.pdf <= 0.0f) | (ps.wi.z == 0.0f)) { $continue; };
                        beta *= m_ctx.albedo * ps.p / ps.pdf;
                        w_local = ps.wi;
                        w       = exit_interface.to_world(w_local);
                        z       = zp;
                        $if((((z < exit_z) & (w_local.z > 0.0f)) |
                             ((z > exit_z) & (w_local.z < 0.0f))) &
                            !is_delta_lobe(exit_interface.lobe_flags()))
                        {
                            auto exit    = exit_interface.evaluate(-w, wi, mode, ScatterFlags::Transmission);
                            auto wt_exit = power_heuristic(ps.pdf, exit.pdf);
                            f += beta * transmittance(zp - exit_z, w_local) *
                                 exit_interface.unprojected_f(exit, wi) * wt_exit;
                        };
                        $continue;
                    };
                    z = clamp(zp, 0.0f, m_ctx.thickness);
                };

                $if(z == exit_z)
                {
                    auto bs = exit_interface.sample(-w, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), mode, ScatterFlags::Reflection);
                    $if(bs.eval.pdf <= 0.0f) { $break; };
                    beta *= bs.eval.f / bs.eval.pdf;
                    w       = bs.wi;
                    w_local = exit_interface.to_local(w);
                }
                $else
                {
                    $if(!is_delta_lobe(nonexit_interface.lobe_flags()))
                    {
                        auto nee = nonexit_interface.evaluate(-w, -wis.wi, mode, ScatterFlags::Reflection);
                        auto wt  = ite(is_delta_lobe(exit_interface.lobe_flags()), 1.0f, power_heuristic(wis.eval.pdf, nee.pdf));
                        f += beta * nee.f * wt * transmittance(m_ctx.thickness, wis_local) *
                             wis_f / wis.eval.pdf;
                    };
                    auto bs = nonexit_interface.sample(-w, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), mode, ScatterFlags::Reflection);
                    $if(bs.eval.pdf <= 0.0f) { $break; };
                    beta *= bs.eval.f / bs.eval.pdf;
                    w       = bs.wi;
                    w_local = nonexit_interface.to_local(w);
                    $if(!is_delta_lobe(exit_interface.lobe_flags()))
                    {
                        auto exit = exit_interface.evaluate(-w, wi, mode, ScatterFlags::Transmission);
                        auto wt   = ite(is_delta_lobe(nonexit_interface.lobe_flags()), 1.0f, power_heuristic(bs.eval.pdf, exit.pdf));
                        f += beta * transmittance(m_ctx.thickness, w_local) *
                             exit_interface.unprojected_f(exit, wi) * wt;
                    };
                };
            };
        };

        auto allowed = ite(same,
                           has_scatter_flag(requested_flags, ScatterFlags::Reflection),
                           has_scatter_flag(requested_flags, ScatterFlags::Transmission));
        result.f     = ite(allowed, f * abs_cos_theta(wi_local) / Float{m_ctx.samples}, 0.0f);
        result.pdf   = ite(allowed, pdf(wo, wi, mode), 0.0f);
        return result;
    }

    [[nodiscard]] Surface::Sample sample(Expr<float3> wo, Expr<float> u_lobe,
                                         Expr<float2> u, TransportMode mode,
                                         ScatterFlags requested_flags) const noexcept
    {
        auto result      = Surface::Sample::zero(m_ctx.albedo.dimension());
        auto&& it        = m_ctx.it;
        auto wo_local    = it.shading.world_to_local(wo);
        auto entered_top = wo_local.z > 0.0f;
        auto entrance    = TopOrBottom{m_top, m_bottom, entered_top};
        auto bs          = entrance.sample(wo, u_lobe, u, mode);
        $if(bs.eval.pdf > 0.0f)
        {
            auto wi_local = it.shading.world_to_local(bs.wi);
            $if(same_hemisphere(wo_local, wi_local))
            {
                result = bs;
            }
            $else
            {
                auto w       = def(bs.wi);
                auto w_local = entrance.to_local(w);
                PBRTRNG rng{pbrt_hash64(make_uint4(0u, as<UInt3>(wo_local))),
                            pbrt_hash64(as<UInt3>(make_float3(u_lobe, u)))};
                auto f             = bs.eval.f;
                auto path_pdf      = def(bs.eval.pdf);
                auto z             = ite(entered_top, m_ctx.thickness, 0.0f);
                auto specular_path = def(bs.delta);
                HGPhaseFunction phase{m_ctx.g};

                $for(depth, m_ctx.max_depth)
                {
                    auto rr_beta = f.max() / path_pdf;
                    $if((depth > 3u) & (rr_beta < 0.25f))
                    {
                        auto q = max(0.0f, 1.0f - rr_beta);
                        $if(rng.uniform() < q) { $break; };
                        path_pdf *= 1.0f - q;
                    };
                    $if(w_local.z == 0.0f) { $break; };
                    $if(!m_ctx.albedo.is_zero())
                    {
                        auto dz = sample_exponential(rng.uniform(), 1.0f / abs(w_local.z));
                        auto zp = ite(w_local.z > 0.0f, z + dz, z - dz);
                        $if(zp == z) { $break; };
                        $if((zp > 0.0f) & (zp < m_ctx.thickness))
                        {
                            auto ps = phase.sample(-w_local, make_float2(rng.uniform(), rng.uniform()));
                            $if((ps.pdf <= 0.0f) | (ps.wi.z == 0.0f)) { $break; };
                            f *= m_ctx.albedo * ps.p;
                            path_pdf *= ps.pdf;
                            specular_path = false;
                            w_local       = ps.wi;
                            w             = it.shading.local_to_world(w_local);
                            z             = zp;
                            $continue;
                        };
                        z = clamp(zp, 0.0f, m_ctx.thickness);
                    }
                    $else
                    {
                        z = ite(z == m_ctx.thickness, 0.0f, m_ctx.thickness);
                        f *= transmittance(m_ctx.thickness, w_local);
                    };

                    auto interface = TopOrBottom{m_bottom, m_top, z == 0.0f};
                    auto next      = interface.sample(-w, rng.uniform(), make_float2(rng.uniform(), rng.uniform()), mode);
                    $if(next.eval.pdf <= 0.0f) { $break; };
                    f *= next.eval.f;
                    path_pdf *= next.eval.pdf;
                    specular_path &= next.delta;
                    w       = next.wi;
                    w_local = it.shading.world_to_local(w);
                    $if((next.event & Surface::event_transmit) != 0u)
                    {
                        auto same_outer = same_hemisphere(wo_local, w_local);
                        auto event      = ite(same_outer, Surface::event_reflect, ite(w_local.z > 0.0f, Surface::event_exit, Surface::event_enter));
                        result          = Surface::Sample{
                            .eval    = {.f = f, .pdf = path_pdf, .f_diffuse = SampledSpectrum{m_ctx.albedo.dimension()}, .pdf_diffuse = 0.0f},
                            .wi      = w,
                            .event   = event,
                            .pdf_mis = 0.0f,
                            .delta   = specular_path,
                            .eta     = 1.0f};
                        $break;
                    };
                };
            };
        };

        auto result_local = it.shading.world_to_local(result.wi);
        auto allowed      = ite(same_hemisphere(wo_local, result_local),
                                has_scatter_flag(requested_flags, ScatterFlags::Reflection),
                                has_scatter_flag(requested_flags, ScatterFlags::Transmission));
        result.eval.f     = ite(allowed, result.eval.f, 0.0f);
        result.eval.pdf   = ite(allowed, result.eval.pdf, 0.0f);
        result.pdf_mis    = ite(allowed & (result.eval.pdf > 0.0f),
                                pdf(wo, result.wi, mode),
                                0.0f);
        return result;
    }
};

Layered::Layered(const Surface* top, const Surface* bottom,
                 const Texture* thickness, const Texture* albedo, const Texture* g,
                 uint max_depth, uint samples, bool two_sided) noexcept
    : Surface{two_sided}, m_top{top}, m_bottom{bottom}, m_thickness{thickness},
      m_albedo{albedo}, m_g{g}, m_max_depth{max_depth}, m_samples{samples}
{
    LUISA_ASSERT(m_top != nullptr && m_bottom != nullptr,
                 "Layered surface requires non-null top and bottom surfaces.");
    LUISA_ASSERT(m_top->is_transmissive() || m_bottom->is_transmissive(),
                 "At least one Layered interface must be transmissive.");
}

uint Layered::properties() const noexcept
{
    auto properties = 0u;
    if (m_top->is_reflective() || m_bottom->is_reflective())
    {
        properties |= property_reflective;
    }
    if (m_top->is_transmissive() && m_bottom->is_transmissive())
    {
        properties |= property_transmissive;
    }
    return properties;
}

luisa::unique_ptr<Surface::Instance> Layered::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    return luisa::make_unique<Instance>(
        renderer,
        this,
        m_top->build(renderer, command_buffer),
        m_bottom->build(renderer, command_buffer),
        renderer.build_texture(command_buffer, m_thickness),
        renderer.build_texture(command_buffer, m_albedo),
        renderer.build_texture(command_buffer, m_g));
}

Layered::Instance::Instance(const Renderer& renderer, const Layered* surface,
                            luisa::unique_ptr<Surface::Instance> top,
                            luisa::unique_ptr<Surface::Instance> bottom,
                            const Texture::Instance* thickness,
                            const Texture::Instance* albedo,
                            const Texture::Instance* g) noexcept
    : Surface::Instance{renderer, surface}, m_top{std::move(top)},
      m_bottom{std::move(bottom)}, m_thickness{thickness},
      m_albedo{albedo}, m_g{g} {}

luisa::string Layered::Instance::closure_identifier() const noexcept
{
    return luisa::format("Layered<{},{}>", m_top->closure_identifier(), m_bottom->closure_identifier());
}

luisa::unique_ptr<Surface::Closure> Layered::Instance::create_closure(
    SampledWavelengths& swl, Expr<float> time) const noexcept
{
    return luisa::make_unique<Closure>(
        renderer(),
        swl,
        time,
        m_top->create_closure(swl, time),
        m_bottom->create_closure(swl, time));
}

void Layered::Instance::populate_closure(Surface::Closure* closure_in,
                                         const Interaction& it,
                                         Expr<float3> wo,
                                         Expr<float> eta_i) const noexcept
{
    auto closure   = static_cast<Closure*>(closure_in);
    auto time      = closure->time();
    auto dimension = closure->swl().dimension();
    auto thickness = m_thickness ? max(m_thickness->evaluate(it, time).x,
                                       std::numeric_limits<float>::min())
                                 : Float{0.01f};
    auto albedo    = m_albedo ? m_albedo->evaluate_albedo_spectrum(it, closure->swl(), time).value : SampledSpectrum{dimension};
    auto g         = m_g ? clamp(m_g->evaluate(it, time).x, -1.0f, 1.0f) : Float{0.0f};
    closure->bind(Closure::Context{
        .it        = it,
        .thickness = thickness,
        .albedo    = albedo,
        .g         = g,
        .max_depth = base<Layered>()->max_depth(),
        .samples   = base<Layered>()->samples()});

    m_top->populate_closure(closure->top(), it, wo, eta_i);
    auto eta_between = closure->top()->eta().value_or(eta_i);
    m_bottom->populate_closure(closure->bottom(), it, wo, eta_between);
}

Layered::Closure::Closure(const Renderer& renderer, const SampledWavelengths& swl,
                          Expr<float> time,
                          luisa::unique_ptr<Surface::Closure> top,
                          luisa::unique_ptr<Surface::Closure> bottom) noexcept
    : Surface::Closure{renderer, swl, time}, m_top{std::move(top)},
      m_bottom{std::move(bottom)} {}

Layered::Closure::~Closure() noexcept = default;

UInt Layered::Closure::lobe_flags() const noexcept
{
    auto top_flags         = m_top->lobe_flags();
    auto bottom_flags      = m_bottom->lobe_flags();
    auto both_transmissive = ((top_flags & Surface::lobe_transmission) != 0u) &
                             ((bottom_flags & Surface::lobe_transmission) != 0u);
    auto directional       = Surface::lobe_reflection |
                             ite(both_transmissive, Surface::lobe_transmission, 0u);
    auto has_diffuse       = ((top_flags | bottom_flags) & Surface::lobe_diffuse) != 0u |
                             !context<Context>().albedo.is_zero();
    auto has_glossy        = ((top_flags | bottom_flags) & Surface::lobe_glossy) != 0u;
    auto type              = ite(has_diffuse, Surface::lobe_diffuse, ite(has_glossy, Surface::lobe_glossy, Surface::lobe_delta));
    return directional | type;
}

void Layered::Closure::pre_eval() noexcept
{
    m_top->pre_eval();
    m_bottom->pre_eval();
    m_impl = luisa::make_unique<Impl>(context<Context>(), m_top.get(), m_bottom.get());
}

void Layered::Closure::post_eval() noexcept
{
    m_impl = nullptr;
    m_bottom->post_eval();
    m_top->post_eval();
}

Surface::Sample Layered::Closure::sample_impl(Expr<float3> wo, Expr<float> u_lobe,
                                              Expr<float2> u, TransportMode mode,
                                              ScatterFlags flags) const noexcept
{
    return m_impl->sample(wo, u_lobe, u, mode, flags);
}

Surface::Evaluation Layered::Closure::evaluate_impl(Expr<float3> wo, Expr<float3> wi,
                                                    TransportMode mode,
                                                    ScatterFlags flags) const noexcept
{
    return m_impl->evaluate(wo, wi, mode, flags);
}

luisa::optional<luisa::string> LayeredSurfaceSpec::validate() const noexcept
{
    if (m_params.max_depth == 0u)
    {
        return spec_validation_error("Layered max_depth must be positive.");
    }
    if (m_params.samples == 0u)
    {
        return spec_validation_error("Layered samples must be positive.");
    }
    return luisa::nullopt;
}

void LayeredSurfaceSpec::visit_dependencies(SpecDependencyVisitor& visitor) const noexcept
{
    visitor.visit(m_params.top);
    visitor.visit(m_params.bottom);
    if (m_params.thickness)
    {
        visitor.visit(*m_params.thickness);
    }
    if (m_params.albedo)
    {
        visitor.visit(*m_params.albedo);
    }
    if (m_params.g)
    {
        visitor.visit(*m_params.g);
    }
}

const Surface* LayeredSurfaceSpec::build(SceneBuilder& builder) const noexcept
{
    auto resolve = [&builder](const luisa::optional<TextureRef>& ref) noexcept -> const Texture*
    {
        return ref ? builder.resolve(*ref) : nullptr;
    };
    return builder.emplace<Surface, Layered>(
        builder.resolve(m_params.top),
        builder.resolve(m_params.bottom),
        resolve(m_params.thickness),
        resolve(m_params.albedo),
        resolve(m_params.g),
        m_params.max_depth,
        m_params.samples,
        m_params.two_sided);
}
} // namespace Yutrel
