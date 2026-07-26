#include "vol_path.h"

#include <limits>

#include <luisa/luisa-compute.h>

#include "base/geometry.h"
#include "base/interaction.h"
#include "base/light_sampler.h"
#include "base/medium.h"
#include "base/phase_function.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/sampling.h"

namespace Yutrel
{
namespace
{
constexpr uint max_tracking_iterations = 256u;

[[nodiscard]] SampledSpectrum clamp_zero(const SampledSpectrum& value) noexcept
{
    return max(value, 0.0f);
}

[[nodiscard]] Bool spectrum_invalid(const SampledSpectrum& value) noexcept
{
    return value.any([](auto v) noexcept
    {
        return compute::isnan(v) | compute::isinf(v);
    });
}

[[nodiscard]] Float safe_average(const SampledSpectrum& value) noexcept
{
    return max(value.average(), 1e-30f);
}
} // namespace

VolPathIntegrator::VolPathIntegrator(uint max_depth) noexcept
    : ProgressiveIntegrator{max_depth}
{
}

VolPathIntegrator::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const VolPathIntegrator* integrator, const Sampler* sampler) noexcept
    : ProgressiveIntegrator::Instance{renderer, command_buffer, integrator, sampler}
{
}

luisa::unique_ptr<Integrator::Instance> VolPathIntegrator::build(Renderer& renderer, CommandBuffer& command_buffer, const Sampler* sampler) const noexcept
{
    return luisa::make_unique<Instance>(renderer, command_buffer, this, sampler);
}

const Integrator* VolPathIntegratorSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Integrator, VolPathIntegrator>(_max_depth);
}

VolPathIntegrator::Instance::TransmittanceResult VolPathIntegrator::Instance::trace_transmittance(
    Var<Ray> ray, UInt current_medium, const SampledWavelengths& swl,
    Expr<float> time, PBRTRNG& rng) const noexcept
{
    TransmittanceResult result{
        .T       = SampledSpectrum{swl.dimension(), 1.0f},
        .r_l     = SampledSpectrum{swl.dimension(), 1.0f},
        .r_u     = SampledSpectrum{swl.dimension(), 1.0f},
        .visible = true,
    };
    auto target         = ray->origin() + ray->direction() * ray->t_max();
    auto finite_target  = ray->t_max() < Interaction::default_t_max * 0.5f;
    auto boundary_count = def(0u);

    $loop
    {
        $if(boundary_count >= max_tracking_iterations)
        {
            result.visible = false;
            $break;
        };
        boundary_count += 1u;

        auto it             = renderer().geometry()->intersect(ray);
        auto segment_length = ite(it->is_surface_interaction(), distance(ray->origin(), it->p_g), ray->t_max());

        if (!renderer().media().empty())
        {
            $if(current_medium != Medium::vacuum_tag)
            {
                auto properties = Medium::Properties::vacuum(swl.dimension());
                auto medium_it  = Interaction::from_point(ray->origin());
                renderer().media().dispatch(current_medium - 1u, [&](auto medium) noexcept
                {
                    properties = medium->properties(medium_it, swl, time);
                });

                $if(properties.sigma_s.is_zero())
                {
                    auto dt = ite(compute::isinf(segment_length), std::numeric_limits<float>::max(), segment_length);
                    result.T *= exp(-dt * properties.sigma_a);
                }
                $else
                {
                    auto sigma_maj_0    = properties.sigma_maj[0u];
                    auto t_min          = def(0.0f);
                    auto T_maj          = SampledSpectrum{swl.dimension(), 1.0f};
                    auto tracking_count = def(0u);
                    auto finished       = def(false);

                    $if(sigma_maj_0 > 0.0f)
                    {
                        auto u = rng.uniform();
                        $loop
                        {
                            $if(tracking_count >= max_tracking_iterations)
                            {
                                result.T = 0.0f;
                                finished = true;
                                $break;
                            };
                            tracking_count += 1u;
                            auto t = t_min - log(max(1e-7f, 1.0f - u)) / sigma_maj_0;
                            u      = rng.uniform();
                            $if(t < segment_length)
                            {
                                T_maj *= exp(-(t - t_min) * properties.sigma_maj);
                                auto sigma_n = clamp_zero(properties.sigma_maj - properties.sigma_a - properties.sigma_s);
                                auto pdf     = T_maj[0u] * sigma_maj_0;
                                auto inv_pdf = ite(pdf > 0.0f, 1.0f / pdf, 0.0f);
                                result.T *= T_maj * sigma_n * inv_pdf;
                                result.r_l *= T_maj * properties.sigma_maj * inv_pdf;
                                result.r_u *= T_maj * sigma_n * inv_pdf;
                                T_maj = 1.0f;
                                t_min = t;

                                auto Tr = result.T / safe_average(result.r_l + result.r_u);
                                $if(Tr.max() < 0.05f)
                                {
                                    constexpr auto q = 0.75f;
                                    $if(rng.uniform() < q) { result.T = 0.0f; }
                                    $else { result.T /= 1.0f - q; };
                                };
                                $if(result.T.is_zero() | spectrum_invalid(result.T))
                                {
                                    finished = true;
                                    $break;
                                };
                            }
                            $else
                            {
                                auto dt = segment_length - t_min;
                                T_maj *= exp(-dt * properties.sigma_maj);
                                finished = true;
                                $break;
                            };
                        };
                    }
                    $else
                    {
                        auto dt = ite(compute::isinf(segment_length), std::numeric_limits<float>::max(), segment_length);
                        T_maj *= exp(-dt * properties.sigma_maj);
                        finished = true;
                    };

                    $if(finished & !result.T.is_zero())
                    {
                        auto inv_t0 = ite(T_maj[0u] > 0.0f, 1.0f / T_maj[0u], 0.0f);
                        result.T *= T_maj * inv_t0;
                        result.r_l *= T_maj * inv_t0;
                        result.r_u *= T_maj * inv_t0;
                    };
                };
            };
        }

        $if(result.T.is_zero() | spectrum_invalid(result.T))
        {
            result.visible = false;
            $break;
        };
        $if(!it->is_surface_interaction()) { $break; };
        $if(it->shape.has_surface())
        {
            result.visible = false;
            $break;
        };

        current_medium = renderer().geometry()->next_medium(*it, ray->direction());
        $if(finite_target) { ray = it->spawn_ray_to(target); }
        $else { ray = it->spawn_ray(ray->direction()); };
    };
    result.T = ite(result.visible, result.T, 0.0f);
    return result;
}

Float3 VolPathIntegrator::Instance::Li(const Camera::Instance* camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept
{
    sampler()->start(pixel_id, frame_index);
    auto u_filter                       = sampler()->generate_pixel_2d();
    auto u_lens                         = camera->base()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(0.5f);
    auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);

    auto spectrum = renderer().spectrum();
    auto swl      = spectrum->sample(spectrum->base()->is_fixed() ? 0.0f : sampler()->generate_1d());
    SampledSpectrum L{swl.dimension(), 0.0f};
    SampledSpectrum beta{swl.dimension(), camera_weight};
    SampledSpectrum r_u{swl.dimension(), 1.0f};
    SampledSpectrum r_l{swl.dimension(), 1.0f};

    auto ray             = camera_ray;
    auto current_medium  = def(Medium::vacuum_tag);
    auto depth           = def(0u);
    auto eta_scale       = def(1.0f);
    auto delta_bounce    = def(false);
    auto path_done       = def(false);
    auto path_iterations = def(0u);

    $loop
    {
        $if(path_iterations >= 4096u) { $break; };
        path_iterations += 1u;

        auto it                = renderer().geometry()->intersect(ray);
        auto segment_length    = ite(it->is_surface_interaction(), distance(ray->origin(), it->p_g), std::numeric_limits<float>::max());
        auto medium_scattered  = def(false);
        auto medium_terminated = def(false);

        if (!renderer().media().empty())
        {
            $if(current_medium != Medium::vacuum_tag)
            {
                auto properties = Medium::Properties::vacuum(swl.dimension());
                auto medium_it  = Interaction::from_point(ray->origin());
                renderer().media().dispatch(current_medium - 1u, [&](auto medium) noexcept
                {
                    properties = medium->properties(medium_it, swl, time);
                });

                $if(properties.sigma_s.is_zero())
                {
                    beta *= exp(-segment_length * properties.sigma_a);
                }
                $else
                {
                    auto seed_u = sampler()->generate_2d();
                    PBRTRNG rng{
                        pbrt_hash64(make_uint4(as<UInt3>(ray->origin()), as<uint>(seed_u.x))),
                        pbrt_hash64(make_uint4(as<UInt3>(ray->direction()), as<uint>(seed_u.y)))};
                    auto T_maj          = SampledSpectrum{swl.dimension(), 1.0f};
                    auto sigma_maj_0    = properties.sigma_maj[0u];
                    auto t_min          = def(0.0f);
                    auto u              = sampler()->generate_1d();
                    auto tracking_count = def(0u);

                    $if(sigma_maj_0 > 0.0f)
                    {
                        $loop
                        {
                            $if(tracking_count >= max_tracking_iterations)
                            {
                                medium_terminated = true;
                                $break;
                            };
                            tracking_count += 1u;
                            auto t = t_min - log(max(1e-7f, 1.0f - u)) / sigma_maj_0;
                            u      = rng.uniform();
                            $if(t < segment_length)
                            {
                                T_maj *= exp(-(t - t_min) * properties.sigma_maj);
                                auto p_absorb  = properties.sigma_a[0u] / sigma_maj_0;
                                auto p_scatter = properties.sigma_s[0u] / sigma_maj_0;
                                auto p_null    = max(0.0f, 1.0f - p_absorb - p_scatter);
                                auto u_mode    = rng.uniform();

                                $if(u_mode < p_absorb)
                                {
                                    medium_terminated = true;
                                    $break;
                                }
                                $elif(u_mode < p_absorb + p_scatter)
                                {
                                    $if(depth >= max_depth())
                                    {
                                        medium_terminated = true;
                                        $break;
                                    };
                                    depth += 1u;
                                    auto pdf     = T_maj[0u] * properties.sigma_s[0u];
                                    auto inv_pdf = ite(pdf > 0.0f, 1.0f / pdf, 0.0f);
                                    beta *= T_maj * properties.sigma_s * inv_pdf;
                                    r_u *= T_maj * properties.sigma_s * inv_pdf;

                                    auto p            = ray->origin() + ray->direction() * t;
                                    auto scatter_it   = Interaction::from_point(p);
                                    auto light_sample = light_sampler()->sample(
                                        scatter_it,
                                        sampler()->generate_1d(),
                                        sampler()->generate_2d(),
                                        swl,
                                        time);
                                    $if(light_sample.eval.pdf > 0.0f)
                                    {
                                        HGPhaseFunction phase{properties.g};
                                        auto wi      = light_sample.shadow_ray->direction();
                                        auto phase_p = phase.p(-ray->direction(), wi);
                                        PBRTRNG shadow_rng{
                                            pbrt_hash64(as<UInt3>(light_sample.shadow_ray->origin())),
                                            pbrt_hash64(as<UInt3>(light_sample.shadow_ray->direction()))};
                                        auto tr         = trace_transmittance(light_sample.shadow_ray, current_medium, swl, time, shadow_rng);
                                        auto direct_r_l = tr.r_l * r_u * light_sample.eval.pdf;
                                        auto direct_r_u = tr.r_u * r_u * phase_p;
                                        auto denom      = ite(light_sample.delta, safe_average(direct_r_l), safe_average(direct_r_l + direct_r_u));
                                        L += beta * phase_p * tr.T * light_sample.eval.L / denom;
                                    };

                                    HGPhaseFunction phase{properties.g};
                                    auto phase_sample  = phase.sample(-ray->direction(), sampler()->generate_2d());
                                    auto inv_phase_pdf = ite(phase_sample.pdf > 0.0f, 1.0f / phase_sample.pdf, 0.0f);
                                    beta *= phase_sample.p * inv_phase_pdf;
                                    r_l              = r_u * inv_phase_pdf;
                                    ray              = make_ray(p, phase_sample.wi);
                                    delta_bounce     = false;
                                    medium_scattered = phase_sample.pdf > 0.0f;
                                    medium_terminated |= phase_sample.pdf <= 0.0f;
                                    $break;
                                }
                                $else
                                {
                                    auto sigma_n = clamp_zero(properties.sigma_maj - properties.sigma_a - properties.sigma_s);
                                    auto pdf     = T_maj[0u] * sigma_n[0u];
                                    auto inv_pdf = ite(pdf > 0.0f, 1.0f / pdf, 0.0f);
                                    beta *= T_maj * sigma_n * inv_pdf;
                                    r_u *= T_maj * sigma_n * inv_pdf;
                                    r_l *= T_maj * properties.sigma_maj * inv_pdf;
                                    medium_terminated |= (pdf <= 0.0f) | (p_null <= 0.0f);
                                    T_maj = 1.0f;
                                    t_min = t;
                                    $if(medium_terminated) { $break; };
                                };
                            }
                            $else
                            {
                                T_maj *= exp(-(segment_length - t_min) * properties.sigma_maj);
                                $break;
                            };
                        };
                    }
                    $else
                    {
                        T_maj *= exp(-segment_length * properties.sigma_maj);
                    };

                    $if(!medium_scattered & !medium_terminated)
                    {
                        auto inv_t0 = ite(T_maj[0u] > 0.0f, 1.0f / T_maj[0u], 0.0f);
                        beta *= T_maj * inv_t0;
                        r_u *= T_maj * inv_t0;
                        r_l *= T_maj * inv_t0;
                    };
                };
            };
        }

        $if(medium_terminated | beta.is_zero() | r_u.is_zero() | spectrum_invalid(beta))
        {
            path_done = true;
        };
        $if(path_done) { $break; };
        $if(medium_scattered) { $continue; };

        $if(!it->is_surface_interaction())
        {
            if (renderer().environment() != nullptr)
            {
                auto eval  = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                auto denom = def(1.0f);
                $if((depth == 0u) | delta_bounce) { denom = safe_average(r_u); }
                $else
                {
                    r_l *= eval.pdf;
                    denom = safe_average(r_u + r_l);
                };
                L += beta * eval.L / denom;
            }
            $break;
        };

        $if(!renderer().lights().empty())
        {
            $if(it->shape.has_light())
            {
                auto it_from = Interaction::from_point(ray->origin());
                auto eval  = light_sampler()->evaluate_hit(*it, it_from, swl, time);
                auto denom = def(1.0f);
                $if((depth == 0u) | delta_bounce) { denom = safe_average(r_u); }
                $else
                {
                    r_l *= eval.pdf;
                    denom = safe_average(r_u + r_l);
                };
                L += beta * eval.L / denom;
            };
        };

        $if(!it->shape.has_surface())
        {
            current_medium = renderer().geometry()->next_medium(*it, ray->direction());
            ray            = it->spawn_ray(ray->direction());
            $continue;
        };

        $if(depth >= max_depth()) { $break; };
        depth += 1u;
        auto wo = -ray->direction();

        $outline
        {
            PolymorphicCall<Surface::Closure> call;
            renderer().surfaces().dispatch(it->shape.surface_tag(), [&](auto surface) noexcept
            {
                surface->closure(call, *it, wo, swl, time, 1.0f);
            });
            call.execute([&](const Surface::Closure* closure) noexcept
            {
                if (auto dispersive = closure->is_dispersive())
                {
                    $if(*dispersive) { swl.terminate_secondary(); };
                }
                auto lobe_flags = closure->lobe_flags();
                auto non_specular =
                    (lobe_flags & (Surface::lobe_diffuse | Surface::lobe_glossy)) != 0u;
                $if(non_specular)
                {
                    auto light_sample = light_sampler()->sample(
                        *it,
                        sampler()->generate_1d(),
                        sampler()->generate_2d(),
                        swl,
                        time);
                    $if(light_sample.eval.pdf > 0.0f)
                    {
                        auto wi   = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        $if(eval.pdf > 0.0f | !eval.f.is_zero())
                        {
                            auto shadow_medium = renderer().geometry()->next_medium(*it, wi);
                            PBRTRNG shadow_rng{
                                pbrt_hash64(as<UInt3>(light_sample.shadow_ray->origin())),
                                pbrt_hash64(as<UInt3>(light_sample.shadow_ray->direction()))};
                            auto tr         = trace_transmittance(light_sample.shadow_ray, shadow_medium, swl, time, shadow_rng);
                            auto direct_r_l = tr.r_l * r_u * light_sample.eval.pdf;
                            auto direct_r_u = tr.r_u * r_u * eval.pdf;
                            auto denom      = ite(light_sample.delta, safe_average(direct_r_l), safe_average(direct_r_l + direct_r_u));
                            L += beta * eval.f * tr.T * light_sample.eval.L / denom;
                        };
                    };
                };

                auto surface_sample = closure->sample(
                    wo,
                    sampler()->generate_1d(),
                    sampler()->generate_2d());
                auto inv_pdf = ite(surface_sample.eval.pdf > 0.0f, 1.0f / surface_sample.eval.pdf, 0.0f);
                beta *= surface_sample.eval.f * inv_pdf;
                auto inv_pdf_mis  = ite(surface_sample.pdf_mis > 0.0f, 1.0f / surface_sample.pdf_mis, 0.0f);
                r_l               = r_u * inv_pdf_mis;
                delta_bounce      = surface_sample.delta;
                auto transmission = (surface_sample.event & Surface::event_transmit) != 0u;
                eta_scale *= ite(transmission, sqr(surface_sample.eta), 1.0f);
                current_medium = renderer().geometry()->next_medium(*it, surface_sample.wi);
                ray            = it->spawn_ray(surface_sample.wi);
            });
        };

        auto beta_has_nan = beta.any([](auto v) noexcept
        {
            return compute::isnan(v);
        });
        auto beta_has_inf = beta.any([](auto v) noexcept
        {
            return compute::isinf(v);
        });
        renderer().record_path_non_finite(beta_has_nan, beta_has_inf);
        $if(beta.is_zero() | r_u.is_zero() | beta_has_nan | beta_has_inf) { $break; };

        auto rr_beta_max = beta.max() * eta_scale / safe_average(r_u);
        auto u_rr        = sampler()->generate_1d();
        $if((depth > 1u) & (rr_beta_max < 1.0f))
        {
            auto q = max(0.0f, 1.0f - rr_beta_max);
            $if(u_rr < q) { $break; };
            beta /= 1.0f - q;
        };
    };

    return spectrum->srgb(swl, L);
}

} // namespace Yutrel
