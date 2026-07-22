#include "path.h"

#include <luisa/luisa-compute.h>

#include "base/geometry.h"
#include "base/interaction.h"
#include "base/light_sampler.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/sampling.h"
#include "utils/spectra.h"

namespace Yutrel
{

PathIntegrator::PathIntegrator(uint max_depth) noexcept
    : ProgressiveIntegrator{max_depth}
{
}

PathIntegrator::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const PathIntegrator* integrator, const Sampler* sampler) noexcept
    : ProgressiveIntegrator::Instance{renderer, command_buffer, integrator, sampler}
{
}

luisa::unique_ptr<Integrator::Instance> PathIntegrator::build(Renderer& renderer, CommandBuffer& command_buffer, const Sampler* sampler) const noexcept
{
    return luisa::make_unique<PathIntegrator::Instance>(renderer, command_buffer, this, sampler);
}

const Integrator* PathIntegratorSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Integrator, PathIntegrator>(_max_depth);
}

Float3 PathIntegrator::Instance::Li(const Camera::Instance* camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept
{
    sampler()->start(pixel_id, frame_index);

    auto u_filter = sampler()->generate_pixel_2d();
    auto u_lens   = camera->base()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(0.5f);

    auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);

    auto spectrum = renderer().spectrum();
    auto swl      = spectrum->sample(spectrum->base()->is_fixed() ? 0.0f : sampler()->generate_1d());
    SampledSpectrum Li{swl.dimension(), 0.0f};
    SampledSpectrum beta{swl.dimension(), camera_weight};
    auto eta_scale = def(1.0f);

    auto ray          = camera_ray;
    auto pdf_bsdf     = def(1e16f);
    auto delta_bounce = def(true);
    auto depth        = def(0u);
    $loop
    {
        auto wo                           = -ray->direction();
        luisa::shared_ptr<Interaction> it = renderer().geometry()->intersect(ray);

        $if(!it->is_surface_interaction())
        {
            if (renderer().environment() != nullptr)
            {
                auto eval   = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                auto weight = ite((depth == 0u) | delta_bounce, 1.0f, power_heuristic(pdf_bsdf, eval.pdf));
                Li += beta * eval.L * weight;
            }
            $break;
        };

        $if(!renderer().lights().empty())
        {
            $outline
            {
                $if(it->shape.has_light())
                {
                    auto it_from = Interaction::from_point(ray->origin());
                    auto eval   = light_sampler()->evaluate_hit(*it, it_from, swl, time);
                    auto weight = ite(delta_bounce, 1.0f, power_heuristic(pdf_bsdf, eval.pdf));
                    Li += beta * eval.L * weight;
                };
            };
        };

        $if(depth == max_depth()) { $break; };

        // Null surfaces are transparent boundaries for the surface-only path integrator.
        $if(!it->shape.has_surface())
        {
            ray = it->spawn_ray(ray->direction());
            $continue;
        };

        depth += 1u;

        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface   = sampler()->generate_2d();
        auto light_sample      = LightSampler::Sample::zero(swl.dimension());
        $outline
        {
            light_sample = light_sampler()->sample(*it, u_light_selection, u_light_surface, swl, time);
        };

        auto occluded = def(false);
        if (renderer().has_lighting())
        {
            occluded = renderer().geometry()->intersect_any(light_sample.shadow_ray);
        }

        auto u_lobe = sampler()->generate_1d();
        auto u_bsdf = sampler()->generate_2d();

        $outline
        {
            PolymorphicCall<Surface::Closure> call;
            renderer().surfaces().dispatch(it->shape.surface_tag(), [&](auto surface) noexcept
            {
                surface->closure(call, *it, wo, swl, time, 1.0f);
            });
            call.execute([&](const Surface::Closure* closure) noexcept
            {
                $if(light_sample.eval.pdf > 0.0f & !occluded)
                {
                    auto wi   = light_sample.shadow_ray->direction();
                    auto eval = closure->evaluate(wo, wi);
                    auto mis  = ite(light_sample.delta, 1.0f, power_heuristic(light_sample.eval.pdf, eval.pdf));
                    Li += mis / light_sample.eval.pdf * beta * eval.f * light_sample.eval.L;
                };

                auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                ray                 = it->spawn_ray(surface_sample.wi);
                pdf_bsdf            = surface_sample.pdf_mis;
                delta_bounce        = surface_sample.delta;
                auto w              = ite(surface_sample.eval.pdf > 0.0f, 1.0f / surface_sample.eval.pdf, 0.0f);
                beta *= w * surface_sample.eval.f;
                auto transmission = (surface_sample.event & Surface::event_transmit) != 0u;
                eta_scale *= ite(transmission, sqr(surface_sample.eta), 1.0f);
            });
        };

        auto beta_has_nan = beta.any([](const auto& value) noexcept
        {
            return compute::isnan(value);
        });
        auto beta_has_inf = beta.any([](const auto& value) noexcept
        {
            return compute::isinf(value);
        });
        auto beta_invalid = beta_has_nan | beta_has_inf;
        renderer().record_path_non_finite(beta_has_nan, beta_has_inf);
        beta = beta.map([&](auto value) noexcept
        {
            return ite(beta_invalid, 0.0f, value);
        });
        $if(beta.all([](auto b) noexcept
        {
            return b <= 0.0f;
        }))
        {
            $break;
        };

        auto rr_beta_max = beta.max() * eta_scale;
        $if((depth > 1u) & (rr_beta_max < 1.0f))
        {
            auto q    = max(0.0f, 1.0f - rr_beta_max);
            auto u_rr = sampler()->generate_1d();
            $if(u_rr < q) { $break; };
            beta /= 1.0f - q;
        };
    };

    return spectrum->srgb(swl, Li);
}

} // namespace Yutrel
