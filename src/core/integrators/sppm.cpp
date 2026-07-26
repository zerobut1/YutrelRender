#include "sppm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>

#include <luisa/luisa-compute.h>

#include "base/camera.h"
#include "base/film.h"
#include "base/geometry.h"
#include "base/interaction.h"
#include "base/light_sampler.h"
#include "base/renderer.h"
#include "base/sampler.h"
#include "scene/scene_builder.h"
#include "utils/image_io.h"
#include "utils/progress_bar.h"
#include "utils/rng.h"
#include "utils/sampling.h"
#include "utils/spectra.h"

namespace Yutrel
{

SPPMIntegrator::SPPMIntegrator(uint max_depth, uint photons_per_iteration, float initial_radius) noexcept
    : _max_depth{max_depth},
      _photons_per_iteration{photons_per_iteration},
      _initial_radius{initial_radius}
{
}

SPPMIntegrator::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const SPPMIntegrator* integrator, const Sampler* sampler) noexcept
    : Integrator::Instance{renderer, command_buffer, integrator, sampler}
{
    // Register light handle buffer for photon emission sampling
    auto light_instances_span = renderer.geometry()->light_instances();
    _n_lights                 = static_cast<uint>(light_instances_span.size());
    if (_n_lights > 0u)
    {
        auto [view, buffer_id] = renderer.bindless_arena_buffer<Light::Handle>(_n_lights);
        _light_handle_buffer_id = buffer_id;
        command_buffer
            << view.copy_from(light_instances_span.data())
            << commit();
    }
}

luisa::unique_ptr<Integrator::Instance> SPPMIntegrator::build(Renderer& renderer, CommandBuffer& command_buffer, const Sampler* sampler) const noexcept
{
    return luisa::make_unique<SPPMIntegrator::Instance>(renderer, command_buffer, this, sampler);
}

const Integrator* SPPMIntegratorSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Integrator, SPPMIntegrator>(_max_depth, _photons_per_iteration, _initial_radius);
}

void SPPMIntegrator::Instance::render_interactive(Stream& /*stream*/)
{
    LUISA_ERROR_WITH_LOCATION("SPPM does not support interactive rendering.");
}

void SPPMIntegrator::Instance::render(Stream& stream, bool enable_display)
{
    CommandBuffer command_buffer{stream};

    auto camera      = renderer().camera();
    auto resolution  = camera->film()->base()->resolution();
    auto pixel_count = resolution.x * resolution.y;

    auto iterations       = sampler()->base()->spp();
    auto photons_per_iter = photons_per_iteration();
    auto depth_limit      = max_depth();
    auto r_initial        = initial_radius();

    // SPPM progressive shrinking exponent from Hachisuka et al. 2008, matching
    // pbrt-v4's `Float gamma = (Float)2 / (Float)3;`.
    constexpr float gamma_sppm = 2.0f / 3.0f;
    constexpr float pi_f       = 3.14159265358979323846f;

    if (_n_lights == 0u)
    {
        LUISA_ERROR_WITH_LOCATION("SPPM requires at least one area light source in the scene.");
    }

    LUISA_INFO(
        "SPPM: resolution={}x{}, iterations={}, photons_per_iter={}, max_depth={}, initial_radius={}.",
        resolution.x, resolution.y, iterations, photons_per_iter, depth_limit, r_initial);

    // Photon records are stored in a hash grid whose cell size is fixed at the
    // initial radius. Per-pixel radii only ever shrink, so a 3x3x3 neighbourhood
    // search around a query point always covers its full gather sphere.
    auto grid_cell_size = r_initial;

    // A photon can be deposited at every surface hit past the first one, but
    // allocating `photons_per_iter * max_depth` records is not affordable for
    // large photon counts. Use a bounded heuristic and report overflow instead.
    constexpr auto capacity_depth_budget = 6u;
    auto capacity_64                     = static_cast<uint64_t>(photons_per_iter) *
                       static_cast<uint64_t>(std::min(depth_limit, capacity_depth_budget));
    capacity_64   = std::max<uint64_t>(capacity_64, photons_per_iter);
    capacity_64   = std::min<uint64_t>(capacity_64, 64u * 1024u * 1024u);
    auto capacity = static_cast<uint>(capacity_64);
    auto hash_size = static_cast<uint>(std::bit_ceil(std::max(2u * photons_per_iter, 1u)));

    LUISA_INFO("SPPM: photon buffer capacity={}, hash_size={}, grid_cell_size={}.",
               capacity, hash_size, grid_cell_size);

    // Per-pixel progressive state.
    auto buf_radius   = renderer().create<Buffer<float>>(pixel_count);
    auto buf_N        = renderer().create<Buffer<float>>(pixel_count);
    auto buf_tau_r    = renderer().create<Buffer<float>>(pixel_count);
    auto buf_tau_g    = renderer().create<Buffer<float>>(pixel_count);
    auto buf_tau_b    = renderer().create<Buffer<float>>(pixel_count);
    auto buf_direct_r = renderer().create<Buffer<float>>(pixel_count);
    auto buf_direct_g = renderer().create<Buffer<float>>(pixel_count);
    auto buf_direct_b = renderer().create<Buffer<float>>(pixel_count);
    auto buf_M        = renderer().create<Buffer<uint>>(pixel_count);
    auto buf_phi_r    = renderer().create<Buffer<float>>(pixel_count);
    auto buf_phi_g    = renderer().create<Buffer<float>>(pixel_count);
    auto buf_phi_b    = renderer().create<Buffer<float>>(pixel_count);

    // Photon map.
    auto buf_photon_pos    = renderer().create<Buffer<float4>>(capacity);
    auto buf_photon_wi     = renderer().create<Buffer<float4>>(capacity);
    auto buf_photon_beta_s = renderer().create<Buffer<float4>>(capacity);
    auto buf_photon_next   = renderer().create<Buffer<int>>(capacity);
    auto buf_grid_head     = renderer().create<Buffer<int>>(hash_size);
    // [0] = photon records requested this iteration (may exceed capacity).
    auto buf_photon_count = renderer().create<Buffer<uint>>(1u);

    auto light_handle_buffer_id = _light_handle_buffer_id;
    auto n_lights               = _n_lights;

    // Photon throughput is packed into a float4, so the spectrum must not have
    // more than four components (sRGB has 3, hero-wavelength has 4).
    auto spectrum_dimension = renderer().spectrum()->base()->dimension();
    auto spectrum_is_fixed  = renderer().spectrum()->base()->is_fixed();
    LUISA_ASSERT(spectrum_dimension <= 4u,
                 "SPPM packs photon throughput into a float4 and cannot handle a "
                 "spectrum of dimension {}.",
                 spectrum_dimension);

    // ============================================================
    // KERNEL: Initialize per-pixel state
    // ============================================================
    Kernel1D init_kernel = [&](BufferFloat radius, BufferFloat N,
                               BufferFloat tau_r, BufferFloat tau_g, BufferFloat tau_b,
                               BufferFloat direct_r, BufferFloat direct_g, BufferFloat direct_b) noexcept
    {
        auto i = dispatch_x();
        radius.write(i, r_initial);
        N.write(i, 0.0f);
        tau_r.write(i, 0.0f);
        tau_g.write(i, 0.0f);
        tau_b.write(i, 0.0f);
        direct_r.write(i, 0.0f);
        direct_g.write(i, 0.0f);
        direct_b.write(i, 0.0f);
    };
    auto init_shader = renderer().device().compile(init_kernel);

    Kernel1D clear_grid_kernel = [&](BufferInt grid_head) noexcept
    {
        grid_head.write(dispatch_x(), -1);
    };
    auto clear_grid_shader = renderer().device().compile(clear_grid_kernel);

    Kernel1D clear_count_kernel = [&](BufferUInt count) noexcept
    {
        count.write(0u, 0u);
    };
    auto clear_count_shader = renderer().device().compile(clear_count_kernel);

    Kernel1D clear_iter_kernel = [&](BufferUInt M, BufferFloat phi_r, BufferFloat phi_g, BufferFloat phi_b) noexcept
    {
        auto i = dispatch_x();
        M.write(i, 0u);
        phi_r.write(i, 0.0f);
        phi_g.write(i, 0.0f);
        phi_b.write(i, 0.0f);
    };
    auto clear_iter_shader = renderer().device().compile(clear_iter_kernel);

    // Hash a grid cell the same way on the deposit and the gather side.
    auto cell_of = [grid_cell_size](Expr<float3> p) noexcept
    {
        auto cell_f = floor(p * (1.0f / grid_cell_size));
        return make_int3(cast<int>(cell_f.x), cast<int>(cell_f.y), cast<int>(cell_f.z));
    };
    auto bucket_of = [hash_size](Expr<int3> cell) noexcept
    {
        auto hash_val = xxhash32(make_uint4(
            cast<uint>(cell.x), cast<uint>(cell.y), cast<uint>(cell.z), 0u));
        return hash_val % hash_size;
    };

    // ============================================================
    // KERNEL: Photon trace (importance transport from the lights)
    // ============================================================
    Kernel1D photon_kernel = [&](UInt iteration,
                                 BufferFloat4 p_pos, BufferFloat4 p_wi,
                                 BufferFloat4 p_beta_s,
                                 BufferInt p_next,
                                 BufferInt grid_head_buf, BufferUInt p_count_buf) noexcept
    {
        auto photon_idx = dispatch_x();

        auto rng    = def(xxhash32(make_uint4(photon_idx, iteration, 0x50504D4Du, 0u)));
        auto next_f = [&]() noexcept -> Float
        {
            rng = xxhash32(make_uint4(rng, photon_idx, iteration, 1u));
            return cast<float>(rng) * (1.0f / 4294967296.0f);
        };
        auto next_f2 = [&]() noexcept -> Float2
        {
            return make_float2(next_f(), next_f());
        };

        // The whole pass shares one set of wavelengths, as in pbrt's `passLambda`.
        auto spectrum_inst = renderer().spectrum();
        auto u_wl          = cast<float>(xxhash32(make_uint4(iteration, 0x7777u, 0u, 0u))) * (1.0f / 4294967296.0f);
        auto swl           = spectrum_inst->sample(spectrum_inst->base()->is_fixed() ? 0.0f : u_wl);

        // Uniform light selection; with a single emitter this matches pbrt's
        // power sampler exactly.
        auto n_lights_f = static_cast<float>(n_lights);
        auto u_sel      = next_f();
        auto sel_idx    = cast<uint>(clamp(u_sel * n_lights_f, 0.0f, n_lights_f - 1.0f));
        auto sel_pdf    = 1.0f / n_lights_f;

        auto handle    = renderer().buffer<Light::Handle>(light_handle_buffer_id).read(sel_idx);
        auto inst_id   = handle.instance_id;
        auto light_tag = handle.light_tag;

        auto u_pos = next_f2();
        auto u_dir = next_f2();

        auto le_sample = Light::Closure::EmissionSample::zero(swl.dimension());
        renderer().lights().dispatch(light_tag, [&](auto light) noexcept
        {
            auto closure = light->closure(swl, 0.0f);
            le_sample    = closure->sample_le(inst_id, u_pos, u_dir);
        });

        // beta = |cos| * Le / (p_light * pdf_pos * pdf_dir), as in pbrt.
        auto total_pdf = sel_pdf * le_sample.pdf;
        SampledSpectrum beta{swl.dimension()};
        beta = ite(total_pdf > 0.0f, le_sample.Le * le_sample.cos_theta / total_pdf, 0.0f);

        auto ray        = le_sample.ray;
        auto path_depth = def(0u);

        $loop
        {
            $if(path_depth >= depth_limit) { $break; };
            $if(beta.max() <= 0.0f) { $break; };

            auto wo = -ray->direction();
            auto it = renderer().geometry()->intersect(ray);

            $if(!it->is_surface_interaction()) { $break; };

            // Null surfaces are pass-through boundaries and must not consume depth.
            $if(!it->shape.has_surface())
            {
                ray = it->spawn_ray(ray->direction());
                $continue;
            };

            // Deposit at every hit past the first one. The first hit carries only
            // direct illumination, which the camera pass already accounts for.
            $if(path_depth > 0u)
            {
                auto slot = p_count_buf->atomic(0u).fetch_add(1u);
                $if(slot < capacity)
                {
                    p_pos.write(slot, make_float4(it->p_g, 0.0f));
                    // The w channel records whether this photon collapsed onto its
                    // hero wavelength, mirroring pbrt's `secondaryLambdaTerminated`.
                    p_wi.write(slot, make_float4(wo, ite(swl.secondary_terminated(), 1.0f, 0.0f)));
                    auto beta_packed = def(make_float4(0.0f));
                    for (auto c = 0u; c < spectrum_dimension; c++)
                    {
                        beta_packed[c] = beta[c];
                    }
                    p_beta_s.write(slot, beta_packed);

                    auto bucket = bucket_of(cell_of(it->p_g));
                    auto old    = grid_head_buf->atomic(bucket).exchange(cast<int>(slot));
                    p_next.write(slot, old);
                };
            };

            path_depth += 1u;

            auto u_lobe = next_f();
            auto u_bsdf = next_f2();
            auto u_rr   = next_f();

            $outline
            {
                PolymorphicCall<Surface::Closure> scatter_call;
                renderer().surfaces().dispatch(it->shape.surface_tag(), [&](auto surface) noexcept
                {
                    surface->closure(scatter_call, *it, wo, swl, 0.0f, 1.0f);
                });
                scatter_call.execute([&](const Surface::Closure* closure) noexcept
                {
                    if (auto dispersive = closure->is_dispersive())
                    {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf, TransportMode::IMPORTANCE);
                    auto w              = ite(surface_sample.eval.pdf > 0.0f, 1.0f / surface_sample.eval.pdf, 0.0f);
                    SampledSpectrum beta_new{swl.dimension()};
                    beta_new = beta * w * surface_sample.eval.f;

                    // Russian roulette on the *ratio* of throughputs, matching
                    // pbrt. Testing the absolute value instead never fires for
                    // bright emitters and lets useless paths run to max depth.
                    auto beta_max     = beta.max();
                    auto beta_new_max = beta_new.max();
                    auto beta_ratio   = ite(beta_max > 0.0f, beta_new_max / beta_max, 0.0f);
                    auto q            = max(0.0f, 1.0f - beta_ratio);
                    $if(u_rr < q)
                    {
                        beta = 0.0f;
                    }
                    $else
                    {
                        beta = beta_new / (1.0f - q);
                        ray  = it->spawn_ray(surface_sample.wi);
                    };
                });
            };

            beta = zero_if_any_nan(beta);
        };
    };

    LUISA_INFO("SPPM: Compiling photon trace kernel...");
    Clock clock_compile;
    auto photon_shader = renderer().device().compile(photon_kernel);
    LUISA_INFO("SPPM: Photon trace kernel compiled in {} ms.", clock_compile.toc());

    // ============================================================
    // KERNEL: Camera trace, direct lighting and photon gather
    // ============================================================
    Kernel2D camera_kernel = [&](UInt iteration,
                                 BufferFloat4 p_pos, BufferFloat4 p_wi,
                                 BufferFloat4 p_beta_s,
                                 BufferInt p_next,
                                 BufferInt grid_head_buf,
                                 BufferFloat radius_buf,
                                 BufferUInt M_buf, BufferFloat phi_r_buf, BufferFloat phi_g_buf, BufferFloat phi_b_buf,
                                 BufferFloat direct_r_buf, BufferFloat direct_g_buf, BufferFloat direct_b_buf) noexcept
    {
        set_block_size(16u, 16u, 1u);
        auto pixel_id  = dispatch_id().xy();
        auto pixel_idx = pixel_id.y * resolution.x + pixel_id.x;

        auto rng    = def(xxhash32(make_uint4(pixel_idx, iteration, 0xCAu, 0u)));
        auto next_f = [&]() noexcept -> Float
        {
            rng = xxhash32(make_uint4(rng, pixel_idx, iteration, 2u));
            return cast<float>(rng) * (1.0f / 4294967296.0f);
        };
        auto next_f2 = [&]() noexcept -> Float2
        {
            return make_float2(next_f(), next_f());
        };

        auto spectrum_inst = renderer().spectrum();
        auto u_wl          = cast<float>(xxhash32(make_uint4(iteration, 0x7777u, 0u, 0u))) * (1.0f / 4294967296.0f);
        auto swl           = spectrum_inst->sample(spectrum_inst->base()->is_fixed() ? 0.0f : u_wl);

        auto u_filter                        = next_f2();
        auto u_lens                          = camera->base()->requires_lens_sampling() ? next_f2() : make_float2(0.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, 0.0f, u_filter, u_lens);

        SampledSpectrum camera_beta{swl.dimension(), camera_weight};
        auto radius       = radius_buf.read(pixel_idx);
        auto radius2      = radius * radius;
        auto ray          = camera_ray;
        auto direct       = def(make_float3(0.0f));
        auto path_depth   = def(0u);
        auto gathered     = def(false);
        auto phi_local    = def(make_float3(0.0f));
        auto M_local      = def(0u);
        auto pdf_bsdf     = def(1e16f);
        auto delta_bounce = def(true);
        auto eta_scale    = def(1.0f);

        $loop
        {
            auto wo = -ray->direction();
            auto it = renderer().geometry()->intersect(ray);

            $if(!it->is_surface_interaction())
            {
                if (renderer().environment() != nullptr)
                {
                    auto eval   = light_sampler()->evaluate_miss(ray->direction(), swl, 0.0f);
                    auto weight = ite((path_depth == 0u) | delta_bounce, 1.0f, power_heuristic(pdf_bsdf, eval.pdf));
                    direct += spectrum_inst->srgb(swl, camera_beta * eval.L * weight);
                }
                $break;
            };

            // Emitted radiance seen along the camera path, MIS-weighted against
            // the next-event estimate performed at the previous vertex.
            $if(!renderer().lights().empty())
            {
                $if(it->shape.has_light())
                {
                    auto it_from = Interaction::from_point(ray->origin());
                    auto eval    = light_sampler()->evaluate_hit(*it, it_from, swl, 0.0f);
                    auto weight  = ite((path_depth == 0u) | delta_bounce, 1.0f, power_heuristic(pdf_bsdf, eval.pdf));
                    direct += spectrum_inst->srgb(swl, camera_beta * eval.L * weight);
                };
            };

            $if(!it->shape.has_surface())
            {
                ray = it->spawn_ray(ray->direction());
                $continue;
            };

            $if(path_depth >= depth_limit) { $break; };
            path_depth += 1u;

            auto u_light_sel = next_f();
            auto u_light_srf = next_f2();
            auto u_lobe      = next_f();
            auto u_bsdf      = next_f2();
            auto u_rr        = next_f();

            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            $outline
            {
                light_sample = light_sampler()->sample(*it, u_light_sel, u_light_srf, swl, 0.0f);
            };
            auto occluded = def(false);
            if (renderer().has_lighting())
            {
                occluded = renderer().geometry()->intersect_any(light_sample.shadow_ray);
            }

            $outline
            {
                PolymorphicCall<Surface::Closure> cam_call;
                renderer().surfaces().dispatch(it->shape.surface_tag(), [&](auto surface) noexcept
                {
                    surface->closure(cam_call, *it, wo, swl, 0.0f, 1.0f);
                });
                cam_call.execute([&](const Surface::Closure* closure) noexcept
                {
                    if (auto dispersive = closure->is_dispersive())
                    {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }

                    // Next-event estimation with MIS, matching SPPMIntegrator::SampleLd.
                    $if(light_sample.eval.pdf > 0.0f & !occluded)
                    {
                        auto wi_l  = light_sample.shadow_ray->direction();
                        auto eval  = closure->evaluate(wo, wi_l);
                        auto mis   = ite(light_sample.delta, 1.0f, power_heuristic(light_sample.eval.pdf, eval.pdf));
                        auto L_dir = spectrum_inst->srgb(swl, mis / light_sample.eval.pdf * camera_beta * eval.f * light_sample.eval.L);
                        direct += L_dir;
                    };

                    // A visible point is created at the first diffuse vertex, or
                    // at a glossy vertex once the depth budget is exhausted.
                    auto lobes      = closure->lobe_flags();
                    auto is_diffuse = (lobes & Surface::lobe_diffuse) != 0u;
                    auto is_glossy  = (lobes & Surface::lobe_glossy) != 0u;
                    auto make_vp    = is_diffuse | (is_glossy & (path_depth == depth_limit));

                    $if(make_vp)
                    {
                        gathered      = true;
                        auto vp_term  = swl.secondary_terminated();
                        auto cell     = cell_of(it->p_g);

                        $for(dx, -1, 2)
                        {
                            $for(dy, -1, 2)
                            {
                                $for(dz, -1, 2)
                                {
                                    auto bucket = bucket_of(cell + make_int3(dx, dy, dz));
                                    auto idx    = grid_head_buf.read(bucket);
                                    $while(idx >= 0)
                                    {
                                        auto uidx     = cast<uint>(idx);
                                        auto p_photon = p_pos.read(uidx).xyz();
                                        auto dist2    = length_squared(p_photon - it->p_g);
                                        $if(dist2 < radius2)
                                        {
                                            auto wi_data   = p_wi.read(uidx);
                                            auto photon_wi = wi_data.xyz();
                                            // pbrt gathers with `bsdf.f`, which excludes the
                                            // cosine factor that Evaluation::f folds in.
                                            auto eval_d = closure->evaluate(wo, photon_wi);
                                            auto cos_n  = abs(dot(photon_wi, it->shading.n()));
                                            auto bsdf_f = eval_d.f / max(cos_n, 1e-6f);

                                            auto photon_beta_data = p_beta_s.read(uidx);
                                            SampledSpectrum photon_beta{swl.dimension()};
                                            for (auto c = 0u; c < spectrum_dimension; c++)
                                            {
                                                photon_beta[c] = photon_beta_data[c];
                                            }

                                            SampledSpectrum product{swl.dimension()};
                                            product = camera_beta * photon_beta * bsdf_f;

                                            // If the photon collapsed onto its hero wavelength
                                            // but the visible point did not, the secondary
                                            // components are not comparable. Keep the hero term
                                            // only, rescaled to stay unbiased, mirroring how
                                            // pbrt terminates the photon's own wavelengths.
                                            if (!spectrum_is_fixed && spectrum_dimension > 1u)
                                            {
                                                auto mismatch   = (wi_data.w > 0.5f) & !vp_term;
                                                auto hero_scale = static_cast<float>(spectrum_dimension);
                                                product[0u]     = ite(mismatch, product[0u] * hero_scale, product[0u]);
                                                for (auto c = 1u; c < spectrum_dimension; c++)
                                                {
                                                    product[c] = ite(mismatch, 0.0f, product[c]);
                                                }
                                            }

                                            phi_local += spectrum_inst->srgb(swl, product);
                                            M_local += 1u;
                                        };
                                        idx = p_next.read(uidx);
                                    };
                                };
                            };
                        };
                    };

                    // Only non-visible-point vertices continue the camera path.
                    $if(!gathered)
                    {
                        auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                        ray                 = it->spawn_ray(surface_sample.wi);
                        pdf_bsdf            = surface_sample.pdf_mis;
                        delta_bounce        = surface_sample.delta;
                        auto w              = ite(surface_sample.eval.pdf > 0.0f, 1.0f / surface_sample.eval.pdf, 0.0f);
                        camera_beta *= w * surface_sample.eval.f;
                        auto transmission = (surface_sample.event & Surface::event_transmit) != 0u;
                        eta_scale *= ite(transmission, sqr(surface_sample.eta), 1.0f);
                    };
                });
            };

            $if(gathered) { $break; };

            camera_beta = zero_if_any_nan(camera_beta);
            $if(camera_beta.max() <= 0.0f) { $break; };

            auto rr_beta_max = camera_beta.max() * eta_scale;
            $if(rr_beta_max < 1.0f)
            {
                auto q = max(0.05f, 1.0f - rr_beta_max);
                $if(u_rr < q) { $break; };
                camera_beta /= 1.0f - q;
            };
        };

        auto prev_dr = direct_r_buf.read(pixel_idx);
        auto prev_dg = direct_g_buf.read(pixel_idx);
        auto prev_db = direct_b_buf.read(pixel_idx);
        direct_r_buf.write(pixel_idx, prev_dr + direct.x);
        direct_g_buf.write(pixel_idx, prev_dg + direct.y);
        direct_b_buf.write(pixel_idx, prev_db + direct.z);

        M_buf.write(pixel_idx, M_local);
        phi_r_buf.write(pixel_idx, phi_local.x);
        phi_g_buf.write(pixel_idx, phi_local.y);
        phi_b_buf.write(pixel_idx, phi_local.z);
    };

    LUISA_INFO("SPPM: Compiling camera trace kernel...");
    clock_compile.tic();
    auto camera_shader = renderer().device().compile(camera_kernel);
    LUISA_INFO("SPPM: Camera trace kernel compiled in {} ms.", clock_compile.toc());

    // ============================================================
    // KERNEL: Progressive radius / tau update
    // ============================================================
    Kernel1D update_kernel = [&](BufferFloat radius_buf,
                                 BufferFloat N_buf, BufferFloat tau_r_buf, BufferFloat tau_g_buf, BufferFloat tau_b_buf,
                                 BufferUInt M_buf_in, BufferFloat phi_r_in, BufferFloat phi_g_in, BufferFloat phi_b_in) noexcept
    {
        auto i     = dispatch_x();
        auto M_val = cast<float>(M_buf_in.read(i));

        // pbrt only advances a pixel that actually received photons.
        $if(M_val > 0.0f)
        {
            auto N_old = N_buf.read(i);
            auto r_old = radius_buf.read(i);

            auto N_new = N_old + gamma_sppm * M_val;
            auto r_new = r_old * sqrt(N_new / (N_old + M_val));
            auto scale = (r_new * r_new) / (r_old * r_old);

            tau_r_buf.write(i, (tau_r_buf.read(i) + phi_r_in.read(i)) * scale);
            tau_g_buf.write(i, (tau_g_buf.read(i) + phi_g_in.read(i)) * scale);
            tau_b_buf.write(i, (tau_b_buf.read(i) + phi_b_in.read(i)) * scale);
            N_buf.write(i, N_new);
            radius_buf.write(i, r_new);
        };
    };

    LUISA_INFO("SPPM: Compiling update kernel...");
    clock_compile.tic();
    auto update_shader = renderer().device().compile(update_kernel);
    LUISA_INFO("SPPM: Update kernel compiled in {} ms.", clock_compile.toc());

    renderer().reset_diagnostics(command_buffer);
    camera->film()->prepare(command_buffer, enable_display);
    command_buffer << synchronize();

    // ============================================================
    // KERNEL: Resolve final image
    // ============================================================
    Kernel2D resolve_kernel = [&](UInt iter_count,
                                  BufferFloat radius_buf,
                                  BufferFloat tau_r_buf_r, BufferFloat tau_g_buf_r, BufferFloat tau_b_buf_r,
                                  BufferFloat direct_r_r, BufferFloat direct_g_r, BufferFloat direct_b_r) noexcept
    {
        set_block_size(16u, 16u, 1u);
        auto pixel_id  = dispatch_id().xy();
        auto pixel_idx = pixel_id.y * resolution.x + pixel_id.x;

        auto iters_f   = cast<float>(iter_count);
        auto photons_f = static_cast<float>(photons_per_iter);

        auto tau = make_float3(tau_r_buf_r.read(pixel_idx),
                               tau_g_buf_r.read(pixel_idx),
                               tau_b_buf_r.read(pixel_idx));
        auto Ld  = make_float3(direct_r_r.read(pixel_idx),
                               direct_g_r.read(pixel_idx),
                               direct_b_r.read(pixel_idx));

        auto radius = radius_buf.read(pixel_idx);
        // L = Ld / iterations + tau / (n_photons * pi * r^2), as in pbrt.
        auto indirect = tau / (iters_f * photons_f * pi_f * radius * radius);
        camera->film()->set_pixel_single_writer(pixel_id, Ld / iters_f + indirect);
    };

    LUISA_INFO("SPPM: Compiling resolve kernel...");
    clock_compile.tic();
    auto resolve_shader = renderer().device().compile(resolve_kernel);
    LUISA_INFO("SPPM: Resolve kernel compiled in {} ms.", clock_compile.toc());

    // ============================================================
    // RENDER LOOP
    // ============================================================
    command_buffer
        << init_shader(*buf_radius, *buf_N, *buf_tau_r, *buf_tau_g, *buf_tau_b,
                       *buf_direct_r, *buf_direct_g, *buf_direct_b)
               .dispatch(pixel_count)
        << synchronize();

    LUISA_INFO("SPPM: Rendering started.");
    Clock clock_render;
    ProgressBar progress_bar;
    progress_bar.update(0.0);

    uint64_t photon_overflow_iterations = 0u;
    uint64_t photon_overflow_max        = 0u;

    for (auto iter = 0u; iter < iterations; iter++)
    {
        command_buffer
            << clear_grid_shader(*buf_grid_head).dispatch(hash_size)
            << clear_count_shader(*buf_photon_count).dispatch(1u)
            << clear_iter_shader(*buf_M, *buf_phi_r, *buf_phi_g, *buf_phi_b).dispatch(pixel_count);

        command_buffer
            << photon_shader(iter,
                             *buf_photon_pos, *buf_photon_wi,
                             *buf_photon_beta_s,
                             *buf_photon_next,
                             *buf_grid_head, *buf_photon_count)
                   .dispatch(photons_per_iter);

        command_buffer
            << camera_shader(iter,
                             *buf_photon_pos, *buf_photon_wi,
                             *buf_photon_beta_s,
                             *buf_photon_next,
                             *buf_grid_head,
                             *buf_radius,
                             *buf_M, *buf_phi_r, *buf_phi_g, *buf_phi_b,
                             *buf_direct_r, *buf_direct_g, *buf_direct_b)
                   .dispatch(resolution);

        command_buffer
            << update_shader(*buf_radius,
                             *buf_N, *buf_tau_r, *buf_tau_g, *buf_tau_b,
                             *buf_M, *buf_phi_r, *buf_phi_g, *buf_phi_b)
                   .dispatch(pixel_count);

        command_buffer
            << resolve_shader(iter + 1u,
                              *buf_radius,
                              *buf_tau_r, *buf_tau_g, *buf_tau_b,
                              *buf_direct_r, *buf_direct_g, *buf_direct_b)
                   .dispatch(resolution);

        // Photon-map overflow silently drops energy, so surface it to the user.
        {
            uint requested{};
            command_buffer << buf_photon_count->copy_to(&requested) << synchronize();
            if (requested > capacity)
            {
                photon_overflow_iterations++;
                photon_overflow_max = std::max<uint64_t>(photon_overflow_max, requested);
            }
        }

        if (camera->film()->show(command_buffer))
        { /* frame displayed */
        }
        if (camera->film()->should_close()) [[unlikely]]
        {
            command_buffer << synchronize();
            progress_bar.cancel();
            camera->film()->release();
            return;
        }

        auto progress_stride = std::max(1u, iterations / 100u);
        if ((iter + 1u) % progress_stride == 0u || iter + 1u == iterations)
        {
            command_buffer << [&progress_bar, iter, iterations]
            {
                progress_bar.update(static_cast<double>(iter + 1u) / static_cast<double>(iterations));
            };
        }

        command_buffer << commit();
    }

    command_buffer << synchronize();
    progress_bar.done();
    LUISA_INFO("SPPM: Rendering finished in {} ms.", clock_render.toc());

    if (photon_overflow_iterations != 0u)
    {
        LUISA_WARNING(
            "SPPM photon map overflowed in {} of {} iteration(s) (peak {} records for a capacity of {}). "
            "Caustics will be darker than expected; lower 'photonsperiteration' or 'maxdepth'.",
            photon_overflow_iterations, iterations, photon_overflow_max, capacity);
    }

    // ============================================================
    // DOWNLOAD AND SAVE
    // ============================================================
    bool output_saved = false;
    RenderDiagnostics diagnostics{};
    uint64_t host_nan_count{};
    uint64_t host_inf_count{};
    auto output_path = camera->film()->base()->filename();
    {
        luisa::vector<float4> pixels(pixel_count);
        camera->film()->download(command_buffer, pixels.data());
        std::array<uint, 4u> diagnostic_values{};
        renderer().download_diagnostics(command_buffer, diagnostic_values);
        command_buffer << synchronize();

        diagnostics = RenderDiagnostics{
            .path_nan = diagnostic_values[0u],
            .path_inf = diagnostic_values[1u],
            .film_nan = diagnostic_values[2u],
            .film_inf = diagnostic_values[3u],
        };
        for (auto& pixel : pixels)
        {
            auto values = reinterpret_cast<float*>(&pixel);
            for (auto channel = 0u; channel < 3u; channel++)
            {
                if (std::isnan(values[channel]))
                {
                    host_nan_count++;
                    values[channel] = 0.0f;
                }
                else if (std::isinf(values[channel]))
                {
                    host_inf_count++;
                    values[channel] = 0.0f;
                }
            }
        }
        Clock clock_save;
        output_saved = save_image(output_path, reinterpret_cast<const float*>(pixels.data()), resolution);
        if (output_saved)
        {
            LUISA_INFO("Saved render output '{}' in {} ms.", output_path.string(), clock_save.toc());
        }
    }
    camera->film()->release();

    auto invalid_count = diagnostics.total() + host_nan_count + host_inf_count;
    if (invalid_count != 0u)
    {
        LUISA_WARNING(
            "Non-finite render values: path NaN={}, path Inf={}, film NaN={}, film Inf={}, output NaN={}, output Inf={}.",
            diagnostics.path_nan,
            diagnostics.path_inf,
            diagnostics.film_nan,
            diagnostics.film_inf,
            host_nan_count,
            host_inf_count);
    }
    if (!output_saved)
    {
        throw std::runtime_error{luisa::format("Failed to save render output '{}'.", output_path.string()).c_str()};
    }
    if (invalid_count != 0u)
    {
        throw std::runtime_error{luisa::format(
                                     "Render failed with {} non-finite value(s). Debug output was saved to '{}'.",
                                     invalid_count,
                                     output_path.string())
                                     .c_str()};
    }
}

} // namespace Yutrel
