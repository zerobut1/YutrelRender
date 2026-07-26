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

    auto iterations             = sampler()->base()->spp();
    auto photons_per_iter       = photons_per_iteration();
    auto depth_limit            = max_depth();
    auto r_initial              = initial_radius();
    constexpr float gamma_sppm  = 2.0f / 3.0f;

    // Validate that we have area lights
    if (_n_lights == 0u)
    {
        LUISA_ERROR_WITH_LOCATION("SPPM requires at least one area light source in the scene.");
    }

    LUISA_INFO(
        "SPPM: resolution={}x{}, iterations={}, photons_per_iter={}, max_depth={}, initial_radius={}.",
        resolution.x, resolution.y, iterations, photons_per_iter, depth_limit, r_initial);

    // Capacity for photon buffer
    auto capacity_64 = static_cast<uint64_t>(photons_per_iter) * static_cast<uint64_t>(depth_limit);
    if (capacity_64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
    {
        LUISA_ERROR_WITH_LOCATION(
            "SPPM photon buffer capacity overflow: {} * {} = {} exceeds uint32 max.",
            photons_per_iter, depth_limit, capacity_64);
    }
    auto capacity  = static_cast<uint>(capacity_64);
    auto hash_size = static_cast<uint>(std::bit_ceil(std::max(2u * photons_per_iter, 1u)));

    LUISA_INFO("SPPM: photon buffer capacity={}, hash_size={}.", capacity, hash_size);

    // Allocate GPU buffers - per-pixel state
    auto buf_N          = renderer().create<Buffer<float>>(pixel_count);
    auto buf_tau_r      = renderer().create<Buffer<float>>(pixel_count);
    auto buf_tau_g      = renderer().create<Buffer<float>>(pixel_count);
    auto buf_tau_b      = renderer().create<Buffer<float>>(pixel_count);
    auto buf_direct_r   = renderer().create<Buffer<float>>(pixel_count);
    auto buf_direct_g   = renderer().create<Buffer<float>>(pixel_count);
    auto buf_direct_b   = renderer().create<Buffer<float>>(pixel_count);
    auto buf_M          = renderer().create<Buffer<uint>>(pixel_count);
    auto buf_phi_r      = renderer().create<Buffer<float>>(pixel_count);
    auto buf_phi_g      = renderer().create<Buffer<float>>(pixel_count);
    auto buf_phi_b      = renderer().create<Buffer<float>>(pixel_count);

    // Photon map buffers
    auto buf_photon_pos      = renderer().create<Buffer<float4>>(capacity);
    auto buf_photon_wi       = renderer().create<Buffer<float4>>(capacity);
    auto buf_photon_beta_s   = renderer().create<Buffer<float4>>(capacity); // spectral beta (4 wavelengths)
    auto buf_photon_next     = renderer().create<Buffer<int>>(capacity);
    auto buf_grid_head       = renderer().create<Buffer<int>>(hash_size);
    auto buf_photon_count    = renderer().create<Buffer<uint>>(1u);

    // Capture member variables for kernels
    auto light_handle_buffer_id = _light_handle_buffer_id;
    auto n_lights               = _n_lights;

    // ============================================================
    // KERNEL: Initialize per-pixel state
    // ============================================================
    Kernel1D init_kernel = [&](BufferFloat N, BufferFloat tau_r, BufferFloat tau_g, BufferFloat tau_b,
                               BufferFloat direct_r, BufferFloat direct_g, BufferFloat direct_b) noexcept
    {
        auto i = dispatch_x();
        N.write(i, 0.0f);
        tau_r.write(i, 0.0f);
        tau_g.write(i, 0.0f);
        tau_b.write(i, 0.0f);
        direct_r.write(i, 0.0f);
        direct_g.write(i, 0.0f);
        direct_b.write(i, 0.0f);
    };
    auto init_shader = renderer().device().compile(init_kernel);

    // ============================================================
    // KERNEL: Clear grid
    // ============================================================
    Kernel1D clear_grid_kernel = [&](BufferInt grid_head) noexcept
    {
        grid_head.write(dispatch_x(), -1);
    };
    auto clear_grid_shader = renderer().device().compile(clear_grid_kernel);

    // ============================================================
    // KERNEL: Clear photon count
    // ============================================================
    Kernel1D clear_count_kernel = [&](BufferUInt count) noexcept
    {
        count.write(0u, 0u);
    };
    auto clear_count_shader = renderer().device().compile(clear_count_kernel);

    // ============================================================
    // KERNEL: Clear per-iteration accumulators
    // ============================================================
    Kernel1D clear_iter_kernel = [&](BufferUInt M, BufferFloat phi_r, BufferFloat phi_g, BufferFloat phi_b) noexcept
    {
        auto i = dispatch_x();
        M.write(i, 0u);
        phi_r.write(i, 0.0f);
        phi_g.write(i, 0.0f);
        phi_b.write(i, 0.0f);
    };
    auto clear_iter_shader = renderer().device().compile(clear_iter_kernel);

    // ============================================================
    // KERNEL: Photon trace
    // ============================================================
    Kernel1D photon_kernel = [&](UInt iteration, Float shared_radius,
                                 BufferFloat4 p_pos, BufferFloat4 p_wi,
                                 BufferFloat4 p_beta_s,
                                 BufferInt p_next,
                                 BufferInt grid_head_buf, BufferUInt p_count_buf) noexcept
    {
        auto photon_idx = dispatch_x();

        // Hash-based RNG
        auto rng = def(xxhash32(make_uint4(photon_idx, iteration, 0x50504D4Du, 0u)));
        auto next_f = [&]() noexcept -> Float
        {
            rng = xxhash32(make_uint4(rng, photon_idx, iteration, 1u));
            return cast<float>(rng) * (1.0f / 4294967296.0f);
        };
        auto next_f2 = [&]() noexcept -> Float2
        {
            return make_float2(next_f(), next_f());
        };

        // Sample wavelengths - shared across iteration
        auto spectrum_inst = renderer().spectrum();
        auto u_wl          = cast<float>(xxhash32(make_uint4(iteration, 0x7777u, 0u, 0u))) * (1.0f / 4294967296.0f);
        auto swl           = spectrum_inst->sample(spectrum_inst->base()->is_fixed() ? 0.0f : u_wl);

        // Select light uniformly
        auto n_lights_f = static_cast<float>(n_lights);
        auto u_sel      = next_f();
        auto sel_idx    = cast<uint>(clamp(u_sel * n_lights_f, 0.0f, n_lights_f - 1.0f));
        auto sel_pdf    = 1.0f / n_lights_f;

        // Read light handle from buffer
        auto handle    = renderer().buffer<Light::Handle>(light_handle_buffer_id).read(sel_idx);
        auto inst_id   = handle.instance_id;
        auto light_tag = handle.light_tag;

        // Sample emission
        auto u_pos = next_f2();
        auto u_dir = next_f2();

        auto le_sample = Light::Closure::EmissionSample::zero(swl.dimension());
        renderer().lights().dispatch(light_tag, [&](auto light) noexcept
        {
            auto closure = light->closure(swl, 0.0f);
            le_sample    = closure->sample_le(inst_id, u_pos, u_dir);
        });

        // Compute initial beta = Le * cos_theta / (select_pdf * pdf)
        auto total_pdf = sel_pdf * le_sample.pdf;
        SampledSpectrum beta{swl.dimension()};
        beta = ite(total_pdf > 0.0f, le_sample.Le * le_sample.cos_theta / total_pdf, 0.0f);

        auto ray          = le_sample.ray;
        auto first_bounce = def(true);
        auto path_depth   = def(0u);

        $loop
        {
            $if(path_depth >= depth_limit) { $break; };

            auto wo = -ray->direction();
            auto it = renderer().geometry()->intersect(ray);

            $if(!it->is_surface_interaction()) { $break; };

            // Skip null surfaces
            $if(!it->shape.has_surface())
            {
                ray = it->spawn_ray(ray->direction());
                $continue;
            };

            path_depth += 1u;

            // Skip first bounce to avoid double-counting direct light
            $if(first_bounce)
            {
                first_bounce = false;
            }
            $else
            {
                // Store photon at diffuse surfaces
                $outline
                {
                    PolymorphicCall<Surface::Closure> check_call;
                    renderer().surfaces().dispatch(it->shape.surface_tag(), [&](auto surface) noexcept
                    {
                        surface->closure(check_call, *it, wo, swl, 0.0f, 1.0f);
                    });
                    check_call.execute([&](const Surface::Closure* closure) noexcept
                    {
                        auto lobes = closure->lobe_flags();
                        $if((lobes & Surface::lobe_diffuse) != 0u)
                        {
                            auto slot = p_count_buf->atomic(0u).fetch_add(1u);
                            $if(slot < capacity)
                            {
                                p_pos.write(slot, make_float4(it->p_g, 0.0f));
                                p_wi.write(slot, make_float4(wo, 0.0f));
                                // Store spectral beta (4 wavelength components)
                                p_beta_s.write(slot, make_float4(beta[0u], beta[1u], beta[2u], beta[3u]));

                                // Insert into hash grid
                                auto cell_f = floor(it->p_g / shared_radius);
                                auto cell = make_int3(cast<int>(cell_f.x), cast<int>(cell_f.y), cast<int>(cell_f.z));
                                auto hash_val = xxhash32(make_uint4(
                                    cast<uint>(cell.x), cast<uint>(cell.y), cast<uint>(cell.z), 0u));
                                auto bucket = hash_val % hash_size;
                                auto old    = grid_head_buf->atomic(bucket).exchange(cast<int>(slot));
                                p_next.write(slot, old);
                            };
                        };
                    });
                };
            };

            // Sample BSDF to continue path (importance transport)
            auto u_lobe = next_f();
            auto u_bsdf = next_f2();

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
                    ray                 = it->spawn_ray(surface_sample.wi);
                    auto w              = ite(surface_sample.eval.pdf > 0.0f, 1.0f / surface_sample.eval.pdf, 0.0f);
                    beta *= w * surface_sample.eval.f;
                });
            };

            // Russian roulette
            auto beta_max = beta.max();
            $if(beta_max <= 0.0f) { $break; };
            $if(path_depth > 1u)
            {
                auto q    = max(0.0f, 1.0f - beta_max);
                auto u_rr = next_f();
                $if(u_rr < q) { $break; };
                beta /= 1.0f - q;
            };
        };
    };

    LUISA_INFO("SPPM: Compiling photon trace kernel...");
    Clock clock_compile;
    auto photon_shader = renderer().device().compile(photon_kernel);
    LUISA_INFO("SPPM: Photon trace kernel compiled in {} ms.", clock_compile.toc());

    // ============================================================
    // KERNEL: Camera trace + gather
    // ============================================================
    Kernel2D camera_kernel = [&](UInt iteration, Float shared_radius,
                                 BufferFloat4 p_pos, BufferFloat4 p_wi,
                                 BufferFloat4 p_beta_s,
                                 BufferInt p_next,
                                 BufferInt grid_head_buf,
                                 BufferUInt M_buf, BufferFloat phi_r_buf, BufferFloat phi_g_buf, BufferFloat phi_b_buf,
                                 BufferFloat direct_r_buf, BufferFloat direct_g_buf, BufferFloat direct_b_buf) noexcept
    {
        set_block_size(16u, 16u, 1u);
        auto pixel_id  = dispatch_id().xy();
        auto pixel_idx = pixel_id.y * resolution.x + pixel_id.x;

        // Hash-based RNG
        auto rng = def(xxhash32(make_uint4(pixel_idx, iteration, 0xCAu, 0u)));
        auto next_f = [&]() noexcept -> Float
        {
            rng = xxhash32(make_uint4(rng, pixel_idx, iteration, 2u));
            return cast<float>(rng) * (1.0f / 4294967296.0f);
        };
        auto next_f2 = [&]() noexcept -> Float2
        {
            return make_float2(next_f(), next_f());
        };

        // Sample wavelengths (same as photons for this iteration)
        auto spectrum_inst = renderer().spectrum();
        auto u_wl          = cast<float>(xxhash32(make_uint4(iteration, 0x7777u, 0u, 0u))) * (1.0f / 4294967296.0f);
        auto swl           = spectrum_inst->sample(spectrum_inst->base()->is_fixed() ? 0.0f : u_wl);

        // Generate camera ray
        auto u_filter = next_f2();
        auto u_lens   = camera->base()->requires_lens_sampling() ? next_f2() : make_float2(0.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, 0.0f, u_filter, u_lens);

        SampledSpectrum camera_beta{swl.dimension(), camera_weight};
        auto ray       = camera_ray;
        auto direct    = def(make_float3(0.0f));
        auto path_depth = def(0u);
        auto gathered  = def(false);
        auto phi_local = def(make_float3(0.0f));
        auto M_local   = def(0u);

        $loop
        {
            $if(path_depth >= depth_limit) { $break; };

            auto wo = -ray->direction();
            auto it = renderer().geometry()->intersect(ray);

            $if(!it->is_surface_interaction())
            {
                // Environment light (direct on first bounce)
                if (renderer().environment() != nullptr)
                {
                    $if(path_depth == 0u)
                    {
                        auto eval  = light_sampler()->evaluate_miss(ray->direction(), swl, 0.0f);
                        auto L_env = spectrum_inst->srgb(swl, camera_beta * eval.L);
                        direct += L_env;
                    };
                }
                $break;
            };

            // Direct emission on first bounce
            $if(!renderer().lights().empty())
            {
                $if(path_depth == 0u & it->shape.has_light())
                {
                    auto it_from = Interaction::from_point(ray->origin());
                    auto eval    = light_sampler()->evaluate_hit(*it, it_from, swl, 0.0f);
                    auto L_emit  = spectrum_inst->srgb(swl, camera_beta * eval.L);
                    direct += L_emit;
                };
            };

            // Skip null surfaces
            $if(!it->shape.has_surface())
            {
                ray = it->spawn_ray(ray->direction());
                $continue;
            };

            path_depth += 1u;

            // Surface interaction
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

                    auto lobes = closure->lobe_flags();
                    $if((lobes & Surface::lobe_diffuse) != 0u & !gathered)
                    {
                        // NEE direct lighting
                        auto u_light_sel  = next_f();
                        auto u_light_srf  = next_f2();
                        auto light_sample = light_sampler()->sample(*it, u_light_sel, u_light_srf, swl, 0.0f);
                        auto occluded     = def(false);
                        if (renderer().has_lighting())
                        {
                            occluded = renderer().geometry()->intersect_any(light_sample.shadow_ray);
                        }
                        $if(light_sample.eval.pdf > 0.0f & !occluded)
                        {
                            auto wi_l  = light_sample.shadow_ray->direction();
                            auto eval  = closure->evaluate(wo, wi_l);
                            auto L_dir = spectrum_inst->srgb(swl, camera_beta * eval.f * light_sample.eval.L / light_sample.eval.pdf);
                            direct += L_dir;
                        };

                        // Gather photons
                        auto r2 = shared_radius * shared_radius;
                        auto cell_f = floor(it->p_g / shared_radius);
                        auto cell = make_int3(cast<int>(cell_f.x), cast<int>(cell_f.y), cast<int>(cell_f.z));

                        $for(dx, -1, 2)
                        {
                            $for(dy, -1, 2)
                            {
                                $for(dz, -1, 2)
                                {
                                    auto nc   = cell + make_int3(dx, dy, dz);
                                    auto hash = xxhash32(make_uint4(
                                        cast<uint>(nc.x), cast<uint>(nc.y), cast<uint>(nc.z), 0u));
                                    auto bucket = hash % hash_size;
                                    auto idx    = grid_head_buf.read(cast<uint>(bucket));
                                    $while(idx >= 0)
                                    {
                                        auto uidx     = cast<uint>(idx);
                                        auto pos_data = p_pos.read(uidx);
                                        auto p_photon = pos_data.xyz();
                                        auto dist2    = length_squared(p_photon - it->p_g);
                                        $if(dist2 < r2)
                                        {
                                            auto photon_wi = p_wi.read(uidx).xyz();
                                            auto eval_d    = closure->evaluate(wo, photon_wi);
                                            // Use f_diffuse / cos to remove the implicit cosine
                                            auto cos_n  = abs(dot(photon_wi, it->shading.n()));
                                            auto bsdf_f = eval_d.f_diffuse / max(cos_n, 1e-6f);

                                            // Read photon's spectral beta
                                            auto photon_beta_data = p_beta_s.read(uidx);
                                            SampledSpectrum photon_beta{swl.dimension()};
                                            photon_beta[0u] = photon_beta_data.x;
                                            photon_beta[1u] = photon_beta_data.y;
                                            photon_beta[2u] = photon_beta_data.z;
                                            photon_beta[3u] = photon_beta_data.w;

                                            // Spectral product: camera_beta * photon_beta * bsdf_f
                                            // Convert to RGB ONCE (correct spectral rendering)
                                            auto contribution = spectrum_inst->srgb(swl, camera_beta * photon_beta * bsdf_f);
                                            phi_local += contribution;
                                            M_local += 1u;
                                        };
                                        idx = p_next.read(uidx);
                                    };
                                };
                            };
                        };

                        gathered = true;
                    };

                    // Continue along non-diffuse path
                    $if(!gathered)
                    {
                        auto surface_sample = closure->sample(wo, next_f(), next_f2());
                        ray                 = it->spawn_ray(surface_sample.wi);
                        auto w              = ite(surface_sample.eval.pdf > 0.0f, 1.0f / surface_sample.eval.pdf, 0.0f);
                        camera_beta *= w * surface_sample.eval.f;
                    };
                });
            };

            $if(gathered) { $break; };

            auto cam_beta_max = camera_beta.max();
            $if(cam_beta_max <= 0.0f) { $break; };
        };

        // Write per-pixel results
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
    // KERNEL: Progressive update
    // ============================================================
    Kernel1D update_kernel = [&](Float radius_in,
                                 BufferFloat N_buf, BufferFloat tau_r_buf, BufferFloat tau_g_buf, BufferFloat tau_b_buf,
                                 BufferUInt M_buf_in, BufferFloat phi_r_in, BufferFloat phi_g_in, BufferFloat phi_b_in) noexcept
    {
        auto i      = dispatch_x();
        auto N_old  = N_buf.read(i);
        auto M_val  = cast<float>(M_buf_in.read(i));
        auto phi_r  = phi_r_in.read(i);
        auto phi_g  = phi_g_in.read(i);
        auto phi_b  = phi_b_in.read(i);
        auto tau_r  = tau_r_buf.read(i);
        auto tau_g  = tau_g_buf.read(i);
        auto tau_b  = tau_b_buf.read(i);

        auto N_new = N_old + gamma_sppm * M_val;
        auto R_old = radius_in;
        auto ratio = ite(N_old + M_val > 0.0f, N_new / (N_old + M_val), 1.0f);
        auto scale = ratio; // R_new^2/R_old^2 = ratio

        tau_r_buf.write(i, (tau_r + phi_r) * scale);
        tau_g_buf.write(i, (tau_g + phi_g) * scale);
        tau_b_buf.write(i, (tau_b + phi_b) * scale);
        N_buf.write(i, N_new);
    };

    LUISA_INFO("SPPM: Compiling update kernel...");
    clock_compile.tic();
    auto update_shader = renderer().device().compile(update_kernel);
    LUISA_INFO("SPPM: Update kernel compiled in {} ms.", clock_compile.toc());

    // ============================================================
    // PREPARE FILM (must be done before compiling resolve kernel)
    // ============================================================
    renderer().reset_diagnostics(command_buffer);
    camera->film()->prepare(command_buffer, enable_display);
    command_buffer << synchronize();

    // ============================================================
    // KERNEL: Resolve final image
    // ============================================================
    Kernel2D resolve_kernel = [&](UInt iter_count, Float current_radius,
                                  BufferFloat tau_r_buf_r, BufferFloat tau_g_buf_r, BufferFloat tau_b_buf_r,
                                  BufferFloat direct_r_r, BufferFloat direct_g_r, BufferFloat direct_b_r) noexcept
    {
        set_block_size(16u, 16u, 1u);
        auto pixel_id  = dispatch_id().xy();
        auto pixel_idx = pixel_id.y * resolution.x + pixel_id.x;

        auto iters_f   = cast<float>(iter_count);
        auto photons_f = static_cast<float>(photons_per_iter);
        constexpr auto pi = 3.14159265358979323846f;

        auto tau_r = tau_r_buf_r.read(pixel_idx);
        auto tau_g = tau_g_buf_r.read(pixel_idx);
        auto tau_b = tau_b_buf_r.read(pixel_idx);
        auto dr    = direct_r_r.read(pixel_idx);
        auto dg    = direct_g_r.read(pixel_idx);
        auto db    = direct_b_r.read(pixel_idx);

        auto R2         = current_radius * current_radius;
        auto indirect   = make_float3(tau_r, tau_g, tau_b) / (iters_f * photons_f * pi * R2);
        auto direct_avg = make_float3(dr, dg, db) / iters_f;

        auto L = direct_avg + indirect;
        camera->film()->set_pixel_single_writer(pixel_id, L);
    };

    LUISA_INFO("SPPM: Compiling resolve kernel...");
    clock_compile.tic();
    auto resolve_shader = renderer().device().compile(resolve_kernel);
    LUISA_INFO("SPPM: Resolve kernel compiled in {} ms.", clock_compile.toc());

    // ============================================================
    // RENDER LOOP
    // ============================================================

    // Initialize per-pixel buffers
    command_buffer
        << init_shader(*buf_N, *buf_tau_r, *buf_tau_g, *buf_tau_b,
                       *buf_direct_r, *buf_direct_g, *buf_direct_b).dispatch(pixel_count)
        << synchronize();

    LUISA_INFO("SPPM: Rendering started.");
    Clock clock_render;
    ProgressBar progress_bar;
    progress_bar.update(0.0);

    float current_radius = r_initial;

    for (auto iter = 0u; iter < iterations; iter++)
    {
        // 1. Clear grid, photon count, and per-iteration accumulators
        command_buffer
            << clear_grid_shader(*buf_grid_head).dispatch(hash_size)
            << clear_count_shader(*buf_photon_count).dispatch(1u)
            << clear_iter_shader(*buf_M, *buf_phi_r, *buf_phi_g, *buf_phi_b).dispatch(pixel_count);

        // 2. Trace photons
        command_buffer
            << photon_shader(iter, current_radius,
                             *buf_photon_pos, *buf_photon_wi,
                             *buf_photon_beta_s,
                             *buf_photon_next,
                             *buf_grid_head, *buf_photon_count).dispatch(photons_per_iter);

        // 3. Camera trace and gather
        command_buffer
            << camera_shader(iter, current_radius,
                             *buf_photon_pos, *buf_photon_wi,
                             *buf_photon_beta_s,
                             *buf_photon_next,
                             *buf_grid_head,
                             *buf_M, *buf_phi_r, *buf_phi_g, *buf_phi_b,
                             *buf_direct_r, *buf_direct_g, *buf_direct_b).dispatch(resolution);

        // 4. Progressive update of tau and N
        command_buffer
            << update_shader(current_radius,
                             *buf_N, *buf_tau_r, *buf_tau_g, *buf_tau_b,
                             *buf_M, *buf_phi_r, *buf_phi_g, *buf_phi_b).dispatch(pixel_count);

        // 5. Resolve to film
        auto iter_count = iter + 1u;
        command_buffer
            << resolve_shader(iter_count, current_radius,
                              *buf_tau_r, *buf_tau_g, *buf_tau_b,
                              *buf_direct_r, *buf_direct_g, *buf_direct_b).dispatch(resolution);

        // Display
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

        // Update shared radius: R_k = R_0 * k^{(gamma-1)/2}
        current_radius = r_initial * std::pow(static_cast<float>(iter + 1u), (gamma_sppm - 1.0f) / 2.0f);

        // Progress
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
