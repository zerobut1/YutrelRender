#include "integrator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <luisa/luisa-compute.h>

#include "base/camera.h"
#include "base/camera_controller.h"
#include "base/film.h"
#include "base/light_sampler.h"
#include "base/renderer.h"
#include "base/sampler.h"
#include "utils/command_buffer.h"
#include "utils/image_io.h"
#include "utils/progress_bar.h"

namespace Yutrel
{
Integrator::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const Integrator* integrator, const Sampler* sampler) noexcept
    : _renderer{renderer},
      _integrator{integrator},
      _sampler{sampler->build(renderer)},
      _light_sampler{LightSampler::create(renderer, command_buffer)}
{
}

Integrator::Instance::~Instance() noexcept = default;

ProgressiveIntegrator::ProgressiveIntegrator(uint max_depth) noexcept
    : _max_depth{max_depth}
{
}

ProgressiveIntegrator::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const ProgressiveIntegrator* integrator, const Sampler* sampler) noexcept
    : Integrator::Instance{renderer, command_buffer, integrator, sampler}
{
}

bool ProgressiveIntegrator::Instance::prepare_external_render() noexcept
{
    if (_external_render)
    {
        return true;
    }
    try
    {
        auto camera = renderer().camera();
        Kernel2D render_kernel = [&](ImageFloat accumulation, UInt sample_index) noexcept
        {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L        = Li(camera, sample_index, pixel_id, 0.0f);
            auto invalid  = any(compute::isnan(L)) | any(compute::isinf(L));
            L             = ite(invalid, make_float3(0.0f), L);

            auto previous = accumulation.read(pixel_id);
            accumulation.write(pixel_id, previous + make_float4(L, 1.0f));
        };
        LUISA_INFO("Start compiling external Integrator shader.");
        Clock clock_compile;
        _external_render = renderer().device().compile(render_kernel);
        LUISA_INFO("External Integrator shader compiled in {} ms.", clock_compile.toc());
        return static_cast<bool>(_external_render);
    }
    catch (const std::exception& exception)
    {
        LUISA_WARNING_WITH_LOCATION("Failed to compile external Integrator shader: {}", exception.what());
    }
    catch (...)
    {
        LUISA_WARNING_WITH_LOCATION("Failed to compile external Integrator shader.");
    }
    return false;
}

void ProgressiveIntegrator::Instance::reset_external_sampler(
    CommandBuffer& command_buffer,
    uint2 resolution) noexcept
{
    sampler()->reset(command_buffer, resolution, resolution.x * resolution.y);
}

bool ProgressiveIntegrator::Instance::render_external_sample(
    CommandBuffer& command_buffer,
    ImageView<float> accumulation,
    uint2 resolution,
    uint sample_index) noexcept
{
    auto accumulation_size = accumulation.size();
    if (!_external_render ||
        accumulation_size.x != resolution.x || accumulation_size.y != resolution.y ||
        accumulation.storage() != PixelStorage::FLOAT4)
    {
        return false;
    }
    command_buffer << _external_render(accumulation, sample_index).dispatch(resolution);
    return true;
}

void ProgressiveIntegrator::Instance::render(Stream& stream, bool enable_display)
{
    CommandBuffer command_buffer{stream};

    auto camera      = renderer().camera();
    auto resolution  = camera->film()->base()->resolution();
    auto pixel_count = resolution.x * resolution.y;

    renderer().reset_diagnostics(command_buffer);
    camera->film()->prepare(command_buffer, enable_display);
    bool output_saved = false;
    RenderDiagnostics diagnostics{};
    uint64_t host_nan_count{};
    uint64_t host_inf_count{};
    auto output_path = camera->film()->base()->filename();
    {
        render_one_camera(command_buffer, camera);
        if (camera->film()->should_close())
        {
            camera->film()->release();
            return;
        }
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

void ProgressiveIntegrator::Instance::render_interactive(Stream& stream)
{
    CommandBuffer command_buffer{stream};

    auto camera     = renderer().camera();
    auto resolution = camera->film()->base()->resolution();

    LUISA_INFO(
        "Interactive rendering at {}x{}, sampler_spp={}, seed={}.",
        resolution.x,
        resolution.y,
        sampler()->base()->spp(),
        sampler()->base()->seed());

    renderer().reset_diagnostics(command_buffer);
    camera->film()->prepare(command_buffer, true);
    sampler()->reset(command_buffer, resolution, resolution.x * resolution.y);
    command_buffer << synchronize();

    FpsCameraController controller{
        camera->camera_to_world(),
        camera->base()->world_up(),
        FpsCameraController::Config{}};

    Kernel2D render_kernel = [&](UInt frame_index, Float time) noexcept
    {
        set_block_size(16u, 16u, 1u);
        Var pixel_id = dispatch_id().xy();
        Var L        = Li(camera, frame_index, pixel_id, time);
        camera->film()->accumulate_single_writer(pixel_id, L, 1.0f);
    };
    LUISA_INFO("Start compiling interactive Integrator shader.");
    Clock clock_compile;
    auto render = renderer().device().compile(render_kernel);
    LUISA_INFO("Interactive Integrator shader compiled in {} ms.", clock_compile.toc());

    uint global_sample_index = 0u;
    LUISA_INFO("Interactive rendering started.");
    Clock clock_render;

    while (true)
    {
        // Process window events & draw current accumulation.
        camera->film()->show(command_buffer, true);
        if (camera->film()->should_close())
        {
            break;
        }

        // Update camera from input; reset accumulation if changed.
        if (controller.update())
        {
            auto c2w = controller.camera_to_world();
            camera->set_camera_to_world(command_buffer, c2w);
            camera->film()->prepare(command_buffer, true);
            sampler()->reset(command_buffer, resolution, resolution.x * resolution.y);
            global_sample_index = 0u;
            command_buffer << synchronize();
        }

        command_buffer
            << render(global_sample_index++, 0.0f).dispatch(resolution)
            << commit();
    }

    command_buffer << synchronize();
    camera->film()->release();
    LUISA_INFO(
        "Interactive rendering finished after {} samples in {} ms.",
        global_sample_index,
        clock_render.toc());
}

void ProgressiveIntegrator::Instance::render_one_camera(CommandBuffer& command_buffer, Camera::Instance* camera)
{
    auto spp        = sampler()->base()->spp();
    auto resolution = camera->film()->base()->resolution();

    sampler()->reset(command_buffer, resolution, resolution.x * resolution.y);
    command_buffer << synchronize();

    LUISA_INFO(
        "Rendering to '{}' at {}x{} and {} spp.",
        camera->film()->base()->filename().string(),
        resolution.x,
        resolution.y,
        spp);

    Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept
    {
        set_block_size(16u, 16u, 1u);
        Var pixel_id = dispatch_id().xy();
        Var L        = Li(camera, frame_index, pixel_id, time);
        camera->film()->accumulate_single_writer(pixel_id, L * shutter_weight, 1.0f);
    };

    LUISA_INFO("Start compiling Integrator shader.");
    Clock clock_compile;
    auto render = renderer().device().compile(render_kernel);
    LUISA_INFO("Integrator shader compiled in {} ms.", clock_compile.toc());
    command_buffer << synchronize();

    auto shutter_samples = camera->base()->shutter_samples(spp, sampler()->base()->seed());
    LUISA_INFO("Rendering started.");
    Clock clock_render;
    ProgressBar progress_bar;
    progress_bar.update(0.0);
    constexpr auto dispatches_per_commit = 4u;
    constexpr auto max_progress_updates  = 100u;
    auto progress_stride                 = std::max(1u, (spp + max_progress_updates - 1u) / max_progress_updates);
    auto dispatch_count                  = 0u;
    auto global_sample_index             = 0u;
    for (const auto& s : shutter_samples)
    {
        for (auto i = 0u; i < s.spp; i++)
        {
            dispatch_count++;
            command_buffer << render(global_sample_index++, s.time, s.weight).dispatch(resolution);

            if (camera->film()->show(command_buffer))
            {
                dispatch_count = 0u;
            }
            if (camera->film()->should_close()) [[unlikely]]
            {
                command_buffer << synchronize();
                progress_bar.cancel();
                return;
            }

            auto progress_due = global_sample_index < spp && global_sample_index % progress_stride == 0u;
            if (progress_due) [[unlikely]]
            {
                dispatch_count = 0u;
                auto p         = global_sample_index / static_cast<double>(spp);
                command_buffer << [&progress_bar, p]
                {
                    progress_bar.update(p);
                };
            }
            else if (dispatch_count >= dispatches_per_commit) [[unlikely]]
            {
                dispatch_count = 0u;
                command_buffer << commit();
            }
        }
    }
    command_buffer << synchronize();
    progress_bar.done();
    LUISA_INFO("Rendering finished in {} ms.", clock_render.toc());
}

} // namespace Yutrel
