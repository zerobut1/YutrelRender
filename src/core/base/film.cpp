#include "film.h"

#include <luisa/luisa-compute.h>

#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
Film::Film(uint2 resolution, bool display_hdr, std::filesystem::path filename,
           float imaging_ratio, float max_component_value) noexcept
    : m_resolution{resolution},
      m_display_hdr{display_hdr},
      m_filename{std::move(filename)},
      m_imaging_ratio{imaging_ratio},
      m_max_component_value{max_component_value} {}

const Film* RGBFilmSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Film, Film>(
        _resolution, _display_hdr, _filename, _imaging_ratio, _max_component_value);
}

Film::~Film() noexcept = default;

Float4 Film::Instance::filtered_contribution(Expr<float3> rgb, Expr<float> effective_spp) const noexcept
{
    LUISA_ASSERT(m_image && m_converted, "Film is not prepared.");

    auto contribution = def(make_float4(0.0f));
    auto exposed_rgb  = rgb * base()->imaging_ratio();
    auto has_nan      = any(compute::isnan(exposed_rgb));
    auto has_inf      = any(compute::isinf(exposed_rgb));
    renderer().record_film_non_finite(has_nan, has_inf);
    $if(!(has_nan | has_inf))
    {
        if (std::isfinite(base()->max_component_value()))
        {
            auto max_component = max(exposed_rgb.x, max(exposed_rgb.y, exposed_rgb.z));
            $if(max_component > base()->max_component_value())
            {
                exposed_rgb *= base()->max_component_value() / max_component;
            };
        }
        contribution = make_float4(exposed_rgb, effective_spp);
    };
    return contribution;
}

void Film::Instance::accumulate(Expr<uint2> pixel, Expr<float3> rgb, Expr<float> effective_spp) const noexcept
{
    auto pixel_id     = pixel.y * base()->resolution().x + pixel.x;
    auto contribution = filtered_contribution(rgb, effective_spp);

    $if(any(contribution.xyz() != 0.0f))
    {
        m_image->atomic(pixel_id).x.fetch_add(contribution.x);
        m_image->atomic(pixel_id).y.fetch_add(contribution.y);
        m_image->atomic(pixel_id).z.fetch_add(contribution.z);
    };
    $if(contribution.w != 0.0f)
    {
        m_image->atomic(pixel_id).w.fetch_add(contribution.w);
    };
}

void Film::Instance::accumulate_single_writer(Expr<uint2> pixel, Expr<float3> rgb, Expr<float> effective_spp) const noexcept
{
    auto pixel_id     = pixel.y * base()->resolution().x + pixel.x;
    auto contribution = filtered_contribution(rgb, effective_spp);

    $if(any(contribution != 0.0f))
    {
        auto previous = m_image->read(pixel_id);
        m_image->write(pixel_id, previous + contribution);
    };
}

void Film::Instance::prepare(CommandBuffer& command_buffer, bool enable_display) noexcept
{
    m_rendering_finished = false;

    // render image
    uint2 render_resolution = base()->resolution();
    auto pixel_count        = render_resolution.x * render_resolution.y;

    if (!m_image)
    {
        m_image = renderer().device().create_buffer<float4>(pixel_count);

        Kernel1D clear_image_kernel = [](BufferFloat4 image) noexcept
        {
            image.write(dispatch_x(), make_float4(0.f));
        };
        m_clear_image = m_renderer.device().compile(clear_image_kernel);
    }
    if (!m_converted)
    {
        m_converted = renderer().device().create_buffer<float4>(pixel_count);

        Kernel1D convert_image_kernel = [this](BufferFloat4 accum, BufferFloat4 output) noexcept
        {
            auto i     = dispatch_x();
            auto c     = accum.read(i);
            auto n     = max(c.w, 1.f);
            auto scale = (1.f / n);
            output.write(i, make_float4(scale * c.xyz(), 1.f));
        };
        m_convert_image = m_renderer.device().compile(convert_image_kernel);
    }
    command_buffer << m_clear_image(m_image).dispatch(pixel_count);

    if (enable_display && !m_window)
    {
        auto&& device = m_renderer.device();
        m_stream      = command_buffer.stream();

        auto window_resolution = render_resolution;

        m_window = luisa::make_unique<ImGuiWindow>(
            device,
            *command_buffer.stream(),
            "Yutrel",
            ImGuiWindow::Config{
                .size         = window_resolution,
                .vsync        = true,
                .hdr          = base()->display_hdr(),
                .back_buffers = 3,
            });
        m_framebuffer = device.create_image<float>(PixelStorage::FLOAT4, render_resolution);
        m_background  = m_window->register_texture(m_framebuffer, compute::Sampler::linear_linear_zero());

        Kernel2D blit_kernel = [&](Bool is_ldr) noexcept
        {
            auto pixel_coord = dispatch_id().xy();
            auto pixel_id    = pixel_coord.y * base()->resolution().x + pixel_coord.x;
            auto image_data  = m_image->read(pixel_id);
            auto inv_n       = (1.0f / max(image_data.w, 1e-6f));
            auto color       = image_data.xyz() * inv_n;

            $if(is_ldr)
            {
                // linear to sRGB
                color = ite(color <= .0031308f, color * 12.92f, 1.055f * pow(color, 1.f / 2.4f) - .055f);
            };

            m_framebuffer->write(pixel_coord, make_float4(color, 1.0f));
        };
        m_blit = device.compile(blit_kernel);

        Kernel2D clear_kernel = [](ImageFloat image) noexcept
        {
            image->write(dispatch_id().xy(), make_float4(0.0f));
        };
        m_clear = device.compile(clear_kernel);
    }
    if (enable_display)
    {
        m_framerate.clear();
    }
}

void Film::Instance::download(CommandBuffer& command_buffer, float4* buffer) const noexcept
{
    LUISA_ASSERT(m_image && m_converted, "Film is not prepared.");

    auto pixel_count = base()->resolution().x * base()->resolution().y;

    command_buffer
        << m_convert_image(m_image, m_converted).dispatch(pixel_count)
        << m_converted.copy_to(buffer);
}

void Film::Instance::release() noexcept
{
    m_rendering_finished = true;

    if (!m_window)
    {
        return;
    }

    CommandBuffer command_buffer{*m_stream};

    while (!m_window->should_close())
    {
        this->show(command_buffer);
    }

    command_buffer << synchronize();
    m_window      = nullptr;
    m_framebuffer = {};
}

bool Film::Instance::should_close() const noexcept
{
    return m_window && m_window->should_close();
}

bool Film::Instance::show(CommandBuffer& command_buffer, bool force) const noexcept
{
    if (!m_window)
    {
        return false;
    }

    LUISA_ASSERT(command_buffer.stream() == m_stream, "Command buffer stream mismatch.");

    static const auto target_fps = 60.0;

    if (!force && m_framerate.duration() < 1.0 / target_fps)
    {
        return false;
    }

    if (!m_rendering_finished && this->should_close())
    {
        // Let the caller decide how to exit (interactive loop wants to break gracefully).
        command_buffer << synchronize();
        return true;
    }
    m_framerate.record();

    m_window->prepare_frame();

    auto is_ldr = m_window->framebuffer().storage() != PixelStorage::FLOAT4;

    command_buffer
        << m_clear(m_window->framebuffer()).dispatch(m_window->framebuffer().size())
        << m_blit(is_ldr).dispatch(base()->resolution())
        << commit();
    display();
    m_window->render_frame();

    return true;
}

void Film::Instance::display() const noexcept
{
    auto viewport = ImGui::GetMainViewport();

    // aspect fit
    auto frame_size      = make_float2(base()->resolution());
    auto viewport_size   = make_float2(viewport->Size.x, viewport->Size.y);
    auto aspect          = frame_size.x / frame_size.y;
    auto viewport_aspect = viewport_size.x / viewport_size.y;
    auto ratio           = aspect > viewport_aspect ? viewport_size.x / frame_size.x : viewport_size.y / frame_size.y;

    auto bg_size = frame_size * ratio;
    auto p_min   = make_float2(viewport->Pos.x, viewport->Pos.y) + 0.5f * (viewport_size - bg_size);

    ImGui::GetBackgroundDrawList()->AddImage(m_background, ImVec2{p_min.x, p_min.y}, ImVec2{p_min.x + bg_size.x, p_min.y + bg_size.y});
    ImGui::Begin("Console", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    {
        ImGui::Text("Render: %ux%u",
                    base()->resolution().x,
                    base()->resolution().y);
        ImGui::Text("Display: %ux%u (%.2ffps)",
                    static_cast<uint>(viewport->Size.x),
                    static_cast<uint>(viewport->Size.y),
                    ImGui::GetIO().Framerate);
    }
    ImGui::End();
}

} // namespace Yutrel
