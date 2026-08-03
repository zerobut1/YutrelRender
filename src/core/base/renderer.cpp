#include "renderer.h"

#include <cmath>
#include <limits>

#include <luisa/luisa-compute.h>

#include "base/camera.h"
#include "base/geometry.h"
#include "base/integrator.h"
#include "base/scene.h"
#include "environments/distant.h"

namespace Yutrel
{
Renderer::Renderer(Device& device) noexcept
    : m_device(device),
      m_diagnostics(device.create_buffer<uint>(diagnostic_count)),
      m_bindless_array(device.create_bindless_array()) {}

Renderer::~Renderer() noexcept = default;

uint Renderer::register_surface(CommandBuffer& command_buffer, const Surface* surface) noexcept
{
    if (auto iter = m_surface_tags.find(surface);
        iter != m_surface_tags.end())
    {
        return iter->second;
    }
    auto tag = m_surfaces.emplace(surface->build(*this, command_buffer));
    m_surface_tags.emplace(surface, tag);
    return tag;
}

uint Renderer::register_light(CommandBuffer& command_buffer, const Light* light) noexcept
{
    if (auto iter = m_light_tags.find(light);
        iter != m_light_tags.end())
    {
        return iter->second;
    }
    auto tag = m_lights.emplace(light->build(*this, command_buffer));
    m_light_tags.emplace(light, tag);
    return tag;
}

uint Renderer::register_medium(CommandBuffer& command_buffer, const Medium* medium) noexcept
{
    if (auto iter = m_medium_tags.find(medium); iter != m_medium_tags.end())
    {
        return iter->second;
    }
    auto tag = m_media.emplace(medium->build(*this, command_buffer)) + 1u;
    m_medium_tags.emplace(medium, tag);
    return tag;
}

luisa::unique_ptr<Renderer> Renderer::create(Device& device, Stream& stream, const Scene& scene) noexcept
{
    Clock clock;
    auto renderer = luisa::make_unique<Renderer>(device);

    CommandBuffer command_buffer{stream};

    auto update_bindless_if_dirty = [&renderer, &command_buffer]
    {
        if (renderer->m_bindless_array.dirty())
        {
            command_buffer << renderer->m_bindless_array.update();
        }
    };

    renderer->m_spectrum = scene.spectrum()->build(*renderer, command_buffer);
    update_bindless_if_dirty();

    renderer->m_camera = scene.camera()->build(*renderer, command_buffer, scene.film(), scene.filter());
    update_bindless_if_dirty();

    renderer->m_geometry = luisa::make_unique<Geometry>(*renderer);
    renderer->m_geometry->build(command_buffer, scene.instances());
    auto area_lights = renderer->m_geometry->light_instances();
    renderer->m_light_handles.assign(area_lights.begin(), area_lights.end());
    for (auto light : scene.standalone_lights())
    {
        if (light != nullptr && !light->is_null())
        {
            renderer->m_light_handles.emplace_back(Light::Handle{
                .instance_id = ~0u,
                .light_tag   = renderer->register_light(command_buffer, light),
            });
        }
    }
    update_bindless_if_dirty();

    if (!scene.environment()->is_black())
    {
        renderer->m_environment = scene.environment()->build(*renderer, command_buffer);
        update_bindless_if_dirty();
    }

    renderer->m_integrator = scene.integrator()->build(*renderer, command_buffer, scene.sampler());
    update_bindless_if_dirty();

    command_buffer << synchronize();

    LUISA_INFO(
        "Created renderer in {} ms with {} shape instances, {} area-light instances, {} standalone-light instances, {} surface implementations, {} light implementations, environment={}.",
        clock.toc(),
        renderer->m_geometry->instances().size(),
        renderer->m_geometry->light_instances().size(),
        scene.standalone_lights().size(),
        renderer->m_surfaces.size(),
        renderer->m_lights.size(),
        renderer->m_environment == nullptr ? "none" : "infinite");

    return renderer;
}

void Renderer::render(Stream& stream, bool enable_display)
{
    m_integrator->render(stream, enable_display);
}

void Renderer::render_interactive(Stream& stream)
{
    m_integrator->render_interactive(stream);
}

bool Renderer::prepare_external_render() noexcept
{
    auto integrator = dynamic_cast<ProgressiveIntegrator::Instance*>(m_integrator.get());
    return integrator != nullptr && integrator->prepare_external_render();
}

bool Renderer::prepare_external_scene_updates(
    CommandBuffer& command_buffer,
    luisa::span<const uint64_t> initial_instance_ids,
    const Surface* default_surface) noexcept
{
    return m_geometry != nullptr && m_geometry->prepare_external_updates(
                                        command_buffer,
                                        initial_instance_ids,
                                        default_surface);
}

bool Renderer::update_external_scene(
    CommandBuffer& command_buffer,
    luisa::span<const ExternalMeshUpdate> mesh_updates,
    luisa::optional<ExternalDirectionalLightState> light_update) noexcept
{
    if (m_geometry == nullptr)
    {
        return false;
    }
    if (light_update)
    {
        auto& light = *light_update;
        auto direction_length_squared = dot(light.direction, light.direction);
        if (!std::isfinite(light.color.x) || !std::isfinite(light.color.y) ||
            !std::isfinite(light.color.z) || light.color.x < 0.0f ||
            light.color.y < 0.0f || light.color.z < 0.0f ||
            !std::isfinite(light.illuminance_lux) || light.illuminance_lux < 0.0f ||
            !std::isfinite(light.direction.x) || !std::isfinite(light.direction.y) ||
            !std::isfinite(light.direction.z) || !std::isfinite(direction_length_squared) ||
            std::abs(direction_length_squared - 1.0f) > 1e-4f || light.enabled > 1u)
        {
            return false;
        }
        if (dynamic_cast<DistantEnvironment::Instance*>(m_environment.get()) == nullptr)
        {
            return false;
        }
    }
    if (!mesh_updates.empty() &&
        !m_geometry->update_external(command_buffer, mesh_updates))
    {
        return false;
    }
    if (light_update)
    {
        static_cast<DistantEnvironment::Instance*>(m_environment.get())
            ->update_external_state(command_buffer, *light_update);
    }
    return true;
}

bool Renderer::update_external_camera(
    CommandBuffer& command_buffer,
    const ExternalCameraState& state) noexcept
{
    auto pixel_count = static_cast<uint64_t>(state.resolution.x) * state.resolution.y;
    if (state.resolution.x == 0u || state.resolution.y == 0u ||
        pixel_count > std::numeric_limits<uint>::max() ||
        !std::isfinite(state.vertical_fov_degrees) ||
        state.vertical_fov_degrees <= 0.0f || state.vertical_fov_degrees >= 180.0f ||
        validate_camera_to_world(state.camera_to_world))
    {
        return false;
    }
    auto integrator = dynamic_cast<ProgressiveIntegrator::Instance*>(m_integrator.get());
    if (integrator == nullptr ||
        !m_camera->set_external_projection(
            command_buffer,
            state.resolution,
            state.vertical_fov_degrees))
    {
        return false;
    }
    m_camera->set_camera_to_world(command_buffer, state.camera_to_world);
    integrator->reset_external_sampler(command_buffer, state.resolution);
    return true;
}

bool Renderer::render_external_sample(
    CommandBuffer& command_buffer,
    ImageView<float> accumulation,
    uint2 resolution,
    uint sample_index) noexcept
{
    auto integrator = dynamic_cast<ProgressiveIntegrator::Instance*>(m_integrator.get());
    return integrator != nullptr && integrator->render_external_sample(
                                        command_buffer,
                                        accumulation,
                                        resolution,
                                        sample_index);
}

void Renderer::reset_diagnostics(CommandBuffer& command_buffer) noexcept
{
    command_buffer << m_diagnostics.copy_from(luisa::span{m_diagnostic_zeros.data(), m_diagnostic_zeros.size()});
}

void Renderer::download_diagnostics(CommandBuffer& command_buffer,
                                    std::array<uint, diagnostic_count>& values) const noexcept
{
    command_buffer << m_diagnostics.copy_to(luisa::span{values.data(), values.size()});
}

void Renderer::record_path_non_finite(Expr<bool> has_nan, Expr<bool> has_inf) const noexcept
{
    $if(has_nan) { m_diagnostics->atomic(diagnostic_path_nan).fetch_add(1u); };
    $if(has_inf) { m_diagnostics->atomic(diagnostic_path_inf).fetch_add(1u); };
}

void Renderer::record_film_non_finite(Expr<bool> has_nan, Expr<bool> has_inf) const noexcept
{
    $if(has_nan) { m_diagnostics->atomic(diagnostic_film_nan).fetch_add(1u); };
    $if(has_inf) { m_diagnostics->atomic(diagnostic_film_inf).fetch_add(1u); };
}

const Texture::Instance* Renderer::build_texture(CommandBuffer& command_buffer, const Texture* texture) noexcept
{
    if (texture == nullptr)
        return nullptr;
    if (auto iter = m_textures.find(texture); iter != m_textures.end())
    {
        return iter->second.get();
    }
    auto t = texture->build(*this, command_buffer);
    return m_textures.emplace(texture, std::move(t)).first->second.get();
}

} // namespace Yutrel
