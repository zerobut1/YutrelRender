#include "pinhole.h"

#include "base/film.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
PinholeCamera::PinholeCamera(float4x4 camera_to_world, float2 shutter_span,
                             uint shutter_samples_count, float fov) noexcept
    : Camera{camera_to_world, shutter_span, shutter_samples_count},
      m_fov{radians(fov)}
{
}

luisa::unique_ptr<Camera::Instance> PinholeCamera::build(Renderer& renderer, CommandBuffer& command_buffer, const Film* film, const Filter* filter) const noexcept
{
    return luisa::make_unique<Instance>(renderer, command_buffer, this, film, filter);
}

PinholeCamera::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const PinholeCamera* camera, const Film* film, const Filter* filter) noexcept
    : Camera::Instance(renderer, command_buffer, camera, film, filter),
      m_device_data(renderer.arena_buffer<PinholeCameraData>(1u))
{
    PinholeCameraData host_data{make_float2(this->film()->base()->resolution()), tan(camera->m_fov * 0.5f)};
    m_device_data = renderer.arena_buffer<PinholeCameraData>(1u);
    command_buffer
        << m_device_data.copy_from(&host_data)
        << commit();
}

const Camera* PinholeCameraSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Camera, PinholeCamera>(
        _camera_to_world,
        _shutter_span,
        _shutter_samples_count,
        _fov);
}

Var<Ray> PinholeCamera::Instance::generate_ray_in_camera_space(Expr<float2> pixel, Expr<float> time, Expr<float2> u_lens) const noexcept
{
    auto data         = m_device_data->read(0u);
    auto p            = (pixel * 2.0f - data.resolution) * (data.tan_half_fov / data.resolution.y);
    auto direction_cs = normalize(make_float3(p.x, -p.y, -1.0f));

    return make_ray(make_float3(0.0f), direction_cs);
}

} // namespace Yutrel
