#include "camera.h"

#include "base/film.h"
#include "base/renderer.h"
#include <cmath>
#include <numeric>
#include <random>

namespace Yutrel
{
float4x4 make_view_camera_to_world(float3 position, float3 lookat, float3 up) noexcept
{
    auto w = normalize(position - lookat);
    auto u = normalize(cross(up, w));
    auto v = cross(w, u);
    return make_float4x4(make_float4(u, 0.0f),
                         make_float4(v, 0.0f),
                         make_float4(w, 0.0f),
                         make_float4(position, 1.0f));
}

float camera_linear_determinant(const float4x4& camera_to_world) noexcept
{
    auto x = make_float3(camera_to_world[0]);
    auto y = make_float3(camera_to_world[1]);
    auto z = make_float3(camera_to_world[2]);
    return dot(x, cross(y, z));
}

luisa::optional<luisa::string> validate_camera_to_world(const float4x4& camera_to_world) noexcept
{
    for (auto column = 0u; column < 4u; column++)
    {
        for (auto row = 0u; row < 4u; row++)
        {
            if (!std::isfinite(camera_to_world[column][row]))
            {
                return luisa::string{"Camera-to-world matrix entries must be finite."};
            }
        }
    }
    constexpr auto affine_epsilon = 1e-6f;
    if (std::abs(camera_to_world[0].w) > affine_epsilon ||
        std::abs(camera_to_world[1].w) > affine_epsilon ||
        std::abs(camera_to_world[2].w) > affine_epsilon ||
        std::abs(camera_to_world[3].w - 1.0f) > affine_epsilon)
    {
        return luisa::string{"Camera-to-world matrix must be affine."};
    }
    auto x = make_float3(camera_to_world[0]);
    auto y = make_float3(camera_to_world[1]);
    auto z = make_float3(camera_to_world[2]);
    if (dot(x, x) < 1e-12f || dot(y, y) < 1e-12f || dot(z, z) < 1e-12f ||
        std::abs(camera_linear_determinant(camera_to_world)) < 1e-8f)
    {
        return luisa::string{"Camera-to-world matrix must have a non-singular linear part."};
    }
    return luisa::nullopt;
}

luisa::optional<luisa::string> validate_camera_world_up(float3 world_up) noexcept
{
    if (!std::isfinite(world_up.x) || !std::isfinite(world_up.y) || !std::isfinite(world_up.z))
    {
        return luisa::string{"Camera world-up vector must be finite."};
    }
    if (dot(world_up, world_up) < 1e-12f)
    {
        return luisa::string{"Camera world-up vector must be non-zero."};
    }
    return luisa::nullopt;
}

Camera::Camera(float4x4 camera_to_world, float3 world_up, float2 shutter_span,
               uint shutter_samples_count) noexcept
    : m_initial_camera_to_world{camera_to_world},
      m_world_up{normalize(world_up)},
      m_shutter_span{shutter_span},
      m_shutter_samples_count{shutter_samples_count}
{
    if (m_shutter_span.y < m_shutter_span.x) [[unlikely]]
    {
        LUISA_ERROR(
            "Invalid time span: [{}, {}]",
            m_shutter_span.x,
            m_shutter_span.y);
    }
}

Camera::~Camera() noexcept = default;

luisa::vector<Camera::ShutterSample> Camera::shutter_samples(uint spp, uint seed) const noexcept
{
    if (m_shutter_span.x == m_shutter_span.y)
    {
        return {ShutterSample{m_shutter_span.x, 1.0f, spp}};
    }

    auto shutter_samples_count = m_shutter_samples_count == 0u ? std::min(spp, 256u) : m_shutter_samples_count;
    if (shutter_samples_count > spp)
    {
        LUISA_WARNING("Too many shutter samples ({}), clamping to samples per pixel ({}).", shutter_samples_count, spp);
        shutter_samples_count = spp;
    }
    luisa::vector<ShutterSample> buckets(shutter_samples_count);
    auto duration = m_shutter_span.y - m_shutter_span.x;
    auto inv_n    = 1.0f / static_cast<float>(shutter_samples_count);
    std::uniform_real_distribution<float> dist{};
    std::default_random_engine random{seed};

    for (auto sample = 0u; sample < shutter_samples_count; sample++)
    {
        auto ts         = static_cast<float>(sample) * inv_n * duration;
        auto te         = static_cast<float>(sample + 1u) * inv_n * duration;
        auto a          = dist(random);
        auto t          = m_shutter_span.x + std::lerp(ts, te, a);
        auto w          = 1.0f;
        buckets[sample] = ShutterSample{t, w};
    }

    luisa::vector<uint> indices(shutter_samples_count);
    std::iota(indices.begin(), indices.end(), 0u);
    std::shuffle(indices.begin(), indices.end(), random);
    auto remainder          = spp % shutter_samples_count;
    auto samples_per_bucket = spp / shutter_samples_count;
    for (auto i = 0u; i < remainder; i++)
    {
        buckets[indices[i]].spp = samples_per_bucket + 1u;
    }
    for (auto i = remainder; i < shutter_samples_count; i++)
    {
        buckets[indices[i]].spp = samples_per_bucket;
    }
    auto sum_weights = std::accumulate(
        buckets.cbegin(),
        buckets.cend(),
        0.0f,
        [](float acc, const ShutterSample& s)
    {
        return acc + s.weight * s.spp;
    });

    if (sum_weights == 0.0) [[unlikely]]
    {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid shutter samples generated. "
            "Falling back to uniform shutter curve.");
        for (auto& s : buckets)
        {
            s.weight = 1.0f;
        }
    }
    else
    {
        auto scale = static_cast<float>(spp) / sum_weights;
        for (auto& s : buckets)
        {
            s.weight = static_cast<float>(s.weight * scale);
        }
    }

    return buckets;
}

Camera::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const Camera* camera, const Film* film, const Filter* filter) noexcept
    : m_renderer(renderer),
      m_camera(camera),
      m_film(film->build(renderer, command_buffer)),
      m_filter(filter->build(renderer)),
      m_host_camera_to_world(camera->initial_camera_to_world()),
      m_device_camera_to_world(renderer.arena_buffer<float4x4>(1u))
{
    command_buffer
        << m_device_camera_to_world.copy_from(&m_host_camera_to_world)
        << commit();
}

void Camera::Instance::set_camera_to_world(CommandBuffer& command_buffer, const float4x4& camera_to_world) noexcept
{
    m_host_camera_to_world = camera_to_world;
    command_buffer
        << m_device_camera_to_world.copy_from(&camera_to_world)
        << commit();
}

bool Camera::Instance::set_external_projection(
    CommandBuffer& command_buffer,
    uint2 resolution,
    float vertical_fov_degrees) noexcept
{
    static_cast<void>(command_buffer);
    static_cast<void>(resolution);
    static_cast<void>(vertical_fov_degrees);
    return false;
}

Camera::Sample Camera::Instance::generate_ray(Expr<uint2> pixel_coord, Expr<float> time, Expr<float2> u_filter, Expr<float2> u_lens) const noexcept
{
    auto [filter_offset, filter_weight] = m_filter->sample(u_filter);

    auto pixel = make_float2(pixel_coord) + 0.5f + filter_offset;

    auto ray_cs = generate_ray_in_camera_space(pixel, time, u_lens);

    auto c2w    = m_device_camera_to_world->read(0u);
    auto origin = make_float3(c2w * make_float4(ray_cs->origin(), 1.0f));

    auto d_camera  = make_float3x3(c2w) * ray_cs->direction();
    auto len       = length(d_camera);
    auto direction = ite(len < 1e-7f, make_float3(0.0f, 0.0f, -1.0f), d_camera / len);

    auto ray = make_ray(origin, direction);

    return {std::move(ray), pixel, filter_weight};
}

} // namespace Yutrel
