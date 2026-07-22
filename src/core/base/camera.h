#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/dsl/syntax.h>

#include "base/film.h"
#include "base/filter.h"
#include "utils/command_buffer.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;

[[nodiscard]] float4x4 make_view_camera_to_world(float3 position, float3 lookat, float3 up) noexcept;
[[nodiscard]] float camera_linear_determinant(const float4x4& camera_to_world) noexcept;
[[nodiscard]] luisa::optional<luisa::string> validate_camera_to_world(const float4x4& camera_to_world) noexcept;

class Camera
{
public:
    struct Sample
    {
        Var<Ray> ray;
        Float2 pixel;
        Float weight;
    };

    struct ShutterSample
    {
        float time;
        float weight;
        uint spp;
    };

public:
    class Instance
    {
    private:
        const Renderer& m_renderer;
        const Camera* m_camera;

        luisa::unique_ptr<Film::Instance> m_film;
        luisa::unique_ptr<Filter::Instance> m_filter;
        float4x4 m_host_camera_to_world;
        BufferView<float4x4> m_device_camera_to_world;

    public:
        Instance(Renderer& renderer, CommandBuffer& command_buffer, const Camera* camera, const Film* film, const Filter* filter) noexcept;
        virtual ~Instance() noexcept = default;

        Instance()                           = delete;
        Instance(const Instance&)            = delete;
        Instance& operator=(const Instance&) = delete;
        Instance(Instance&&)                 = delete;
        Instance& operator=(Instance&&)      = delete;

    public:
        template <typename T = Camera>
            requires std::is_base_of_v<Camera, T>
        [[nodiscard]] auto base() const noexcept
        {
            return static_cast<const T*>(m_camera);
        }
        [[nodiscard]] auto film() const noexcept { return m_film.get(); }
        [[nodiscard]] auto filter() const noexcept { return m_filter.get(); }
        [[nodiscard]] auto camera_to_world() const noexcept { return m_host_camera_to_world; }

        void set_camera_to_world(CommandBuffer& command_buffer, const float4x4& camera_to_world) noexcept;
        [[nodiscard]] Sample generate_ray(Expr<uint2> pixel_coord, Expr<float> time, Expr<float2> u_filter, Expr<float2> u_lens) const noexcept;

    private:
        [[nodiscard]] virtual Var<Ray> generate_ray_in_camera_space(Expr<float2> pixel, Expr<float> time, Expr<float2> u_lens) const noexcept = 0;
    };

private:
    float4x4 m_initial_camera_to_world;
    float2 m_shutter_span;
    uint m_shutter_samples_count;

public:
    Camera(float4x4 camera_to_world, float2 shutter_span,
           uint shutter_samples_count) noexcept;
    virtual ~Camera() noexcept;

    Camera()                         = delete;
    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&)                 = delete;
    Camera& operator=(Camera&&)      = delete;

public:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Renderer& renderer, CommandBuffer& command_buffer, const Film* film, const Filter* filter) const noexcept = 0;

    [[nodiscard]] auto initial_camera_to_world() const noexcept { return m_initial_camera_to_world; }
    [[nodiscard]] auto shutter_span() const noexcept { return m_shutter_span; }
    [[nodiscard]] luisa::vector<ShutterSample> shutter_samples(uint spp, uint seed) const noexcept;
    [[nodiscard]] virtual bool requires_lens_sampling() const noexcept { return false; }
};

} // namespace Yutrel
