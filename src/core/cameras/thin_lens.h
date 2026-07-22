#pragma once

#include <cmath>

#include "base/camera.h"
#include "scene/spec_base.h"

namespace Yutrel
{
struct ThinLensCameraData
{
    float2 pixel_offset;
    float2 resolution;
    float focus_distance{};
    float lens_radius{};
    float projected_pixel_size{};
};

} // namespace Yutrel

LUISA_STRUCT(Yutrel::ThinLensCameraData,
             pixel_offset, resolution, focus_distance,
             lens_radius, projected_pixel_size){};

namespace Yutrel
{
class ThinLensCamera final : public Camera
{
public:
    class Instance final : public Camera::Instance
    {
    private:
        BufferView<ThinLensCameraData> m_device_data;

    public:
        Instance(Renderer& renderer, CommandBuffer& command_buffer, const ThinLensCamera* camera, const Film* film, const Filter* filter) noexcept;
        ~Instance() noexcept override = default;

    private:
        [[nodiscard]] Var<Ray> generate_ray_in_camera_space(Expr<float2> pixel, Expr<float> time, Expr<float2> u_lens) const noexcept override;
    };

private:
    float m_aperture;
    float m_focal_length;
    float m_focus_distance;

public:
    ThinLensCamera(float4x4 camera_to_world, float2 shutter_span,
                   uint shutter_samples_count, float aperture,
                   float focal_length, float focus_distance) noexcept;
    ~ThinLensCamera() noexcept override = default;

private:
    [[nodiscard]] bool requires_lens_sampling() const noexcept override { return true; }

    [[nodiscard]] auto aperture() const noexcept { return m_aperture; }
    [[nodiscard]] auto focal_length() const noexcept { return m_focal_length; }
    [[nodiscard]] auto focus_distance() const noexcept { return m_focus_distance; }

public:
    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(Renderer& renderer, CommandBuffer& command_buffer, const Film* film, const Filter* filter) const noexcept override;
};

class ThinLensCameraSpec final : public CameraSpec
{
private:
    float4x4 _camera_to_world;
    float2 _shutter_span;
    uint _shutter_samples_count;
    float _aperture;
    float _focal_length;
    float _focus_distance;

public:
    ThinLensCameraSpec(float4x4 camera_to_world, float2 shutter_span,
                       uint shutter_samples_count, float aperture,
                       float focal_length, float focus_distance) noexcept
        : _camera_to_world{camera_to_world}, _shutter_span{shutter_span},
          _shutter_samples_count{shutter_samples_count}, _aperture{aperture},
          _focal_length{focal_length}, _focus_distance{focus_distance} {}

    ThinLensCameraSpec(float3 position, float3 lookat, float3 up, float2 shutter_span, uint shutter_samples_count, float aperture, float focal_length, float focus_distance) noexcept
        : ThinLensCameraSpec{make_view_camera_to_world(position, lookat, up),
                             shutter_span, shutter_samples_count, aperture,
                             focal_length, focus_distance} {}

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        if (auto error = validate_camera_to_world(_camera_to_world))
        {
            return error;
        }
        if (!std::isfinite(_shutter_span.x) || !std::isfinite(_shutter_span.y) || _shutter_span.y < _shutter_span.x)
        {
            return spec_validation_error("Camera shutter span is invalid.");
        }
        if (!std::isfinite(_aperture) || !std::isfinite(_focal_length) || !std::isfinite(_focus_distance) || _aperture <= 0.0f || _focal_length <= 0.0f || _focus_distance <= 0.0f)
        {
            return spec_validation_error("Thin-lens camera lens parameters must be finite and positive.");
        }
        return luisa::nullopt;
    }
    [[nodiscard]] auto camera_to_world() const noexcept { return _camera_to_world; }
    [[nodiscard]] const Camera* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
