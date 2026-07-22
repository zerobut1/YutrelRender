#pragma once

#include <cmath>

#include "base/camera.h"
#include "scene/spec_base.h"

namespace Yutrel
{
struct PinholeCameraData
{
    luisa::float2 resolution;
    float tan_half_fov{};
};
} // namespace Yutrel

LUISA_STRUCT(Yutrel::PinholeCameraData, resolution, tan_half_fov){};

namespace Yutrel
{
class PinholeCamera final : public Camera
{
    class Instance final : public Camera::Instance
    {
    private:
        BufferView<PinholeCameraData> m_device_data;

    public:
        Instance(Renderer& renderer, CommandBuffer& command_buffer, const PinholeCamera* camera, const Film* film, const Filter* filter) noexcept;
        ~Instance() noexcept override = default;

    private:
        [[nodiscard]] Var<Ray> generate_ray_in_camera_space(Expr<float2> pixel, Expr<float> time, Expr<float2> u_lens) const noexcept override;
    };

private:
    float m_fov;

public:
    PinholeCamera(float4x4 camera_to_world, float2 shutter_span,
                  uint shutter_samples_count, float fov) noexcept;
    ~PinholeCamera() noexcept override = default;

public:
    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(Renderer& renderer, CommandBuffer& command_buffer, const Film* film, const Filter* filter) const noexcept override;
};

class PinholeCameraSpec final : public CameraSpec
{
private:
    float4x4 _camera_to_world;
    float2 _shutter_span;
    uint _shutter_samples_count;
    float _fov;

public:
    PinholeCameraSpec(float4x4 camera_to_world, float2 shutter_span,
                      uint shutter_samples_count, float fov) noexcept
        : _camera_to_world{camera_to_world}, _shutter_span{shutter_span},
          _shutter_samples_count{shutter_samples_count}, _fov{fov} {}

    PinholeCameraSpec(float3 position, float3 lookat, float3 up, float2 shutter_span,
                      uint shutter_samples_count, float fov) noexcept
        : PinholeCameraSpec{make_view_camera_to_world(position, lookat, up),
                            shutter_span, shutter_samples_count, fov} {}

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        if (auto error = validate_camera_to_world(_camera_to_world))
        {
            return error;
        }
        if (!std::isfinite(_fov) || _fov <= 0.0f || _fov >= 180.0f)
        {
            return spec_validation_error("Pinhole camera FOV must be in (0, 180) degrees.");
        }
        if (!std::isfinite(_shutter_span.x) || !std::isfinite(_shutter_span.y) || _shutter_span.y < _shutter_span.x)
        {
            return spec_validation_error("Camera shutter span is invalid.");
        }
        return luisa::nullopt;
    }
    [[nodiscard]] auto camera_to_world() const noexcept { return _camera_to_world; }
    [[nodiscard]] auto shutter_span() const noexcept { return _shutter_span; }
    [[nodiscard]] auto fov() const noexcept { return _fov; }
    [[nodiscard]] const Camera* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
