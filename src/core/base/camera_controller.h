#pragma once

#include <imgui.h>
#include <luisa/core/basic_types.h>
#include <luisa/dsl/syntax.h>

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class FpsCameraController
{
public:
    struct Config
    {
        float move_speed{10.0f};
        float fast_move_speed{30.0f};
        float mouse_sensitivity{1.0f / 600.0f};
        float max_pitch_radians{1.55334306f}; // ~89 degrees
    };

private:
    float3 m_position{};
    float3 m_forward{0.0f, 1.0f, 0.0f};
    float3 m_world_up{0.0f, 0.0f, 1.0f};
    float m_orientation_sign{1.0f};
    Config m_config;

public:
    explicit FpsCameraController(const float4x4& camera_to_world, Config config) noexcept;

    [[nodiscard]] bool update() noexcept;
    [[nodiscard]] float4x4 camera_to_world() const noexcept;

private:
    [[nodiscard]] static float3 rotate_axis_angle(float3 v, float3 axis, float radians) noexcept;
};

} // namespace Yutrel
