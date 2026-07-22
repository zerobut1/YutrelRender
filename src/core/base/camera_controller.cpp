#include "camera_controller.h"

#include <algorithm>
#include <cmath>

#include "base/camera.h"

namespace Yutrel
{

FpsCameraController::FpsCameraController(const float4x4& camera_to_world, float3 world_up, Config config) noexcept
    : m_world_up(normalize(world_up)), m_config(config)
{
    m_orientation_sign = camera_linear_determinant(camera_to_world) < 0.0f ? -1.0f : 1.0f;

    auto c0 = camera_to_world[0];
    auto c2 = camera_to_world[2];
    auto c3 = camera_to_world[3];

    m_position = make_float3(c3.x, c3.y, c3.z);

    auto w       = make_float3(c2.x, c2.y, c2.z);
    auto forward = -w;
    auto len     = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    if (len > 1e-6f)
    {
        m_forward = make_float3(forward.x / len, forward.y / len, forward.z / len);
    }

    // If forward is too close to up, nudge it.
    if (std::abs(dot(m_forward, m_world_up)) > 0.999f)
    {
        auto u = make_float3(c0.x, c0.y, c0.z);
        if (length(u) > 1e-6f)
        {
            m_forward = normalize(cross(u, m_world_up));
        }
    }
}

float3 FpsCameraController::rotate_axis_angle(float3 v, float3 axis, float radians) noexcept
{
    auto a_len = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    if (a_len < 1e-6f)
    {
        return v;
    }
    axis = make_float3(axis.x / a_len, axis.y / a_len, axis.z / a_len);

    auto c = std::cos(radians);
    auto s = std::sin(radians);
    // Rodrigues' rotation formula
    auto term1 = v * static_cast<float>(c);
    auto term2 = cross(axis, v) * static_cast<float>(s);
    auto term3 = axis * (dot(axis, v) * static_cast<float>(1.0f - c));
    return term1 + term2 + term3;
}

bool FpsCameraController::update() noexcept
{
    auto& io = ImGui::GetIO();

    if (io.WantCaptureKeyboard && io.WantCaptureMouse)
    {
        return false;
    }

    auto dt = io.DeltaTime;
    if (!(dt > 0.0f))
    {
        dt = 1.0f / 60.0f;
    }

    Input input;
    input.mouse_look = !io.WantCaptureMouse && io.MouseDown[1];
    if (input.mouse_look)
    {
        input.look_delta = make_float2(
            static_cast<float>(io.MouseDelta.x),
            static_cast<float>(io.MouseDelta.y));
    }

    if (!io.WantCaptureKeyboard)
    {
        if (ImGui::IsKeyDown(ImGuiKey_W))
        {
            input.movement.y += 1.0f;
        }
        if (ImGui::IsKeyDown(ImGuiKey_S))
        {
            input.movement.y -= 1.0f;
        }
        if (ImGui::IsKeyDown(ImGuiKey_D))
        {
            input.movement.x += 1.0f;
        }
        if (ImGui::IsKeyDown(ImGuiKey_A))
        {
            input.movement.x -= 1.0f;
        }
        if (ImGui::IsKeyDown(ImGuiKey_E))
        {
            input.movement.z += 1.0f;
        }
        if (ImGui::IsKeyDown(ImGuiKey_Q))
        {
            input.movement.z -= 1.0f;
        }
        input.fast = ImGui::IsKeyDown(ImGuiKey_LeftShift);
    }
    return update(input, dt);
}

bool FpsCameraController::update(const Input& input, float delta_time) noexcept
{
    bool changed = false;

    if (input.mouse_look && (input.look_delta.x != 0.0f || input.look_delta.y != 0.0f))
    {
        auto yaw   = -input.look_delta.x * m_config.mouse_sensitivity * m_orientation_sign;
        auto pitch = -input.look_delta.y * m_config.mouse_sensitivity * m_orientation_sign;

        auto forward = rotate_axis_angle(m_forward, m_world_up, yaw);
        auto right   = m_orientation_sign * cross(forward, m_world_up);
        if (length(right) > 1e-6f)
        {
            right   = normalize(right);
            forward = rotate_axis_angle(forward, right, pitch);
        }

        auto d             = clamp(dot(normalize(forward), m_world_up), -1.0f, 1.0f);
        auto pitch_angle   = std::asin(d);
        auto clamped_pitch = clamp(pitch_angle, -m_config.max_pitch_radians, m_config.max_pitch_radians);
        if (clamped_pitch != pitch_angle)
        {
            auto planar = forward - m_world_up * dot(forward, m_world_up);
            if (length(planar) > 1e-6f)
            {
                planar  = normalize(planar);
                forward = planar * static_cast<float>(std::cos(clamped_pitch)) +
                          m_world_up * static_cast<float>(std::sin(clamped_pitch));
            }
        }

        m_forward = normalize(forward);
        changed   = true;
    }

    auto right = m_orientation_sign * cross(m_forward, m_world_up);
    if (length(right) > 1e-6f)
    {
        right = normalize(right);
    }
    else
    {
        right = make_float3(1.0f, 0.0f, 0.0f);
    }
    auto move = right * input.movement.x +
                m_forward * input.movement.y +
                m_world_up * input.movement.z;

    auto move_len = length(move);
    if (move_len > 1e-6f)
    {
        move       = move * (1.0f / move_len);
        auto speed = input.fast ? m_config.fast_move_speed : m_config.move_speed;
        auto dt    = delta_time > 0.0f ? delta_time : 1.0f / 60.0f;
        m_position = m_position + move * (speed * dt);
        changed    = true;
    }

    return changed;
}

float4x4 FpsCameraController::camera_to_world() const noexcept
{
    // Match Camera constructor convention:
    // w = normalize(position - lookat) = -forward
    auto w           = normalize(make_float3(-m_forward.x, -m_forward.y, -m_forward.z));
    auto canonical_u = normalize(cross(m_world_up, w));
    auto u           = canonical_u * m_orientation_sign;
    auto v           = cross(w, canonical_u);

    return make_float4x4(
        make_float4(u, 0.0f),
        make_float4(v, 0.0f),
        make_float4(w, 0.0f),
        make_float4(m_position, 1.0f));
}

} // namespace Yutrel
