#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/dsl/syntax.h>

namespace Yutrel
{
using namespace luisa;

[[nodiscard]] const float3x3& cie_xyz_to_linear_srgb_matrix() noexcept;
[[nodiscard]] const float3x3& linear_srgb_to_cie_xyz_matrix() noexcept;

template <typename T>
[[nodiscard]] inline auto cie_xyz_to_linear_srgb(T&& xyz) noexcept
{
    return cie_xyz_to_linear_srgb_matrix() * std::forward<T>(xyz);
}

template <typename T>
[[nodiscard]] inline auto linear_srgb_to_cie_y(T&& rgb) noexcept
{
    return (linear_srgb_to_cie_xyz_matrix() * std::forward<T>(rgb)).y;
}

template <typename T>
[[nodiscard]] inline auto linear_srgb_to_cie_xyz(T&& rgb) noexcept
{
    return linear_srgb_to_cie_xyz_matrix() * std::forward<T>(rgb);
}

template <typename T>
[[nodiscard]] inline auto cie_xyz_to_lab(T&& xyz) noexcept
{
    static constexpr auto delta = 6.f / 29.f;
    auto f                      = [](const auto& x) noexcept
    {
        constexpr auto d3           = delta * delta * delta;
        constexpr auto one_over_3d2 = 1.f / (3.f * delta * delta);
        if constexpr (compute::is_dsl_v<T>)
        {
            return compute::ite(x > d3, compute::pow(x, 1.f / 3.f), one_over_3d2 * x + (4.f / 29.f));
        }
        else
        {
            return x > d3 ? std::pow(x, 1.f / 3.f) : one_over_3d2 * x + 4.f / 29.f;
        }
    };
    auto f_y_yn = f(xyz.y);
    auto l      = 116.f * f_y_yn - 16.f;
    auto a      = 500.f * (f(xyz.x / .950489f) - f_y_yn);
    auto b      = 200.f * (f_y_yn - f(xyz.z / 1.08884f));
    return make_float3(l, a, b);
}

template <typename T>
[[nodiscard]] inline auto lab_to_cie_xyz(T&& lab) noexcept
{
    static constexpr auto delta = 6.f / 29.f;
    auto f                      = [](const auto& x) noexcept
    {
        constexpr auto three_dd = 3.f * delta * delta;
        if constexpr (compute::is_dsl_v<T>)
        {
            return compute::ite(x > delta, x * x * x, three_dd * x - (three_dd * 4.f / 29.f));
        }
        else
        {
            return x > delta ? x * x * x : three_dd * x - three_dd * 4.f / 29.f;
        }
    };
    auto v = (lab.x + 16.f) / 116.f;
    auto x = .950489f * f(v + lab.y / 500.f);
    auto y = f(v);
    auto z = 1.08884f * f(v - lab.z / 200.f);
    return make_float3(x, y, z);
}

} // namespace Yutrel
