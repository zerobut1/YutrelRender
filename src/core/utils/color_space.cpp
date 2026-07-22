#include "color_space.h"

#include <luisa/core/mathematics.h>

#include "spectra.h"

namespace Yutrel
{
namespace
{

struct LinearSRGBMatrices
{
    float3x3 rgb_from_xyz;
    float3x3 xyz_from_rgb;
};

[[nodiscard]] constexpr float3 xyz_from_xy(float x, float y) noexcept
{
    return make_float3(x / y, 1.f, (1.f - x - y) / y);
}

[[nodiscard]] LinearSRGBMatrices compute_linear_srgb_matrices() noexcept
{
    auto white = make_float3(0.f);
    for (auto i = 0u; i < cie_sample_count; i++)
    {
        white += cie_d65_samples[i] *
                 make_float3(cie_x_samples[i], cie_y_samples[i], cie_z_samples[i]);
    }
    white /= white.y;

    // Rec. ITU-R BT.709 primaries used by linear sRGB.
    const auto primaries = make_float3x3(
        xyz_from_xy(.64f, .33f),
        xyz_from_xy(.30f, .60f),
        xyz_from_xy(.15f, .06f));
    const auto scale        = inverse(primaries) * white;
    const auto xyz_from_rgb = make_float3x3(
        primaries[0] * scale.x,
        primaries[1] * scale.y,
        primaries[2] * scale.z);
    return {.rgb_from_xyz = inverse(xyz_from_rgb),
            .xyz_from_rgb = xyz_from_rgb};
}

[[nodiscard]] const LinearSRGBMatrices& linear_srgb_matrices() noexcept
{
    static const auto matrices = compute_linear_srgb_matrices();
    return matrices;
}

} // namespace

const float3x3& cie_xyz_to_linear_srgb_matrix() noexcept
{
    return linear_srgb_matrices().rgb_from_xyz;
}

const float3x3& linear_srgb_to_cie_xyz_matrix() noexcept
{
    return linear_srgb_matrices().xyz_from_rgb;
}

} // namespace Yutrel
