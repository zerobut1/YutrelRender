#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "base/spd.h"
#include "spectrum/hero.h"
#include "spectrum/srgb.h"
#include "utils/color_space.h"
#include "utils/spectra.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;
using namespace Yutrel;

namespace
{

[[nodiscard]] bool close(float a, float b, float tolerance = 2e-6f) noexcept
{
    return std::abs(a - b) <= tolerance;
}

[[nodiscard]] bool close_relative(float actual, float expected, float tolerance) noexcept
{
    auto scale = std::max(std::abs(expected), 1e-6f);
    return std::abs(actual - expected) <= tolerance * scale;
}

[[nodiscard]] float sigmoid(float value) noexcept
{
    if (std::isinf(value))
    {
        return value > 0.0f ? 1.0f : 0.0f;
    }
    return 0.5f + value / (2.0f * std::sqrt(1.0f + value * value));
}

[[nodiscard]] float3 decode_hero_illuminant_to_linear_srgb(float4 encoded) noexcept
{
    auto xyz = make_float3(0.0f);
    for (auto sample_index = 0u; sample_index < cie_sample_count; sample_index++)
    {
        auto wavelength = visible_wavelength_min + static_cast<float>(sample_index);
        auto polynomial = encoded.x * wavelength * wavelength +
                          encoded.y * wavelength + encoded.z;
        auto value = sigmoid(polynomial) * encoded.w * cie_d65_samples[sample_index];
        xyz += value * make_float3(
                           cie_x_samples[sample_index],
                           cie_y_samples[sample_index],
                           cie_z_samples[sample_index]);
    }
    return cie_xyz_to_linear_srgb(xyz / SPD::cie_y_integral());
}

} // namespace

static auto test_color_space_registration = []
{
    "d65_maps_to_linear_srgb_white"_test = []
    {
        auto white = make_float3(0.f);
        for (auto i = 0u; i < cie_sample_count; i++)
        {
            white += cie_d65_samples[i] *
                     make_float3(cie_x_samples[i], cie_y_samples[i], cie_z_samples[i]);
        }
        white /= white.y;

        const auto rgb = cie_xyz_to_linear_srgb(white);
        expect(close(rgb.x, 1.f));
        expect(close(rgb.y, 1.f));
        expect(close(rgb.z, 1.f));
    };

    "linear_srgb_matrices_are_inverses"_test = []
    {
        const auto rgb        = make_float3(.17f, .42f, .91f);
        const auto round_trip = cie_xyz_to_linear_srgb(linear_srgb_to_cie_xyz(rgb));
        expect(close(round_trip.x, rgb.x));
        expect(close(round_trip.y, rgb.y));
        expect(close(round_trip.z, rgb.z));
    };

    "srgb_spectrum_static_albedo_is_saturated"_test = []
    {
        SRGBSpectrum spectrum;
        const auto encoded = spectrum.encode_static_srgb_albedo(make_float3(-0.25f, 0.5f, 1.25f));
        expect(close(encoded.x, 0.0f));
        expect(close(encoded.y, 0.5f));
        expect(close(encoded.z, 1.0f));
        expect(close(encoded.w, 1.0f));
    };

    "hero_illuminant_round_trip_preserves_linear_srgb_and_cie_y"_test = []
    {
        HeroWavelengthSpectrum spectrum;
        constexpr std::array colors{
            float3{1.0f, 1.0f, 1.0f},
            float3{0.18f, 0.18f, 0.18f},
            float3{1.0f, 0.0f, 0.0f},
            float3{0.0f, 1.0f, 0.0f},
            float3{0.0f, 0.0f, 1.0f},
            float3{8.0f, 0.25f, 0.05f},
            float3{100000.0f, 60000.0f, 10000.0f},
        };
        for (auto color : colors)
        {
            auto encoded = spectrum.encode_static_srgb_illuminant(color);
            auto reconstructed = decode_hero_illuminant_to_linear_srgb(encoded);
            auto expected_y = linear_srgb_to_cie_y(color);
            auto actual_y = linear_srgb_to_cie_y(reconstructed);
            expect(close_relative(actual_y, expected_y, 0.01f));

            auto component_scale = std::max(std::max(color.x, color.y), std::max(color.z, 1e-3f));
            expect(std::abs(reconstructed.x - color.x) <= component_scale * 0.02f);
            expect(std::abs(reconstructed.y - color.y) <= component_scale * 0.02f);
            expect(std::abs(reconstructed.z - color.z) <= component_scale * 0.02f);
        }
    };

    "photometric_unit_reference_values"_test = []
    {
        constexpr auto pi = 3.14159265358979323846f;
        constexpr auto lambert_radiance = 100000.0f * 0.18f / pi;
        constexpr auto area_flux = pi * 2.0f * 100.0f;
        expect(close(lambert_radiance, 5729.578f, 1e-3f));
        expect(close(area_flux, 200.0f * pi, 1e-4f));
    };
    return 0;
}();

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
