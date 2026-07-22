#include "ut/ut.hpp"

#include <cmath>

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
    return 0;
}();

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
