#include "ut/ut.hpp"

#include <cmath>
#include <limits>
#include <string_view>

#include "base/camera_controller.h"
#include "cameras/pinhole.h"
#include "cameras/thin_lens.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace Yutrel;

namespace
{

[[nodiscard]] bool is_near(float a, float b, float epsilon = 1e-5f) noexcept
{
    return std::abs(a - b) <= epsilon;
}

[[nodiscard]] bool near_matrix(const float4x4& a, const float4x4& b) noexcept
{
    for (auto column = 0u; column < 4u; column++)
    {
        for (auto row = 0u; row < 4u; row++)
        {
            if (!is_near(a[column][row], b[column][row]))
            {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool contains(
    const luisa::optional<luisa::string>& error,
    std::string_view needle) noexcept
{
    return error && std::string_view{*error}.find(needle) != std::string_view::npos;
}

static auto test_camera_registration = []
{
    "view_convenience_builds_expected_matrix"_test = []
    {
        auto shutter = make_float2(0.0f, 1.0f);
        PinholeCameraSpec pinhole{
            make_float3(0.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 1.0f, 0.0f),
            shutter,
            0u,
            45.0f};
        ThinLensCameraSpec thin_lens{
            make_float3(0.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 1.0f, 0.0f),
            shutter,
            0u,
            2.0f,
            35.0f,
            10.0f};
        expect(near_matrix(pinhole.camera_to_world(), make_float4x4(1.0f)));
        expect(near_matrix(thin_lens.camera_to_world(), make_float4x4(1.0f)));

        auto position = make_float3(1.0f, 2.0f, 3.0f);
        auto lookat   = make_float3(-2.0f, 1.0f, -4.0f);
        auto up       = make_float3(0.2f, 1.0f, 0.1f);
        PinholeCameraSpec oblique_pinhole{position, lookat, up, shutter, 0u, 50.0f};
        ThinLensCameraSpec oblique_thin_lens{position, lookat, up, shutter, 0u, 2.8f, 50.0f, 8.0f};
        expect(near_matrix(oblique_pinhole.camera_to_world(), oblique_thin_lens.camera_to_world()));
        expect(camera_linear_determinant(oblique_pinhole.camera_to_world()) > 0.0f);
    };

    "camera_matrix_validation"_test = []
    {
        auto valid = make_float4x4(
            make_float4(2.0f, 0.0f, 0.0f, 0.0f),
            make_float4(1.0f, 3.0f, 0.0f, 0.0f),
            make_float4(0.0f, 1.0f, 4.0f, 0.0f),
            make_float4(4.0f, 5.0f, 6.0f, 1.0f));
        expect(!PinholeCameraSpec{valid, make_float2(0.0f, 1.0f), 0u, 45.0f}.validate().has_value());
        expect(!ThinLensCameraSpec{valid, make_float2(0.0f, 1.0f), 0u, 2.0f, 35.0f, 10.0f}.validate().has_value());

        auto non_finite    = valid;
        non_finite[0].x    = std::numeric_limits<float>::infinity();
        auto non_affine    = valid;
        non_affine[0].w    = 0.25f;
        auto singular      = valid;
        singular[0]        = make_float4(0.0f);
        auto pinhole_error = PinholeCameraSpec{non_finite, make_float2(0.0f, 1.0f), 0u, 45.0f}.validate();
        expect(contains(pinhole_error, "finite"));
        pinhole_error = PinholeCameraSpec{non_affine, make_float2(0.0f, 1.0f), 0u, 45.0f}.validate();
        expect(contains(pinhole_error, "affine"));
        pinhole_error = PinholeCameraSpec{singular, make_float2(0.0f, 1.0f), 0u, 45.0f}.validate();
        expect(contains(pinhole_error, "singular"));
        auto thin_lens_error = ThinLensCameraSpec{
            singular, make_float2(0.0f, 1.0f), 0u, 2.0f, 35.0f, 10.0f}
                                   .validate();
        expect(contains(thin_lens_error, "singular"));
    };

    "fps_controller_preserves_orientation"_test = []
    {
        auto regular = make_float4x4(
            make_float4(2.0f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 3.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 4.0f, 0.0f),
            make_float4(5.0f, 6.0f, 7.0f, 1.0f));
        auto mirrored = regular;
        mirrored[0].x = -mirrored[0].x;

        FpsCameraController regular_controller{regular, FpsCameraController::Config{}};
        FpsCameraController mirrored_controller{mirrored, FpsCameraController::Config{}};
        auto regular_result  = regular_controller.camera_to_world();
        auto mirrored_result = mirrored_controller.camera_to_world();

        expect(camera_linear_determinant(regular_result) > 0.0f);
        expect(camera_linear_determinant(mirrored_result) < 0.0f);
        expect(is_near(regular_result[0].x, 1.0f));
        expect(is_near(mirrored_result[0].x, -1.0f));
        expect(is_near(regular_result[1].y, 1.0f));
        expect(is_near(mirrored_result[1].y, 1.0f));
        expect(is_near(regular_result[2].z, 1.0f));
        expect(is_near(mirrored_result[2].z, 1.0f));
        expect(is_near(regular_result[3].x, 5.0f));
        expect(is_near(regular_result[3].y, 6.0f));
        expect(is_near(regular_result[3].z, 7.0f));
        expect(is_near(regular_result[3].x, mirrored_result[3].x));
        expect(is_near(regular_result[3].y, mirrored_result[3].y));
        expect(is_near(regular_result[3].z, mirrored_result[3].z));
        expect(is_near(regular_result[3].w, mirrored_result[3].w));
    };

    return 0;
}();

} // namespace

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
