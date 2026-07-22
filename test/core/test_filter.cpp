#include "ut/ut.hpp"

#include <cmath>
#include <concepts>
#include <limits>

#include "filters/box.h"
#include "filters/gaussian.h"
#include "filters/lanczos_sinc.h"
#include "filters/mitchell.h"
#include "filters/triangle.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace Yutrel;

static_assert(std::derived_from<BoxFilter, Filter>);
static_assert(std::derived_from<GaussianFilter, Filter>);
static_assert(std::derived_from<LanczosSincFilter, Filter>);
static_assert(std::derived_from<MitchellFilter, Filter>);
static_assert(std::derived_from<TriangleFilter, Filter>);

[[nodiscard]] bool is_near(float a, float b, float epsilon = 1e-6f) noexcept
{
    return std::abs(a - b) < epsilon;
}

template <typename Spec>
void check_radius_validation()
{
    expect(!Spec{0.5f}.validate().has_value());
    expect(Spec{0.0f}.validate().has_value());
    expect(Spec{std::numeric_limits<float>::infinity()}.validate().has_value());
}

static auto test_filter_registration = []
{
    "filter_specs_validate_radius"_test = []
    {
        check_radius_validation<BoxFilterSpec>();
        check_radius_validation<GaussianFilterSpec>();
        check_radius_validation<LanczosSincFilterSpec>();
        check_radius_validation<MitchellFilterSpec>();
        check_radius_validation<TriangleFilterSpec>();
    };

    "gaussian_filter_matches_pbrt_v4"_test = []
    {
        constexpr auto radius = 1.5f;
        constexpr auto sigma  = 0.5f;
        GaussianFilter filter{radius, sigma};
        auto gaussian = [](float x, float s) noexcept
        {
            return std::exp(-x * x / (2.0f * s * s)) /
                   std::sqrt(2.0f * pi * s * s);
        };
        auto expected = [&](float x) noexcept
        {
            return std::max(0.0f, gaussian(x, sigma) - gaussian(radius, sigma));
        };

        expect(is_near(filter.evaluate(0.0f), expected(0.0f)));
        expect(is_near(filter.evaluate(0.75f), expected(0.75f)));
        expect(is_near(filter.evaluate(0.75f), filter.evaluate(-0.75f)));
        expect(is_near(filter.evaluate(radius), 0.0f));
        expect(is_near(filter.evaluate(-radius), 0.0f));

        constexpr auto narrow_sigma = 0.25f;
        GaussianFilter narrow{radius, narrow_sigma};
        auto narrow_expected = std::max(
            0.0f,
            gaussian(0.75f, narrow_sigma) - gaussian(radius, narrow_sigma));
        expect(is_near(narrow.evaluate(0.75f), narrow_expected));
    };

    "gaussian_filter_spec_validates_sigma"_test = []
    {
        expect(!GaussianFilterSpec{1.5f, 0.5f}.validate().has_value());
        expect(GaussianFilterSpec{1.5f, 0.0f}.validate().has_value());
        expect(GaussianFilterSpec{1.5f, -0.5f}.validate().has_value());
        expect(GaussianFilterSpec{1.5f, std::numeric_limits<float>::infinity()}.validate().has_value());
        expect(GaussianFilterSpec{1.5f, std::numeric_limits<float>::quiet_NaN()}.validate().has_value());
    };
    return 0;
}();

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
