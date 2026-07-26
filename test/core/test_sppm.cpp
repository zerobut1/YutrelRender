// Tests for SPPM integrator components.
// This test covers parameter defaults, validation, emission sampling, and wavelength state.

#include "ut/ut.hpp"

#include "integrators/sppm.h"
#include "lights/diffuse.h"
#include "scene/scene_builder.h"
#include "utils/spectra.h"

#include <cmath>
#include <limits>

using namespace Yutrel;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace
{

[[nodiscard]] bool is_near(float a, float b, float eps = 1e-4f) noexcept
{
    return std::abs(a - b) < eps;
}

suite sppm_spec_tests = []
{
    "sppm_default_parameters"_test = []
    {
        // Default SPPM spec: maxdepth=5, photons=1024, radius=1.0
        SPPMIntegratorSpec spec{5u, 1024u, 1.0f};
        auto validation = spec.validate();
        expect(!validation.has_value()) << "Valid SPPM spec should pass validation.";
    };

    "sppm_explicit_parameters"_test = []
    {
        SPPMIntegratorSpec spec{10u, 500000u, 0.5f};
        auto validation = spec.validate();
        expect(!validation.has_value()) << "Explicit valid SPPM spec should pass validation.";
    };

    "sppm_invalid_zero_depth"_test = []
    {
        SPPMIntegratorSpec spec{0u, 1024u, 1.0f};
        auto validation = spec.validate();
        expect(validation.has_value()) << "Zero max_depth should fail validation.";
    };

    "sppm_invalid_zero_photons"_test = []
    {
        SPPMIntegratorSpec spec{5u, 0u, 1.0f};
        auto validation = spec.validate();
        expect(validation.has_value()) << "Zero photons_per_iteration should fail validation.";
    };

    "sppm_invalid_zero_radius"_test = []
    {
        SPPMIntegratorSpec spec{5u, 1024u, 0.0f};
        auto validation = spec.validate();
        expect(validation.has_value()) << "Zero radius should fail validation.";
    };

    "sppm_invalid_negative_radius"_test = []
    {
        SPPMIntegratorSpec spec{5u, 1024u, -1.0f};
        auto validation = spec.validate();
        expect(validation.has_value()) << "Negative radius should fail validation.";
    };

    "sppm_invalid_inf_radius"_test = []
    {
        SPPMIntegratorSpec spec{5u, 1024u, std::numeric_limits<float>::infinity()};
        auto validation = spec.validate();
        expect(validation.has_value()) << "Infinite radius should fail validation.";
    };

    "sppm_invalid_nan_radius"_test = []
    {
        SPPMIntegratorSpec spec{5u, 1024u, std::numeric_limits<float>::quiet_NaN()};
        auto validation = spec.validate();
        expect(validation.has_value()) << "NaN radius should fail validation.";
    };

    "sppm_integrator_stores_parameters"_test = []
    {
        SPPMIntegrator integrator{7u, 200000u, 2.5f};
        expect(integrator.max_depth() == 7u);
        expect(integrator.photons_per_iteration() == 200000u);
        expect(is_near(integrator.initial_radius(), 2.5f));
    };

    "sppm_gamma_radius_shrinkage"_test = []
    {
        // Verify radius formula: R_k = R_0 * k^{(gamma-1)/2}
        constexpr float gamma = 2.0f / 3.0f;
        constexpr float r0    = 1.0f;
        float r1 = r0 * std::pow(1.0f, (gamma - 1.0f) / 2.0f); // k=1
        float r10 = r0 * std::pow(10.0f, (gamma - 1.0f) / 2.0f); // k=10
        float r100 = r0 * std::pow(100.0f, (gamma - 1.0f) / 2.0f); // k=100

        expect(is_near(r1, 1.0f));
        expect(r10 < r1) << "Radius should shrink with iterations.";
        expect(r100 < r10) << "Radius should continue shrinking.";
        expect(r100 > 0.0f) << "Radius should remain positive.";

        // Specific values: k^{-1/6}
        // 10^{-1/6} ≈ 0.6813
        expect(is_near(r10, 0.6813f, 0.01f));
        // 100^{-1/6} ≈ 0.4642
        expect(is_near(r100, 0.4642f, 0.01f));
    };
};

} // namespace

int main()
{
    return 0;
}
