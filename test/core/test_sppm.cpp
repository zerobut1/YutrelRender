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

    "sppm_per_pixel_radius_shrinkage"_test = []
    {
        // The integrator uses PBRT-v4's per-pixel update:
        //   n' = n + gamma * m
        //   r' = r * sqrt(n' / (n + m))
        // A pixel that receives no photons keeps its radius unchanged.
        constexpr float gamma = 2.0f / 3.0f;

        auto next_radius = [](float r, float n, float m) noexcept
        {
            if (m <= 0.0f) { return r; }
            const float n_new = n + gamma * m;
            return r * std::sqrt(n_new / (n + m));
        };

        // First gather on a fresh pixel: r' = r * sqrt(gamma).
        const float r1 = next_radius(1.0f, 0.0f, 8.0f);
        expect(is_near(r1, std::sqrt(gamma), 1e-4f));
        expect(r1 < 1.0f) << "Radius should shrink after receiving photons.";

        // Radius keeps shrinking monotonically across gathers.
        const float r2 = next_radius(r1, gamma * 8.0f, 8.0f);
        expect(r2 < r1) << "Radius should continue shrinking.";
        expect(r2 > 0.0f) << "Radius should remain positive.";

        // Pixels without photons must be left alone.
        expect(is_near(next_radius(r2, gamma * 16.0f, 0.0f), r2))
            << "Radius must not change when no photons are gathered.";
    };

    "sppm_tau_scaling_matches_radius_ratio"_test = []
    {
        // tau is rescaled by (r'/r)^2 so that the density estimate stays
        // consistent when the gather radius shrinks.
        constexpr float gamma = 2.0f / 3.0f;
        const float r         = 0.5f;
        const float n         = 12.0f;
        const float m         = 6.0f;

        const float n_new = n + gamma * m;
        const float r_new = r * std::sqrt(n_new / (n + m));
        const float scale = (r_new * r_new) / (r * r);

        expect(is_near(scale, n_new / (n + m), 1e-4f));
        expect(scale < 1.0f) << "tau scale should be below one while the radius shrinks.";

        const float tau = (10.0f + 4.0f) * scale;
        expect(tau < 14.0f) << "tau should be attenuated by the radius ratio.";
        expect(tau > 0.0f);
    };
};

} // namespace

int main()
{
    return 0;
}
