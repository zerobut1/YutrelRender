#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <luisa/luisa-compute.h>

#include "base/geometry.h"
#include "base/phase_function.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        return 1;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);

    constexpr auto sample_count = 16384u;
    Kernel1D kernel             = [](BufferFloat4 output) noexcept
    {
        auto i = dispatch_id().x;
        auto u = make_float2(
            (cast<float>(i) + 0.5f) / static_cast<float>(sample_count),
            fract(cast<float>(i) * 0.61803398875f));
        auto z  = 1.0f - 2.0f * u.x;
        auto r  = sqrt(max(0.0f, 1.0f - z * z));
        auto wi = make_float3(r * cos(2.0f * pi * u.y),
                              r * sin(2.0f * pi * u.y),
                              z);
        auto wo = make_float3(0.0f, 0.0f, 1.0f);
        HGPhaseFunction phase{0.3f};
        auto p            = phase.p(wo, wi);
        auto phase_sample = phase.sample(wo, u);

        $if(i == 0u)
        {
            auto sigma_t          = make_float3(0.2f, 0.7f, 1.5f);
            auto transmittance    = exp(-sigma_t * 2.0f);
            auto medium_interface = make_uint2(11u, 29u);
            auto inside           = select_medium_interface(
                medium_interface,
                wo,
                make_float3(0.0f, 0.0f, -1.0f));
            auto outside = select_medium_interface(
                medium_interface,
                wo,
                make_float3(0.0f, 0.0f, 1.0f));
            output.write(i, make_float4(transmittance, cast<float>(inside + outside)));
        }
        $else
        {
            output.write(i, make_float4(p, phase.pdf(wo, wi), abs(phase_sample.p - phase_sample.pdf), length(phase_sample.wi)));
        };
    };

    auto shader = device.compile(kernel);
    auto output = device.create_buffer<float4>(sample_count);
    auto stream = device.create_stream();
    std::vector<float4> result(sample_count);
    stream << shader(output).dispatch(sample_count)
           << output.copy_to(result.data())
           << synchronize();

    auto close = [](float a, float b, float tolerance) noexcept
    {
        return std::abs(a - b) <= tolerance;
    };
    auto valid = close(result[0u].x, std::exp(-0.4f), 1e-6f) &&
                 close(result[0u].y, std::exp(-1.4f), 1e-6f) &&
                 close(result[0u].z, std::exp(-3.0f), 1e-6f) &&
                 result[0u].w == 40.0f;

    double phase_integral = 0.0;
    for (auto i = 1u; i < result.size(); i++)
    {
        auto v = result[i];
        valid  = valid && std::isfinite(v.x) && std::isfinite(v.y) &&
                 std::isfinite(v.z) && std::isfinite(v.w) &&
                 v.x >= 0.0f && close(v.x, v.y, 1e-7f) &&
                 v.z <= 1e-7f && close(v.w, 1.0f, 2e-6f);
        phase_integral += static_cast<double>(v.x) * (4.0 * pi);
    }
    phase_integral /= static_cast<double>(sample_count - 1u);
    valid = valid && std::abs(phase_integral - 1.0) < 2e-3;

    if (!valid)
    {
        std::fprintf(stderr, "medium test failed: phase integral=%g, first=(%g,%g,%g,%g)\n", phase_integral, result[0u].x, result[0u].y, result[0u].z, result[0u].w);
    }
    return valid ? 0 : 1;
}
