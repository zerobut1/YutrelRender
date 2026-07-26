#include <array>
#include <cmath>
#include <cstdio>

#include <luisa/luisa-compute.h>

#include "base/renderer.h"
#include "surfaces/dielectric.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{

[[nodiscard]] bool near(float actual, float expected, float tolerance = 1e-5f) noexcept
{
    return std::abs(actual - expected) <= tolerance;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        return 0;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    Renderer renderer{device};

    Dielectric scalar_surface{nullptr, nullptr, nullptr, nullptr,
                              luisa::nullopt, true, false};
    Dielectric f11_surface{nullptr, nullptr, nullptr, nullptr,
                           glass_f11_cauchy_eta(), true, false};
    Dielectric::Instance scalar{renderer, &scalar_surface, nullptr, nullptr, nullptr, nullptr};
    Dielectric::Instance f11{renderer, &f11_surface, nullptr, nullptr, nullptr, nullptr};

    Kernel1D kernel = [&renderer, &scalar, &f11](BufferFloat4 output) noexcept
    {
        auto it = Interaction::from_surface(
            Shape::Handle::decode(make_uint4(0u)), make_float3(0.0f),
            make_float3(0.0f, 0.0f, 1.0f), make_float2(0.0f), make_float3(0.0f),
            Frame{}, 0u, 0u, 1.0f, true);

        SampledWavelengths scalar_swl{4u};
        scalar_swl.set_lambda(0u, 486.13f);
        auto scalar_closure = scalar.create_closure(scalar_swl, 0.0f);
        scalar.populate_closure(scalar_closure.get(), it, make_float3(0.0f, 0.0f, 1.0f), 1.0f);
        auto scalar_eta = *scalar_closure->eta();
        auto scalar_dispersive = *scalar_closure->is_dispersive();
        scalar_closure->pre_eval();
        auto normal = make_float3(0.0f, 0.0f, 1.0f);
        auto reflected = scalar_closure->sample(
            normal, 0.5f, make_float2(0.5f),
            TransportMode::RADIANCE, ScatterFlags::Reflection);
        auto transmitted = scalar_closure->sample(
            normal, 0.5f, make_float2(0.5f),
            TransportMode::RADIANCE, ScatterFlags::Transmission);
        output.write(0u, make_float4(
                             scalar_eta, ite(scalar_dispersive, 1.0f, 0.0f),
                             reflected.eval.f[0u], reflected.eval.pdf));
        output.write(1u, make_float4(
                             transmitted.eval.f[0u], transmitted.eval.pdf,
                             transmitted.eta, transmitted.wi.z));
        scalar_closure->post_eval();

        auto sample_f11 = [&](uint index, float lambda_nm) noexcept
        {
            SampledWavelengths swl{4u};
            swl.set_lambda(0u, lambda_nm);
            auto closure = f11.create_closure(swl, 0.0f);
            auto wo = make_float3(0.6f, 0.0f, 0.8f);
            f11.populate_closure(closure.get(), it, wo, 1.0f);
            auto eta = *closure->eta();
            auto dispersive = *closure->is_dispersive();
            closure->pre_eval();
            auto sample = closure->sample(
                wo, 0.5f, make_float2(0.5f),
                TransportMode::RADIANCE, ScatterFlags::Transmission);
            output.write(index, make_float4(
                                    eta, ite(dispersive, 1.0f, 0.0f),
                                    sample.wi.x, sample.wi.z));
            closure->post_eval();
        };
        sample_f11(2u, 486.13f);
        sample_f11(3u, 656.27f);
    };

    auto shader = device.compile(kernel);
    auto output = device.create_buffer<float4>(4u);
    auto stream = device.create_stream();
    std::array<float4, 4u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(luisa::span{result})
           << synchronize();

    constexpr auto eta_blue = 1.8065917880168352f;
    constexpr auto eta_red  = 1.7754589288508518f;
    auto expected_wi_x = [](float eta) noexcept { return -0.6f / eta; };
    auto expected_wi_z = [&](float eta) noexcept
    {
        auto x = expected_wi_x(eta);
        return -std::sqrt(1.0f - x * x);
    };
    auto valid = near(result[0u].x, 1.5f) && result[0u].y == 0.0f &&
                 near(result[0u].z, 0.04f) && near(result[0u].w, 1.0f) &&
                 near(result[1u].x, 0.96f / 2.25f) && near(result[1u].y, 1.0f) &&
                 near(result[1u].z, 1.5f) && near(result[1u].w, -1.0f) &&
                 near(result[2u].x, eta_blue) && result[2u].y == 1.0f &&
                 near(result[2u].z, expected_wi_x(eta_blue)) &&
                 near(result[2u].w, expected_wi_z(eta_blue)) &&
                 near(result[3u].x, eta_red) && result[3u].y == 1.0f &&
                 near(result[3u].z, expected_wi_x(eta_red)) &&
                 near(result[3u].w, expected_wi_z(eta_red)) &&
                 result[2u].x > result[3u].x && result[2u].z > result[3u].z;
    if (!valid)
    {
        for (auto i = 0u; i < result.size(); i++)
        {
            auto v = result[i];
            std::fprintf(stderr, "%u=(%g,%g,%g,%g)\n", i, v.x, v.y, v.z, v.w);
        }
    }
    return valid ? 0 : 1;
}
