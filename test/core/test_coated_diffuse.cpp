#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

#include <luisa/luisa-compute.h>

#include "base/renderer.h"
#include "surfaces/dielectric.h"
#include "surfaces/diffuse.h"
#include "surfaces/layered.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

int main(int argc, char* argv[])
{
    if (argc < 2) { return 1; }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    Renderer renderer{device};

    Kernel1D kernel = [&renderer](BufferFloat4 output) noexcept
    {
        SampledWavelengths swl{1u};
        auto it = Interaction::from_surface(
            Shape::Handle::decode(make_uint4(0u)), make_float3(0.0f),
            make_float3(0.0f, 0.0f, 1.0f), make_float2(0.0f), make_float3(0.0f),
            Frame{}, 0u, 0u, 1.0f, true);

        auto make_layered = [&](const Interaction& layer_it, Expr<float2> alpha,
                                Expr<float> eta_i, Expr<float> eta_t,
                                Expr<float> reflectance, uint max_depth = 10u)
        {
            auto top = luisa::make_unique<Dielectric::Closure>(renderer, swl, 0.0f);
            top->bind(Dielectric::Closure::Context{
                .it = layer_it, .alpha = alpha, .eta_i = eta_i, .eta_t = eta_t,
                .dispersive = false});
            auto bottom = luisa::make_unique<Diffuse::Closure>(renderer, swl, 0.0f);
            bottom->bind(Diffuse::Closure::Context{
                .it = layer_it, .reflectance = SampledSpectrum{1u, reflectance}});
            luisa::unique_ptr<Surface::Closure> top_base{std::move(top)};
            luisa::unique_ptr<Surface::Closure> bottom_base{std::move(bottom)};
            auto layered = luisa::make_unique<Layered::Closure>(
                renderer, swl, 0.0f, std::move(top_base), std::move(bottom_base));
            layered->bind(Layered::Closure::Context{
                .it = layer_it,
                .thickness = 0.01f,
                .albedo = SampledSpectrum{1u},
                .g = 0.0f,
                .max_depth = max_depth,
                .samples = 1u});
            return layered;
        };

        auto wo = normalize(make_float3(0.2f, 0.1f, 1.0f));
        auto wi = normalize(make_float3(-0.1f, 0.3f, 1.0f));

        auto rough = make_layered(it, make_float2(0.1f), 1.0f, 1.5f, 0.5f);
        rough->pre_eval();
        auto rough_eval = rough->evaluate(wo, wi);
        auto rough_sample = rough->sample(wo, 0.37f, make_float2(0.23f, 0.71f));
        output.write(0u, make_float4(rough_eval.f[0u], rough_eval.pdf,
                                    rough_sample.eval.f[0u], rough_sample.eval.pdf));
        rough->post_eval();

        auto analytic = make_layered(it, make_float2(0.0f), 1.0f, 1.0f, 0.5f, 1u);
        analytic->pre_eval();
        auto analytic_eval = analytic->evaluate(wo, wi);
        auto expected_f = 0.5f * inv_pi * wi.z *
                          exp(-0.01f / wo.z) * exp(-0.01f / wi.z);
        auto expected_pdf = 0.1f * 0.25f * inv_pi + 0.9f * wi.z * inv_pi;
        output.write(1u, make_float4(analytic_eval.f[0u], expected_f,
                                    analytic_eval.pdf, expected_pdf));
        analytic->post_eval();

        auto delta_layered = make_layered(it, make_float2(0.0005f), 1.0f, 1.5f, 0.5f, 1u);
        delta_layered->pre_eval();
        auto delta_sample = delta_layered->sample(wo, 0.0f, make_float2(0.5f));
        output.write(2u, make_float4(delta_sample.eval.pdf, delta_sample.pdf_mis,
                                    ite(delta_sample.delta, 1.0f, 0.0f),
                                    delta_sample.eta));
        delta_layered->post_eval();

        Dielectric::Closure smooth{renderer, swl, 0.0f};
        smooth.bind(Dielectric::Closure::Context{
            .it = it, .alpha = make_float2(0.0f), .eta_i = 1.0f, .eta_t = 1.5f,
            .dispersive = false});
        smooth.pre_eval();
        auto normal = make_float3(0.0f, 0.0f, 1.0f);
        auto reflected = smooth.sample(normal, 0.5f, make_float2(0.5f),
                                       TransportMode::RADIANCE, ScatterFlags::Reflection);
        auto transmitted = smooth.sample(normal, 0.5f, make_float2(0.5f),
                                          TransportMode::RADIANCE, ScatterFlags::Transmission);
        output.write(3u, make_float4(reflected.eval.f[0u], reflected.eval.pdf,
                                    reflected.eta, ite(reflected.delta, 1.0f, 0.0f)));
        output.write(4u, make_float4(transmitted.eval.f[0u], transmitted.eval.pdf,
                                    transmitted.eta, ite(transmitted.delta, 1.0f, 0.0f)));
        smooth.post_eval();

        Dielectric::Closure tir{renderer, swl, 0.0f};
        tir.bind(Dielectric::Closure::Context{
            .it = it, .alpha = make_float2(0.0f), .eta_i = 1.5f, .eta_t = 1.0f,
            .dispersive = false});
        tir.pre_eval();
        auto tir_wo = normalize(make_float3(0.9f, 0.0f, 0.4358899f));
        auto tir_sample = tir.sample(tir_wo, 0.5f, make_float2(0.5f),
                                     TransportMode::RADIANCE, ScatterFlags::Transmission);
        output.write(5u, make_float4(tir_sample.eval.f[0u], tir_sample.eval.pdf,
                                    length(tir_sample.wi), tir_sample.eta));
        tir.post_eval();

        auto filtered = make_layered(it, make_float2(0.1f), 1.0f, 1.5f, 0.5f);
        filtered->pre_eval();
        auto reflection = filtered->evaluate(wo, wi, TransportMode::RADIANCE,
                                             ScatterFlags::Reflection);
        auto rejected = filtered->evaluate(wo, wi, TransportMode::RADIANCE,
                                           ScatterFlags::Transmission);
        output.write(6u, make_float4(reflection.f[0u], reflection.pdf,
                                    rejected.f[0u], rejected.pdf));
        filtered->post_eval();

        auto flipped_it = it;
        flipped_it.shading = it.shading.flipped(true);
        auto front = make_layered(it, make_float2(0.1f), 1.0f, 1.5f, 0.5f);
        auto back = make_layered(flipped_it, make_float2(0.1f), 1.0f, 1.5f, 0.5f);
        front->pre_eval();
        back->pre_eval();
        auto front_eval = front->evaluate(wo, wi);
        auto back_wo = flipped_it.shading.local_to_world(it.shading.world_to_local(wo));
        auto back_wi = flipped_it.shading.local_to_world(it.shading.world_to_local(wi));
        auto back_eval = back->evaluate(back_wo, back_wi);
        output.write(7u, make_float4(front_eval.f[0u], back_eval.f[0u],
                                    front_eval.pdf, back_eval.pdf));
        back->post_eval();
        front->post_eval();

        auto top = luisa::make_unique<Dielectric::Closure>(renderer, swl, 0.0f);
        top->bind(Dielectric::Closure::Context{
            .it = it, .alpha = make_float2(0.2f), .eta_i = 1.0f, .eta_t = 1.3f,
            .dispersive = false});
        auto bottom = luisa::make_unique<Dielectric::Closure>(renderer, swl, 0.0f);
        bottom->bind(Dielectric::Closure::Context{
            .it = it, .alpha = make_float2(0.2f), .eta_i = 1.3f, .eta_t = 1.0f,
            .dispersive = false});
        luisa::unique_ptr<Surface::Closure> top_base{std::move(top)};
        luisa::unique_ptr<Surface::Closure> bottom_base{std::move(bottom)};
        Layered::Closure transmissive{renderer, swl, 0.0f,
                                      std::move(top_base), std::move(bottom_base)};
        transmissive.bind(Layered::Closure::Context{
            .it = it, .thickness = 0.01f, .albedo = SampledSpectrum{1u},
            .g = 0.0f, .max_depth = 10u, .samples = 1u});
        transmissive.pre_eval();
        auto wi_other = normalize(make_float3(-0.1f, 0.2f, -1.0f));
        auto transmission = transmissive.evaluate(
            wo, wi_other, TransportMode::RADIANCE, ScatterFlags::Transmission);
        auto transmission_rejected = transmissive.evaluate(
            wo, wi_other, TransportMode::RADIANCE, ScatterFlags::Reflection);
        output.write(8u, make_float4(transmission.f[0u], transmission.pdf,
                                    transmission_rejected.f[0u],
                                    transmission_rejected.pdf));
        transmissive.post_eval();

        Dielectric::Closure rough_tir{renderer, swl, 0.0f};
        rough_tir.bind(Dielectric::Closure::Context{
            .it = it, .alpha = make_float2(0.001f), .eta_i = 1.0f, .eta_t = 1.5f,
            .dispersive = false});
        rough_tir.pre_eval();
        auto internal_wo = -normalize(make_float3(0.9f, 0.0f, 0.4358899f));
        auto internal_sample = rough_tir.sample(
            internal_wo, 0.5f, make_float2(0.5f),
            TransportMode::RADIANCE, ScatterFlags::All);
        output.write(9u, make_float4(
            internal_sample.eval.f[0u], internal_sample.eval.pdf,
            ite((internal_sample.event & Surface::event_transmit) == 0u, 1.0f, 0.0f),
            internal_sample.eta));
        rough_tir.post_eval();
    };

    auto shader = device.compile(kernel);
    auto output = device.create_buffer<float4>(10u);
    auto stream = device.create_stream();
    std::array<float4, 10u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(luisa::span{result})
           << synchronize();

    auto finite_nonnegative = [](float4 v) noexcept
    {
        return std::isfinite(v.x) && std::isfinite(v.y) &&
               std::isfinite(v.z) && std::isfinite(v.w) &&
               v.x >= 0.0f && v.y >= 0.0f && v.z >= 0.0f && v.w >= 0.0f;
    };
    auto relative_error = [](float a, float b) noexcept
    {
        return std::abs(a - b) / std::max(1e-6f, std::abs(b));
    };
    auto valid = std::all_of(result.begin(), result.end(), finite_nonnegative) &&
                 relative_error(result[1u].x, result[1u].y) < 1e-4f &&
                 relative_error(result[1u].z, result[1u].w) < 1e-4f &&
                 result[2u].x > 0.0f && result[2u].y > 0.0f && result[2u].z == 1.0f &&
                 relative_error(result[2u].w, 1.0f) < 1e-6f &&
                 relative_error(result[3u].x, 0.04f) < 1e-5f &&
                 relative_error(result[3u].y, 1.0f) < 1e-6f &&
                 relative_error(result[3u].z, 1.0f) < 1e-6f && result[3u].w == 1.0f &&
                 relative_error(result[4u].x, 0.96f / 2.25f) < 1e-5f &&
                 relative_error(result[4u].y, 1.0f) < 1e-6f &&
                 relative_error(result[4u].z, 1.5f) < 1e-6f && result[4u].w == 1.0f &&
                 result[5u].x == 0.0f && result[5u].y == 0.0f &&
                 result[6u].x > 0.0f && result[6u].y > 0.0f &&
                 result[6u].z == 0.0f && result[6u].w == 0.0f &&
                 relative_error(result[7u].x, result[7u].y) < 1e-5f &&
                 relative_error(result[7u].z, result[7u].w) < 1e-5f &&
                 result[8u].z == 0.0f && result[8u].w == 0.0f &&
                 result[9u].x > 0.0f && result[9u].y > 0.0f &&
                 result[9u].z == 1.0f && relative_error(result[9u].w, 1.0f) < 1e-6f;
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
