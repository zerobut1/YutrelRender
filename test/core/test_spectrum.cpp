#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

#include <luisa/luisa-compute.h>

#include "base/renderer.h"
#include "spectrum/hero.h"
#include "utils/command_buffer.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{

[[nodiscard]] double sample_visible(double u) noexcept
{
    return std::clamp(
        538.0 - 138.888889 * std::atanh(0.85691062 - 1.82750197 * u),
        static_cast<double>(visible_wavelength_min),
        static_cast<double>(visible_wavelength_max));
}

[[nodiscard]] double visible_pdf(double lambda) noexcept
{
    auto c = std::cosh(0.0072 * (lambda - 538.0));
    return 0.0039398042 / (c * c);
}

[[nodiscard]] bool close(float actual, double expected, double tolerance) noexcept
{
    return std::abs(static_cast<double>(actual) - expected) <= tolerance;
}

[[nodiscard]] bool test_visible_wavelength_sampling(Device& device, Stream& stream)
{
    Renderer renderer{device};
    CommandBuffer command_buffer{stream};
    HeroWavelengthSpectrum spectrum;
    auto instance = spectrum.build(renderer, command_buffer);
    command_buffer << renderer.bindless_array().update();
    command_buffer << synchronize();

    constexpr auto n = 4u;
    std::array inputs{
        0.0f,
        0.125f,
        0.5f,
        std::nextafter(1.0f, 0.0f),
    };
    auto input_buffer  = device.create_buffer<float>(inputs.size());
    auto lambda_buffer = device.create_buffer<float4>(inputs.size());
    auto pdf_buffer    = device.create_buffer<float4>(inputs.size());

    Kernel1D kernel = [&instance](BufferFloat input, BufferFloat4 lambdas, BufferFloat4 pdfs) noexcept
    {
        auto i   = dispatch_id().x;
        auto swl = instance->sample(input.read(i));
        lambdas.write(i, make_float4(
                             swl.lambda(0u), swl.lambda(1u),
                             swl.lambda(2u), swl.lambda(3u)));
        pdfs.write(i, make_float4(
                         swl.pdf(0u), swl.pdf(1u),
                         swl.pdf(2u), swl.pdf(3u)));
    };
    auto shader = device.compile(kernel);

    std::array<float4, inputs.size()> lambdas{};
    std::array<float4, inputs.size()> pdfs{};
    stream << input_buffer.copy_from(inputs.data())
           << shader(input_buffer, lambda_buffer, pdf_buffer).dispatch(inputs.size())
           << lambda_buffer.copy_to(lambdas.data())
           << pdf_buffer.copy_to(pdfs.data())
           << synchronize();

    for (auto sample_index = 0u; sample_index < inputs.size(); sample_index++)
    {
        std::array actual_lambdas{
            lambdas[sample_index].x, lambdas[sample_index].y,
            lambdas[sample_index].z, lambdas[sample_index].w};
        std::array actual_pdfs{
            pdfs[sample_index].x, pdfs[sample_index].y,
            pdfs[sample_index].z, pdfs[sample_index].w};
        for (auto wavelength_index = 0u; wavelength_index < n; wavelength_index++)
        {
            auto up = std::fmod(
                static_cast<double>(inputs[sample_index]) +
                    static_cast<double>(wavelength_index) / n,
                1.0);
            auto expected_lambda = sample_visible(up);
            auto expected_pdf    = visible_pdf(expected_lambda);
            auto actual_lambda   = actual_lambdas[wavelength_index];
            auto actual_pdf      = actual_pdfs[wavelength_index];
            if (!std::isfinite(actual_lambda) || !std::isfinite(actual_pdf) ||
                actual_lambda < visible_wavelength_min ||
                actual_lambda > visible_wavelength_max || actual_pdf <= 0.0f ||
                !close(actual_lambda, expected_lambda, 2e-3) ||
                !close(actual_pdf, expected_pdf, 5e-7))
            {
                std::cerr << "Visible wavelength sample mismatch at sample "
                          << sample_index << ", wavelength " << wavelength_index
                          << ": lambda=" << actual_lambda
                          << " (expected " << expected_lambda << "), pdf="
                          << actual_pdf << " (expected " << expected_pdf << ")\n";
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool test_terminate_secondary(Device& device, Stream& stream)
{
    Kernel1D kernel = [](BufferFloat4 output) noexcept
    {
        SampledWavelengths swl{4u};
        swl.set_pdf(0u, 0.4f);
        swl.set_pdf(1u, 0.3f);
        swl.set_pdf(2u, 0.2f);
        swl.set_pdf(3u, 0.1f);
        swl.terminate_secondary();
        output.write(0u, make_float4(swl.pdf(0u), swl.pdf(1u), swl.pdf(2u), swl.pdf(3u)));
        swl.terminate_secondary();
        output.write(1u, make_float4(swl.pdf(0u), swl.pdf(1u), swl.pdf(2u), swl.pdf(3u)));

        SampledWavelengths single{1u};
        single.set_pdf(0u, 0.25f);
        single.terminate_secondary();
        output.write(2u, make_float4(single.pdf(0u)));
    };
    auto shader = device.compile(kernel);
    auto output = device.create_buffer<float4>(3u);
    std::array<float4, 3u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(luisa::span{result})
           << synchronize();

    auto terminated = [](float4 value) noexcept
    {
        return value.x == 0.1f && value.y == 0.0f &&
               value.z == 0.0f && value.w == 0.0f;
    };
    if (!terminated(result[0u]) || !terminated(result[1u]) || result[2u].x != 0.25f)
    {
        std::cerr << "terminate_secondary mismatch: first.x=" << result[0u].x
                  << ", second.x=" << result[1u].x
                  << ", single.x=" << result[2u].x << '\n';
        return false;
    }
    return true;
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
    auto stream = device.create_stream();
    return test_visible_wavelength_sampling(device, stream) &&
                   test_terminate_secondary(device, stream)
               ? 0
               : 1;
}
