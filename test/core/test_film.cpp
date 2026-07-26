// Test for Film sample accumulation.
// This test covers PBRT-compatible max-component clamping and its default.

#include <array>
#include <cmath>
#include <iostream>

#include <luisa/luisa-compute.h>

#include "base/film.h"
#include "base/renderer.h"
#include "utils/command_buffer.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{

[[nodiscard]] bool close(float actual, float expected, float tolerance = 1e-5f) noexcept
{
    return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool close(float4 actual, float4 expected) noexcept
{
    return close(actual.x, expected.x) && close(actual.y, expected.y) &&
           close(actual.z, expected.z) && close(actual.w, expected.w);
}

[[nodiscard]] bool test_sample_clamping(Device& device, Stream& stream)
{
    Renderer renderer{device};
    CommandBuffer command_buffer{stream};
    Film clamped{make_uint2(3u, 1u), false, "clamped.exr", 2.0f, 10.0f};
    Film unbounded{make_uint2(1u), false, "unbounded.exr", 2.0f};
    auto clamped_instance   = clamped.build(renderer, command_buffer);
    auto unbounded_instance = unbounded.build(renderer, command_buffer);
    clamped_instance->prepare(command_buffer, false);
    unbounded_instance->prepare(command_buffer, false);

    Kernel1D kernel = [&clamped_instance, &unbounded_instance]() noexcept
    {
        auto rgb = make_float3(20.0f, 4.0f, 1.0f);
        clamped_instance->accumulate_single_writer(make_uint2(0u), rgb, 1.0f);
        unbounded_instance->accumulate_single_writer(make_uint2(0u), rgb, 1.0f);
        clamped_instance->accumulate_single_writer(
            make_uint2(1u, 0u), make_float3(4.0f, -20.0f, 2.0f), 1.0f);
        clamped_instance->accumulate_single_writer(make_uint2(2u, 0u), rgb, 2.0f);
    };
    auto shader = device.compile(kernel);

    std::array<float4, 3u> clamped_result{};
    float4 unbounded_result{};
    command_buffer << shader().dispatch(1u);
    clamped_instance->download(command_buffer, clamped_result.data());
    unbounded_instance->download(command_buffer, &unbounded_result);
    command_buffer << synchronize();

    std::array expected_clamped{
        make_float4(10.0f, 2.0f, 0.5f, 1.0f),
        make_float4(8.0f, -40.0f, 4.0f, 1.0f),
        make_float4(5.0f, 1.0f, 0.25f, 1.0f),
    };
    auto expected_unbounded = make_float4(40.0f, 8.0f, 2.0f, 1.0f);
    for (auto i = 0u; i < clamped_result.size(); i++)
    {
        if (!close(clamped_result[i], expected_clamped[i]))
        {
            std::cerr << "Film clamping mismatch at pixel " << i << '\n';
            return false;
        }
    }
    if (!close(unbounded_result, expected_unbounded))
    {
        std::cerr << "Unbounded Film changed unexpectedly\n";
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
    return test_sample_clamping(device, stream) ? 0 : 1;
}
