// Test for Yutrel robustness primitives.
// This test covers geometric edge cases and device-side non-finite diagnostics.

#include <array>
#include <cmath>
#include <limits>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>

#include "base/interaction.h"
#include "base/renderer.h"
#include "utils/frame.h"

using namespace Yutrel;
using namespace luisa;
using namespace luisa::compute;

namespace
{

[[nodiscard]] bool test_axis_aligned_frame(Device& device)
{
    auto stream = device.create_stream();
    auto output = device.create_buffer<float4>(3u);

    Kernel1D kernel = [](BufferFloat4 result) noexcept
    {
        auto n     = make_float3(0.0f, 1.0f, 0.0f);
        auto frame = Frame::make(n);
        result.write(0u, make_float4(frame.s(), length(frame.s())));
        result.write(1u, make_float4(frame.t(), length(frame.t())));
        result.write(2u, make_float4(dot(frame.s(), n),
                                    dot(frame.t(), n),
                                    dot(frame.s(), frame.t()),
                                    0.0f));
    };
    auto shader = device.compile(kernel);
    std::array<float4, 3u> values{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(luisa::span{values.data(), values.size()})
           << synchronize();

    for (auto value : values)
    {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
            !std::isfinite(value.z) || !std::isfinite(value.w))
        {
            return false;
        }
    }
    return std::abs(values[0u].w - 1.0f) < 1e-4f &&
           std::abs(values[1u].w - 1.0f) < 1e-4f &&
           std::abs(values[2u].x) < 1e-4f &&
           std::abs(values[2u].y) < 1e-4f &&
           std::abs(values[2u].z) < 1e-4f;
}

[[nodiscard]] bool test_geometric_normal_ray_offset(Device& device)
{
    auto stream = device.create_stream();
    auto output = device.create_buffer<float3>(4u);

    Kernel1D kernel = [](BufferFloat3 result) noexcept
    {
        auto make_surface = [](Expr<float3> p) noexcept
        {
            return Interaction::from_surface(
                Shape::Handle::decode(make_uint4(0u)), p, make_float3(0.0f, 1.0f, 0.0f),
                make_float2(0.0f), p, Frame::make(make_float3(1.0f, 0.0f, 0.0f)),
                0u, 0u, 1.0f, true);
        };
        auto near_origin = make_surface(make_float3(0.0f));
        auto large_scale = make_surface(make_float3(1.0e6f));
        result.write(0u, near_origin.p_robust(make_float3(0.0f, 1.0f, 0.0f)));
        result.write(1u, near_origin.p_robust(make_float3(0.0f, -1.0f, 0.0f)));
        result.write(2u, large_scale.p_robust(make_float3(0.0f, 1.0f, 0.0f)));
        result.write(3u, large_scale.p_robust(make_float3(0.0f, -1.0f, 0.0f)));
    };
    auto shader = device.compile(kernel);
    std::array<float3, 4u> values{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(luisa::span{values})
           << synchronize();

    for (auto value : values)
    {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
        {
            return false;
        }
    }
    auto near_up_offset    = values[0u].y;
    auto near_down_offset  = -values[1u].y;
    auto large_up_offset   = values[2u].y - 1.0e6f;
    auto large_down_offset = 1.0e6f - values[3u].y;
    return std::abs(values[0u].x) < 1e-7f && std::abs(values[0u].z) < 1e-7f &&
           std::abs(values[1u].x) < 1e-7f && std::abs(values[1u].z) < 1e-7f &&
           near_up_offset > 0.0f && near_down_offset > 0.0f &&
           large_up_offset > near_up_offset && large_down_offset > near_down_offset;
}

[[nodiscard]] bool test_non_finite_diagnostics(Device& device)
{
    Renderer renderer{device};
    auto stream = device.create_stream();
    auto input  = device.create_buffer<float>(2u);

    Kernel1D kernel = [&](BufferFloat values) noexcept
    {
        auto value   = values.read(dispatch_x());
        auto has_nan = compute::isnan(value);
        auto has_inf = compute::isinf(value);
        renderer.record_path_non_finite(has_nan, has_inf);
        renderer.record_film_non_finite(has_nan, has_inf);
    };
    auto shader = device.compile(kernel);
    std::array<float, 2u> input_values{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
    };
    std::array<uint, 4u> diagnostics{};
    CommandBuffer command_buffer{stream};
    renderer.reset_diagnostics(command_buffer);
    command_buffer
        << input.copy_from(luisa::span{input_values.data(), input_values.size()})
        << shader(input).dispatch(2u);
    renderer.download_diagnostics(command_buffer, diagnostics);
    command_buffer << synchronize();

    return diagnostics[0u] == 1u && diagnostics[1u] == 1u &&
           diagnostics[2u] == 1u && diagnostics[3u] == 1u;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        LUISA_WARNING("Usage: {} <backend>", argv[0]);
        return 1;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    if (!test_axis_aligned_frame(device))
    {
        LUISA_WARNING("Axis-aligned frame test failed.");
        return 2;
    }
    if (!test_non_finite_diagnostics(device))
    {
        LUISA_WARNING("Non-finite diagnostic counter test failed.");
        return 3;
    }
    if (!test_geometric_normal_ray_offset(device))
    {
        LUISA_WARNING("Geometric-normal ray offset test failed.");
        return 4;
    }
    return 0;
}
