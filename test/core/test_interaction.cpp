#include <array>
#include <cmath>
#include <cstdio>

#include <luisa/luisa-compute.h>

#include "base/interaction.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{
[[nodiscard]] bool finite(float4 v) noexcept
{
    return std::isfinite(v.x) && std::isfinite(v.y) &&
           std::isfinite(v.z) && std::isfinite(v.w);
}

[[nodiscard]] bool near(float a, float b, float epsilon = 1e-6f) noexcept
{
    return std::abs(a - b) <= epsilon;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) { return 1; }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto values = device.create_buffer<float4>(14u);
    auto states = device.create_buffer<uint4>(2u);

    Kernel1D kernel = [](BufferFloat4 output, BufferUInt4 state) noexcept
    {
        Interaction empty;
        auto point = Interaction::from_point(make_float3(1.0f, 2.0f, 3.0f));
        auto uv = Interaction::from_uv(make_float2(0.25f, 0.75f));
        auto surface = Interaction::from_surface(
            Shape::Handle::decode(make_uint4(0u)), make_float3(0.0f),
            make_float3(0.0f, 1.0f, 0.0f), make_float2(0.2f, 0.4f),
            make_float3(0.0f), Frame::make(make_float3(0.0f, 1.0f, 0.0f)),
            7u, 11u, 2.0f, true);

        auto point_ray = point.spawn_ray(make_float3(0.0f, 1.0f, 0.0f), 7.0f);
        auto point_to = point.spawn_ray_to(make_float3(4.0f, 6.0f, 3.0f));

        output.write(0u, make_float4(empty.p_g, empty.prim_area));
        output.write(1u, make_float4(empty.n_g, ite(empty.front_face, 1.0f, 0.0f)));
        output.write(2u, make_float4(empty.p_s, ite(empty.is_surface_interaction(), 1.0f, 0.0f)));
        output.write(3u, make_float4(empty.uv, empty.shading.n().xy()));
        output.write(4u, make_float4(empty.shading.n().z, empty.shape.shadow_terminator_factor(),
                                     empty.shape.intersection_offset_factor(), 0.0f));
        output.write(5u, make_float4(point.p_g, ite(point.is_surface_interaction(), 1.0f, 0.0f)));
        output.write(6u, make_float4(point.p_s, length(point.n_g)));
        output.write(7u, make_float4(uv.uv, length(uv.p_g), ite(uv.is_surface_interaction(), 1.0f, 0.0f)));
        output.write(8u, make_float4(surface.p_g, ite(surface.is_surface_interaction(), 1.0f, 0.0f)));
        output.write(9u, make_float4(surface.uv, surface.prim_area, ite(surface.front_face, 1.0f, 0.0f)));
        output.write(10u, make_float4(surface.p_robust(make_float3(0.0f, 1.0f, 0.0f)), 0.0f));
        output.write(11u, make_float4(surface.p_robust(make_float3(0.0f, -1.0f, 0.0f)), 0.0f));
        output.write(12u, make_float4(point_ray->origin(), point_ray->t_max()));
        output.write(13u, make_float4(point_to->origin(), length(point_to->direction())));
        state.write(0u, make_uint4(empty.inst_id, empty.prim_id,
                                   cast<uint>(empty.is_surface_interaction()), empty.shape.property_flags()));
        state.write(1u, make_uint4(surface.inst_id, surface.prim_id,
                                   cast<uint>(surface.is_surface_interaction()), 0u));
    };

    auto shader = device.compile(kernel);
    auto stream = device.create_stream();
    std::array<float4, 14u> result{};
    std::array<uint4, 2u> state{};
    stream << shader(values, states).dispatch(1u)
           << values.copy_to(result.data())
           << states.copy_to(state.data())
           << synchronize();

    auto valid = true;
    for (auto value : result) { valid = valid && finite(value); }
    valid = valid && state[0u].x == ~0u && state[0u].y == ~0u &&
            state[0u].z == 0u && state[0u].w == 0u;
    valid = valid && state[1u].x == 7u && state[1u].y == 11u && state[1u].z == 1u;
    valid = valid && near(result[0u].x, 0.0f) && near(result[0u].w, 0.0f) &&
            near(result[1u].w, 0.0f) && near(result[2u].w, 0.0f) &&
            near(result[3u].x, 0.0f) && near(result[3u].y, 0.0f) &&
            near(result[3u].z, 0.0f) && near(result[3u].w, 1.0f) &&
            near(result[4u].x, 0.0f);
    valid = valid && near(result[5u].x, 1.0f) && near(result[5u].y, 2.0f) &&
            near(result[5u].z, 3.0f) && near(result[5u].w, 0.0f) &&
            near(result[6u].x, 1.0f) && near(result[6u].y, 2.0f) &&
            near(result[6u].z, 3.0f) && near(result[6u].w, 0.0f);
    valid = valid && near(result[7u].x, 0.25f) && near(result[7u].y, 0.75f) &&
            near(result[7u].z, 0.0f) && near(result[7u].w, 0.0f);
    valid = valid && near(result[8u].w, 1.0f) && near(result[9u].x, 0.2f) &&
            near(result[9u].y, 0.4f) && near(result[9u].z, 2.0f) && near(result[9u].w, 1.0f);
    valid = valid && result[10u].y > 0.0f && result[11u].y < 0.0f;
    valid = valid && near(result[12u].x, 1.0f) && near(result[12u].y, 2.0f) &&
            near(result[12u].z, 3.0f) && near(result[12u].w, 7.0f);
    valid = valid && near(result[13u].x, 1.0f) && near(result[13u].y, 2.0f) &&
            near(result[13u].z, 3.0f) && near(result[13u].w, 1.0f);

    if (!valid)
    {
        std::fprintf(stderr, "interaction test failed\n");
        for (auto i = 0u; i < result.size(); i++)
        {
            auto v = result[i];
            std::fprintf(stderr, "  value[%u] = (%g, %g, %g, %g)\n", i, v.x, v.y, v.z, v.w);
        }
        for (auto i = 0u; i < state.size(); i++)
        {
            auto v = state[i];
            std::fprintf(stderr, "  state[%u] = (%u, %u, %u, %u)\n", i, v.x, v.y, v.z, v.w);
        }
    }
    return valid ? 0 : 1;
}
