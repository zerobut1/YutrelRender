#include <array>
#include <cmath>

#include <luisa/luisa-compute.h>

#include "base/geometry.h"
#include "base/interaction.h"
#include "base/renderer.h"
#include "base/scene.h"
#include "pbrt/pbrt_importer.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

int main(int argc, char* argv[])
{
    if (argc < 2) { return 0; }

    auto spec     = PbrtImporter::import("test/scenes/point_basic.pbrt");
    auto scene    = Scene::create(spec);
    Context context{argv[0]};
    auto device   = context.create_device(argv[1]);
    auto stream   = device.create_stream();
    auto renderer = Renderer::create(device, stream, *scene);
    if (renderer->light_handles().size() != 2u) { return 1; }
    auto handle = Light::Handle{};
    auto found_point = false;
    for (auto candidate : renderer->light_handles())
    {
        if (candidate.instance_id == ~0u)
        {
            handle = candidate;
            found_point = true;
        }
    }
    if (!found_point) { return 1; }
    auto output = device.create_buffer<float4>(3u);

    Kernel1D kernel = [&renderer, handle](BufferFloat4 result) noexcept
    {
        auto swl = renderer->spectrum()->sample(0.5f);
        auto s1  = Light::Sample::zero(swl.dimension());
        auto s2  = Light::Sample::zero(swl.dimension());
        auto le  = Light::Closure::EmissionSample::zero(swl.dimension());
        renderer->lights().dispatch(handle.light_tag, [&](auto light) noexcept
        {
            auto closure = light->closure(swl, 0.0f);
            s1 = closure->sample_li(
                handle.instance_id,
                Interaction::from_point(make_float3(3.0f, 2.0f, 2.0f)),
                make_float2(0.25f));
            s2 = closure->sample_li(
                handle.instance_id,
                Interaction::from_point(make_float3(4.0f, 2.0f, 2.0f)),
                make_float2(0.75f));
            le = closure->sample_le(
                handle.instance_id, make_float2(0.5f), make_float2(0.25f, 0.75f));
        });

        auto ray_to_light = make_ray(
            make_float3(2.0f, 2.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
        result.write(0u, make_float4(s1.eval.L[0u], s2.eval.L[0u], s1.eval.pdf,
                                     ite(s1.delta, 1.0f, 0.0f)));
        auto emitted_flux_ratio = le.Le[0u] / max(le.pdf * s1.eval.L[0u], 1e-8f);
        result.write(1u, make_float4(le.pdf, length(le.ray->direction()), le.cos_theta,
                                     emitted_flux_ratio));
        result.write(2u, make_float4(
                             ite(renderer->geometry()->intersect_any(ray_to_light), 1.0f, 0.0f)));
    };

    auto shader = device.compile(kernel);
    std::array<float4, 3u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(result.data())
           << synchronize();

    auto direct = result[0u];
    auto emitted = result[1u];
    auto near = [](float a, float b, float eps = 1e-4f) noexcept
    {
        return std::abs(a - b) <= eps;
    };
    auto inv_four_pi = 0.25f / std::acos(-1.0f);
    auto valid = direct.y > 0.0f && near(direct.x / direct.y, 4.0f, 1e-3f) &&
                 near(direct.z, 1.0f) && near(direct.w, 1.0f) &&
                 near(emitted.x, inv_four_pi) && near(emitted.y, 1.0f) &&
                 near(emitted.z, 1.0f) && near(emitted.w, 4.0f * std::acos(-1.0f), 1e-3f) &&
                 near(result[2u].x, 0.0f);
    return valid ? 0 : 1;
}
