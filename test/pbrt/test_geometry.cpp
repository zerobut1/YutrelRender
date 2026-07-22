#include <array>

#include <luisa/luisa-compute.h>

#include "base/geometry.h"
#include "base/interaction.h"
#include "base/renderer.h"
#include "base/scene.h"
#include "pbrt/pbrt_importer.h"
#include "pbrt/pbrt_parser.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        return 0;
    }

    auto parsed   = PbrtParser::parse("test/scenes/alpha_visibility.pbrt");
    auto spec     = PbrtImporter::import(std::move(parsed));
    auto scene    = Scene::create(spec);
    Context context{argv[0]};
    auto device   = context.create_device(argv[1]);
    auto stream   = device.create_stream();
    auto renderer = Renderer::create(device, stream, *scene);
    auto output   = device.create_buffer<uint4>(1u);

    Kernel1D kernel = [&renderer](BufferUInt4 result) noexcept
    {
        auto center_ray      = make_ray(make_float3(0.0f), make_float3(0.0f, 0.0f, -1.0f));
        auto transparent_ray = make_ray(make_float3(2.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, -1.0f));
        auto center_hit      = renderer->geometry()->intersect(center_ray);
        auto transparent_hit = renderer->geometry()->intersect(transparent_ray);
        result.write(0u, make_uint4(
                             center_hit->inst_id,
                             ite(center_hit->is_surface_interaction(), 1u, 0u),
                             ite(renderer->geometry()->intersect_any(center_ray), 1u, 0u),
                             ite(renderer->geometry()->intersect_any(transparent_ray), 1u, 0u)));
    };

    auto shader = device.compile(kernel);
    std::array<uint4, 1u> result{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(result.data())
           << synchronize();
    auto value = result.front();
    return value.x == 1u && value.y == 1u && value.z == 1u && value.w == 0u ? 0 : 1;
}
