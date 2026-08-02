#include <cmath>
#include <iostream>

#include <luisa/luisa-compute.h>

#include "base/film.h"
#include "base/renderer.h"
#include "base/scene.h"
#include "cameras/pinhole.h"
#include "environments/distant.h"
#include "filters/box.h"
#include "integrators/path.h"
#include "samplers/independent.h"
#include "scene/scene_spec_builder.h"
#include "shapes/inline_mesh.h"
#include "spectrum/srgb.h"
#include "surfaces/diffuse.h"
#include "textures/constant.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{

[[nodiscard]] luisa::unique_ptr<Scene> create_scene(uint2 resolution)
{
    SceneSpecBuilder builder;
    SourceLocation source{.file = "external-render-test"};
    auto albedo = builder.add_anonymous_texture<ConstantTextureSpec>(
        source, make_float4(0.5f, 0.5f, 0.5f, 1.0f));
    auto emission = builder.add_anonymous_texture<ConstantTextureSpec>(
        source, make_float4(1.0f));
    auto surface = builder.add_anonymous_surface<DiffuseSurfaceSpec>(source, albedo, true);
    auto shape = builder.add_anonymous_shape<InlineMeshShapeSpec>(
        source,
        luisa::vector{make_float3(-1.0f, -1.0f, -3.0f), make_float3(1.0f, -1.0f, -3.0f), make_float3(0.0f, 1.0f, -3.0f)},
        luisa::vector(3u, make_float3(0.0f, 0.0f, 1.0f)),
        luisa::vector<float2>{},
        luisa::vector{make_uint3(0u, 1u, 2u)});
    auto spectrum = builder.add_anonymous_spectrum<SRGBSpectrumSpec>(source);
    auto environment = builder.add_anonymous_environment<DistantEnvironmentSpec>(
        source, emission, 1.0f, make_float3(0.0f, 0.0f, 1.0f));
    auto camera = builder.add_anonymous_camera<PinholeCameraSpec>(
        source,
        make_float4x4(1.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float2(0.0f),
        0u,
        45.0f);
    auto film = builder.add_anonymous_film<RGBFilmSpec>(source, resolution, false, "external-render-test.exr");
    auto filter = builder.add_anonymous_filter<BoxFilterSpec>(source, 0.5f);
    auto sampler = builder.add_anonymous_sampler<IndependentSamplerSpec>(source, 1u, 0u);
    auto integrator = builder.add_anonymous_integrator<PathIntegratorSpec>(source, 4u);
    builder.add_instance(ShapeInstanceSpec{
        .source = source,
        .shape = shape,
        .surface = surface,
    });
    builder.set_render(RenderSpec{
        .spectrum = spectrum,
        .environment = environment,
        .camera = camera,
        .film = film,
        .filter = filter,
        .sampler = sampler,
        .integrator = integrator,
    });
    auto spec = builder.finish();
    return Scene::create(spec);
}

[[nodiscard]] bool render_and_check(
    Device& device,
    Stream& stream,
    Renderer& renderer,
    uint2 resolution,
    uint sample_count)
{
    auto accumulation = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    Kernel2D clear = [](ImageFloat image) noexcept
    {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };
    auto clear_shader = device.compile(clear);
    CommandBuffer commands{stream};
    commands << clear_shader(accumulation).dispatch(resolution);
    for (auto sample = 0u; sample < sample_count; sample++)
    {
        if (!renderer.render_external_sample(commands, accumulation, resolution, sample))
        {
            return false;
        }
    }
    luisa::vector<float4> pixels(static_cast<size_t>(resolution.x) * resolution.y);
    commands << accumulation.copy_to(luisa::span{pixels}) << synchronize();

    auto has_radiance = false;
    for (auto pixel : pixels)
    {
        if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
            !std::isfinite(pixel.z) || !std::isfinite(pixel.w) ||
            pixel.w != static_cast<float>(sample_count))
        {
            return false;
        }
        has_radiance |= pixel.x > 0.0f || pixel.y > 0.0f || pixel.z > 0.0f;
    }
    return has_radiance;
}

}// namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        return 0;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();
    auto resolution = make_uint2(16u, 16u);
    auto scene = create_scene(resolution);
    auto renderer = Renderer::create(device, stream, *scene);
    if (!renderer || !renderer->prepare_external_render())
    {
        return 1;
    }
    ExternalCameraState camera{
        .camera_to_world = make_float4x4(1.0f),
        .resolution = resolution,
        .vertical_fov_degrees = 45.0f,
    };
    CommandBuffer commands{stream};
    if (!renderer->update_external_camera(commands, camera))
    {
        return 1;
    }
    commands << synchronize();
    if (!render_and_check(device, stream, *renderer, resolution, 2u))
    {
        std::cerr << "External accumulation test failed.\n";
        return 1;
    }

    camera.resolution = make_uint2(8u, 8u);
    camera.vertical_fov_degrees = 60.0f;
    CommandBuffer resize_commands{stream};
    if (!renderer->update_external_camera(resize_commands, camera))
    {
        return 1;
    }
    resize_commands << synchronize();
    if (!render_and_check(device, stream, *renderer, camera.resolution, 1u))
    {
        std::cerr << "External resize test failed.\n";
        return 1;
    }
    return 0;
}
