#include "ut/ut.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "cameras/pinhole.h"
#include "samplers/zsobol.h"
#include "shapes/inline_mesh.h"
#include "shapes/sphere.h"
#include "textures/constant.h"
#include "textures/image.h"
#include "usd/usd_importer.h"

using namespace Yutrel;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace
{

[[nodiscard]] bool is_near(float a, float b, float epsilon = 1e-4f) noexcept
{
    return std::abs(a - b) < epsilon;
}

[[nodiscard]] bool is_equal(luisa::uint3 a, luisa::uint3 b) noexcept
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

[[nodiscard]] bool is_near(luisa::float2 a, luisa::float2 b) noexcept
{
    return is_near(a.x, b.x) && is_near(a.y, b.y);
}

[[nodiscard]] bool is_near(luisa::float3 a, luisa::float3 b) noexcept
{
    return is_near(a.x, b.x) && is_near(a.y, b.y) && is_near(a.z, b.z);
}

static auto test_usd_import_registration = []
{
    "import_blender_basic_scene"_test = []
    {
        auto spec = UsdImporter::import("scene/blender-basic/scene.usdc");

        expect(spec.instances().size() == 2u);
        expect(spec.shapes().size() == 2u);
        expect(spec.surfaces().size() == 2u);
        expect(spec.textures().size() == 5u);
        expect(spec.lights().size() == 1u);
        expect(spec.environments().size() == 1u);
        expect(spec.cameras().size() == 1u);

        auto light_instance_count = 0u;
        for (auto&& instance : spec.instances())
        {
            if (instance.light)
            {
                light_instance_count++;
                auto&& sphere = static_cast<const SphereShapeSpec&>(
                    spec.shapes().spec(instance.shape));
                expect(is_near(sphere.radius(), 0.1f));
                expect(is_near(instance.transform[3].x, 4.0762453f));
                expect(is_near(instance.transform[3].y, 1.005454f));
                expect(is_near(instance.transform[3].z, 5.903862f));
            }
        }
        expect(light_instance_count == 1u);

        auto&& camera = static_cast<const PinholeCameraSpec&>(
            spec.cameras().spec(spec.render().camera));
        expect(is_near(camera.fov(), 22.8952f));

        auto&& sampler = static_cast<const ZSobolSamplerSpec&>(
            spec.samplers().spec(spec.render().sampler));
        expect(sampler.spp() == 16u);
        expect(sampler.seed() == 20120712u);

        auto found_emission = false;
        spec.textures().visit_entries(
            [&](TextureRef, const SpecMeta& meta, const TextureSpec* texture)
        {
            if (meta.name == "/root/Light/Light::emission")
            {
                auto&& emission = static_cast<const ConstantTextureSpec&>(*texture);
                expect(is_near(emission.value().x, 2533.0295f, 1e-2f));
                found_emission = true;
            }
        });
        expect(found_emission);
    };

    "apply_cli_overrides"_test = []
    {
        auto spec = UsdImporter::import(
            "scene/blender-basic/scene.usdc",
            UsdImportOptions{
                .spp        = 4u,
                .seed       = 42u,
                .resolution = luisa::make_uint2(320u, 180u),
                .output     = std::filesystem::path{"build/usd-test.exr"},
            });
        auto&& sampler = static_cast<const ZSobolSamplerSpec&>(
            spec.samplers().spec(spec.render().sampler));
        expect(sampler.spp() == 4u);
        expect(sampler.seed() == 42u);
    };

    "triangulate_left_handed_face_varying_mesh"_test = []
    {
        auto spec = UsdImporter::import("test/usd/scenes/face_varying_quad.usda");
        expect(spec.instances().size() == 1u);
        auto&& mesh = static_cast<const InlineMeshShapeSpec&>(
            spec.shapes().spec(spec.instances()[0u].shape));
        expect(mesh.positions().size() == 4u);
        expect(mesh.normals().size() == 4u);
        expect(mesh.uvs().size() == 4u);
        expect(mesh.indices().size() == 2u);
        expect(is_equal(mesh.indices()[0u], luisa::make_uint3(0u, 2u, 1u)));
        expect(is_equal(mesh.indices()[1u], luisa::make_uint3(0u, 3u, 2u)));
        expect(is_near(mesh.uvs()[0u], luisa::make_float2(0.0f, 1.0f)));
        expect(is_near(mesh.uvs()[3u], luisa::make_float2(0.0f, 0.0f)));
        for (auto normal : mesh.normals())
        {
            expect(is_near(normal, luisa::make_float3(0.0f, 0.0f, 1.0f)));
        }
    };

    "import_relative_linear_texture"_test = []
    {
        auto spec  = UsdImporter::import("test/usd/scenes/textured_material.usda");
        auto found = false;
        spec.textures().visit_entries(
            [&](TextureRef, const SpecMeta& meta, const TextureSpec* texture)
        {
            if (meta.name == "/Material/Texture")
            {
                auto&& image = static_cast<const ImageTextureSpec&>(*texture);
                expect(image.path().filename() == "color_0C0C0C.exr");
                expect(image.encoding() == Texture::Encoding::LINEAR);
                found = true;
            }
        });
        expect(found);
    };

    "reject_unsupported_subdivision"_test = []
    {
        auto rejected = false;
        try
        {
            (void)UsdImporter::import("test/usd/scenes/unsupported_subdivision.usda");
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            rejected     = message.find("unsupported_subdivision.usda") != std::string::npos &&
                           message.find("/Mesh") != std::string::npos;
        }
        expect(rejected);
    };

    "reject_metallic_material"_test = []
    {
        auto rejected = false;
        try
        {
            (void)UsdImporter::import("test/usd/scenes/unsupported_metallic.usda");
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            rejected     = message.find("unsupported_metallic.usda") != std::string::npos &&
                           message.find("/Material/Preview") != std::string::npos;
        }
        expect(rejected);
    };

    "reject_missing_camera"_test = []
    {
        auto rejected = false;
        try
        {
            (void)UsdImporter::import("test/usd/scenes/missing_camera.usda");
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            rejected     = message.find("missing_camera.usda") != std::string::npos &&
                           message.find("exactly one visible perspective camera") != std::string::npos;
        }
        expect(rejected);
    };

    "reject_missing_file"_test = []
    {
        auto rejected = false;
        try
        {
            (void)UsdImporter::import("test/usd/scenes/missing.usdc");
        }
        catch (const std::runtime_error& error)
        {
            rejected = std::string{error.what()}.find("missing.usdc") != std::string::npos;
        }
        expect(rejected);
    };
    return 0;
}();

} // namespace

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
