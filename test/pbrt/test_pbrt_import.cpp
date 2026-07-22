#include "ut/ut.hpp"

#include "base/film.h"
#include "base/scene.h"
#include "base/shape.h"
#include "cameras/pinhole.h"
#include "environments/distant.h"
#include "environments/grouped.h"
#include "environments/uniform.h"
#include "filters/gaussian.h"
#include "integrators/vol_path.h"
#include "media/homogeneous.h"
#include "pbrt/pbrt_importer.h"
#include "pbrt/pbrt_parser.h"
#include "samplers/independent.h"
#include "samplers/sobol.h"
#include "samplers/zsobol.h"
#include "shapes/mesh.h"
#include "shapes/sphere.h"
#include "surfaces/coated_diffuse.h"
#include "surfaces/diffuse.h"
#include "surfaces/null.h"
#include "surfaces/opacity.h"
#include "textures/checker_board.h"
#include "textures/constant.h"
#include "textures/image.h"
#include "textures/scale.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#include <luisa/core/logging.h>

using namespace Yutrel;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace
{

[[nodiscard]] bool is_near(float a, float b, float epsilon = 1e-4f) noexcept
{
    return std::abs(a - b) < epsilon;
}

[[nodiscard]] float column_length(const luisa::float4x4& m, uint column) noexcept
{
    auto v = m[column];
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

[[nodiscard]] bool matrix_is_near(
    const luisa::float4x4& a,
    const luisa::float4x4& b,
    float epsilon = 1e-4f) noexcept
{
    for (auto column = 0u; column < 4u; column++)
    {
        for (auto row = 0u; row < 4u; row++)
        {
            if (!is_near(a[column][row], b[column][row], epsilon))
            {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] RawParameter test_parameter(
    luisa::string type, luisa::string name, uint line)
{
    return RawParameter{
        .source = SourceLocation{"strict_case.pbrt", line, 1u},
        .type   = std::move(type),
        .name   = std::move(name),
    };
}

static auto test_pbrt_import_registration = []
{
    "import_path_applies_overrides"_test = []
    {
        auto output = std::filesystem::absolute("override.exr");
        auto spec   = PbrtImporter::import(
            "test/scenes/import_geometry.pbrt",
            PbrtImportOptions{
                .spp        = 16u,
                .seed       = 42u,
                .resolution = make_uint2(320u, 200u),
                .output     = output,
            });
        auto scene = Scene::create(spec);
        expect(scene->sampler()->spp() == 16u);
        expect(scene->sampler()->seed() == 42u);
        expect(luisa::all(scene->film()->resolution() == make_uint2(320u, 200u)));
        expect(scene->film()->filename() == output);
    };

    "validate_rgb_film_spec"_test = []
    {
        expect(!RGBFilmSpec{make_uint2(16u), false, "render.exr"}.validate().has_value());
        expect(RGBFilmSpec{make_uint2(16u), false, "render.exr", 0.0f}.validate().has_value());
        expect(RGBFilmSpec{make_uint2(16u), false, "render.exr", -1.0f}.validate().has_value());
        expect(RGBFilmSpec{make_uint2(16u), false, "render.exr", std::numeric_limits<float>::infinity()}.validate().has_value());
        expect(RGBFilmSpec{make_uint2(16u), false, "render.exr", std::numeric_limits<float>::quiet_NaN()}.validate().has_value());
    };

    "import_film_exposure"_test = []
    {
        auto parsed                 = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.film.iso             = 200.0f;
        parsed.camera.shutter_open  = 0.25f;
        parsed.camera.shutter_close = 0.75f;
        auto spec                   = PbrtImporter::import(std::move(parsed));

        auto film   = dynamic_cast<const RGBFilmSpec*>(&spec.films().spec(spec.render().film));
        auto camera = dynamic_cast<const PinholeCameraSpec*>(&spec.cameras().spec(spec.render().camera));
        expect(film != nullptr);
        expect(camera != nullptr);
        if (film != nullptr && camera != nullptr)
        {
            expect(is_near(film->imaging_ratio(), 1.0f));
            expect(is_near(camera->shutter_span().x, 0.25f));
            expect(is_near(camera->shutter_span().y, 0.75f));
        }

        auto default_parsed = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        auto default_spec   = PbrtImporter::import(std::move(default_parsed));
        auto default_film   = dynamic_cast<const RGBFilmSpec*>(&default_spec.films().spec(default_spec.render().film));
        auto default_camera = dynamic_cast<const PinholeCameraSpec*>(&default_spec.cameras().spec(default_spec.render().camera));
        expect(default_film != nullptr);
        expect(default_camera != nullptr);
        if (default_film != nullptr && default_camera != nullptr)
        {
            expect(is_near(default_film->imaging_ratio(), 1.0f));
            expect(is_near(default_camera->shutter_span().x, 0.0f));
            expect(is_near(default_camera->shutter_span().y, 1.0f));
        }
    };

    "import_camera_preserves_full_affine_transform"_test = []
    {
        struct Case
        {
            Matrix4 camera_from_world;
            float4x4 expected_camera_to_world;
        };
        constexpr Matrix4 identity{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};
        constexpr Matrix4 look_at_negative_z{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};
        constexpr Matrix4 single_negative_scale{
            -1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};
        constexpr Matrix4 double_negative_scale{
            -1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, -1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};
        constexpr Matrix4 affine_inverse{
            0.5f, -1.0f / 6.0f, 1.0f / 24.0f, -17.0f / 12.0f,
            0.0f, 1.0f / 3.0f, -1.0f / 12.0f, -7.0f / 6.0f,
            0.0f, 0.0f, 0.25f, -1.5f,
            0.0f, 0.0f, 0.0f, 1.0f};
        std::array cases{
            Case{
                identity,
                make_float4x4(
                    make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                    make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                    make_float4(0.0f, 0.0f, -1.0f, 0.0f),
                    make_float4(0.0f, 0.0f, 0.0f, 1.0f))},
            Case{look_at_negative_z, make_float4x4(1.0f)},
            Case{
                single_negative_scale,
                make_float4x4(
                    make_float4(-1.0f, 0.0f, 0.0f, 0.0f),
                    make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                    make_float4(0.0f, 0.0f, -1.0f, 0.0f),
                    make_float4(0.0f, 0.0f, 0.0f, 1.0f))},
            Case{
                double_negative_scale,
                make_float4x4(
                    make_float4(-1.0f, 0.0f, 0.0f, 0.0f),
                    make_float4(0.0f, -1.0f, 0.0f, 0.0f),
                    make_float4(0.0f, 0.0f, -1.0f, 0.0f),
                    make_float4(0.0f, 0.0f, 0.0f, 1.0f))},
            Case{
                affine_inverse,
                make_float4x4(
                    make_float4(2.0f, 0.0f, 0.0f, 0.0f),
                    make_float4(1.0f, 3.0f, 0.0f, 0.0f),
                    make_float4(0.0f, -1.0f, -4.0f, 0.0f),
                    make_float4(4.0f, 5.0f, 6.0f, 1.0f))},
        };

        for (auto&& test_case : cases)
        {
            auto parsed                  = PbrtParser::parse("test/scenes/import_geometry.pbrt");
            parsed.camera.pbrt_transform = test_case.camera_from_world;
            auto spec                    = PbrtImporter::import(std::move(parsed));
            auto camera                  = dynamic_cast<const PinholeCameraSpec*>(
                &spec.cameras().spec(spec.render().camera));
            expect(camera != nullptr);
            if (camera != nullptr)
            {
                expect(matrix_is_near(camera->camera_to_world(), test_case.expected_camera_to_world));
                auto expected_determinant = camera_linear_determinant(test_case.expected_camera_to_world);
                auto actual_determinant   = camera_linear_determinant(camera->camera_to_world());
                expect((expected_determinant < 0.0f) == (actual_determinant < 0.0f));
            }
        }
    };

    "import_camera_converts_pbrt_short_axis_fov_to_vertical"_test = []
    {
        auto parsed             = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.camera.fov       = 30.0f;
        parsed.film.resolution  = make_uint2(1280u, 1800u);
        auto portrait_spec      = PbrtImporter::import(parsed);
        auto portrait_camera    = dynamic_cast<const PinholeCameraSpec*>(
            &portrait_spec.cameras().spec(portrait_spec.render().camera));
        expect(portrait_camera != nullptr);
        if (portrait_camera != nullptr)
        {
            auto expected = 2.0f * std::atan(
                std::tan(0.5f * 30.0f * 0.01745329251994329577f) *
                (1800.0f / 1280.0f)) *
                            57.295779513082320876f;
            expect(is_near(portrait_camera->fov(), expected));
        }

        parsed.film.resolution = make_uint2(1800u, 1280u);
        auto landscape_spec    = PbrtImporter::import(std::move(parsed));
        auto landscape_camera  = dynamic_cast<const PinholeCameraSpec*>(
            &landscape_spec.cameras().spec(landscape_spec.render().camera));
        expect(landscape_camera != nullptr);
        if (landscape_camera != nullptr)
        {
            expect(is_near(landscape_camera->fov(), 30.0f));
        }
    };

    "reject_invalid_camera_transform_with_source"_test = []
    {
        auto singular = identity_matrix4;
        singular[0u]  = 0.0f;
        auto non_finite = identity_matrix4;
        non_finite[5u]  = std::numeric_limits<float>::infinity();
        auto non_affine = identity_matrix4;
        non_affine[12u] = 0.25f;
        struct Case
        {
            Matrix4 transform;
            const char* expected;
        };
        std::array cases{
            Case{singular, "singular"},
            Case{non_finite, "finite"},
            Case{non_affine, "affine"},
        };
        for (auto&& test_case : cases)
        {
            auto parsed                  = PbrtParser::parse("test/scenes/import_geometry.pbrt");
            parsed.camera.source         = SourceLocation{"bad-camera.pbrt", 77u, 3u};
            parsed.camera.pbrt_transform = test_case.transform;
            auto rejected                = false;
            try
            {
                (void)PbrtImporter::import(std::move(parsed));
            }
            catch (const std::runtime_error& error)
            {
                auto message = std::string{error.what()};
                rejected = message.find("bad-camera.pbrt:77:3") != std::string::npos &&
                           message.find(test_case.expected) != std::string::npos;
            }
            expect(rejected);
        }
    };

    "import_sobol_sampler"_test = []
    {
        auto parsed                  = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.sampler.type          = SamplerDesc::Type::Sobol;
        parsed.sampler.pixel_samples = 32u;
        parsed.sampler.seed          = 42u;
        auto spec                    = PbrtImporter::import(std::move(parsed));
        auto&& sampler               = static_cast<const SobolSamplerSpec&>(
            spec.samplers().spec(spec.render().sampler));
        expect(sampler.spp() == 32u);
        expect(sampler.seed() == 42u);
        expect(!sampler.validate().has_value());
    };

    "import_zsobol_sampler"_test = []
    {
        auto parsed                  = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.sampler.type          = SamplerDesc::Type::ZSobol;
        parsed.sampler.pixel_samples = 32u;
        parsed.sampler.seed          = 42u;
        auto spec                    = PbrtImporter::import(std::move(parsed));
        auto&& sampler               = static_cast<const ZSobolSamplerSpec&>(
            spec.samplers().spec(spec.render().sampler));
        expect(sampler.spp() == 32u);
        expect(sampler.seed() == 42u);
        expect(!sampler.validate().has_value());
        expect(ZSobolSamplerSpec{3u, 42u}.validate().has_value());
        expect(ZSobolSamplerSpec{0u, 42u}.validate().has_value());
    };

    "reject_invalid_zsobol_override"_test = []
    {
        auto rejected = false;
        try
        {
            (void)PbrtImporter::import(
                "test/scenes/zsobol_sampler_default.pbrt",
                PbrtImportOptions{.spp = 3u});
        }
        catch (const std::runtime_error& error)
        {
            rejected = std::string{error.what()}.find("power of two") != std::string::npos;
        }
        expect(rejected);
    };

    "import_independent_sampler_seed"_test = []
    {
        auto parsed    = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        auto spec      = PbrtImporter::import(std::move(parsed));
        auto&& sampler = static_cast<const IndependentSamplerSpec&>(
            spec.samplers().spec(spec.render().sampler));
        expect(sampler.spp() == 1u);
        expect(sampler.seed() == 37u);
    };

    "accept_zero_path_maxdepth"_test = []
    {
        auto parsed                 = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.integrator.max_depth = 0u;
        auto spec                   = PbrtImporter::import(std::move(parsed));
        auto&& integrator           = spec.integrators().spec(spec.render().integrator);
        expect(!integrator.validate().has_value());
    };

    "import_homogeneous_medium_interfaces_and_volpath"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/homogeneous_medium.pbrt");
        auto spec   = PbrtImporter::import(std::move(parsed));
        expect(spec.media().size() == 1u);
        expect(spec.instances().size() == 2u);
        expect(spec.instances()[0u].inside_medium.has_value());
        expect(!spec.instances()[0u].outside_medium.has_value());
        expect(!spec.instances()[1u].inside_medium.has_value());
        expect(spec.instances()[1u].outside_medium.has_value());
        if (spec.instances()[0u].inside_medium && spec.instances()[1u].outside_medium)
        {
            expect(*spec.instances()[0u].inside_medium == *spec.instances()[1u].outside_medium);
        }
        auto&& medium = static_cast<const HomogeneousMediumSpec&>(
            spec.media().spec(*spec.instances()[0u].inside_medium));
        expect(!medium.validate().has_value());
        auto&& integrator = static_cast<const VolPathIntegratorSpec&>(
            spec.integrators().spec(spec.render().integrator));
        expect(!integrator.validate().has_value());

        auto undefined = PbrtParser::parse("test/scenes/homogeneous_medium_undefined.pbrt");
        auto rejected  = false;
        try
        {
            (void)PbrtImporter::import(std::move(undefined));
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            rejected     = message.find("homogeneous_medium_undefined.pbrt") != std::string::npos &&
                           message.find("missing") != std::string::npos;
        }
        expect(rejected);
    };

    "pbrt_v4_defaults_are_supported"_test = []
    {
        auto defaults = PbrtParser::parse("test/scenes/pbrt_defaults.pbrt");
        expect(defaults.integrator.type == IntegratorDesc::Type::VolPath);
        expect(defaults.integrator.max_depth == 5u);
        expect(defaults.sampler.type == SamplerDesc::Type::ZSobol);
        expect(defaults.sampler.pixel_samples == 16u);
        expect(defaults.sampler.seed == 20120712u);
        expect(defaults.filter.type == FilterDesc::Type::Gaussian);
        expect(is_near(defaults.filter.radius.x, 1.5f));
        expect(is_near(defaults.filter.sigma, 0.5f));
        expect(defaults.film.resolution.x == 1280u);
        expect(defaults.film.resolution.y == 720u);
        expect(defaults.film.filename == std::filesystem::path{"pbrt.exr"});
        expect(is_near(defaults.camera.fov, 90.0f));
        expect(is_near(defaults.materials.front().reflectance.x, 0.5f));

        auto volpath_spec               = PbrtImporter::import(defaults);
        auto&& volpath                  = static_cast<const VolPathIntegratorSpec&>(
            volpath_spec.integrators().spec(volpath_spec.render().integrator));
        expect(!volpath.validate().has_value());
        auto&& default_sampler          = static_cast<const ZSobolSamplerSpec&>(
            volpath_spec.samplers().spec(volpath_spec.render().sampler));
        expect(default_sampler.spp() == 16u);
        expect(default_sampler.seed() == 20120712u);

        auto zsobol            = PbrtParser::parse("test/scenes/pbrt_defaults.pbrt");
        zsobol.integrator.type = IntegratorDesc::Type::Path;
        auto path_spec         = PbrtImporter::import(std::move(zsobol));
        auto&& path_sampler    = static_cast<const ZSobolSamplerSpec&>(
            path_spec.samplers().spec(path_spec.render().sampler));
        expect(!path_sampler.validate().has_value());

        auto gaussian                  = PbrtParser::parse("test/scenes/pbrt_defaults.pbrt");
        gaussian.integrator.type       = IntegratorDesc::Type::Path;
        gaussian.sampler.type          = SamplerDesc::Type::Independent;
        gaussian.sampler.pixel_samples = 4u;
        auto gaussian_spec             = PbrtImporter::import(std::move(gaussian));
        auto&& gaussian_filter          = static_cast<const GaussianFilterSpec&>(
            gaussian_spec.filters().spec(gaussian_spec.render().filter));
        expect(!gaussian_filter.validate().has_value());
    };

    "reject_invalid_gaussian_filter"_test = []
    {
        for (auto sigma : {
                 0.0f,
                 -0.5f,
                 std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN(),
             })
        {
            auto scene                  = PbrtParser::parse("test/scenes/pbrt_defaults.pbrt");
            scene.integrator.type       = IntegratorDesc::Type::Path;
            scene.sampler.type          = SamplerDesc::Type::Independent;
            scene.sampler.pixel_samples = 4u;
            scene.filter.sigma          = sigma;
            auto rejected               = false;
            try
            {
                (void)PbrtImporter::import(std::move(scene));
            }
            catch (const std::runtime_error& error)
            {
                rejected = std::string{error.what()}.find("sigma") != std::string::npos;
            }
            expect(rejected);
        }

        auto wrong_type                  = PbrtParser::parse("test/scenes/pbrt_defaults.pbrt");
        wrong_type.integrator.type       = IntegratorDesc::Type::Path;
        wrong_type.sampler.type          = SamplerDesc::Type::Independent;
        wrong_type.sampler.pixel_samples = 4u;
        wrong_type.filter.parameters.emplace_back(test_parameter("integer", "sigma", 42u));
        auto wrong_type_rejected = false;
        try
        {
            (void)PbrtImporter::import(std::move(wrong_type));
        }
        catch (const std::runtime_error& error)
        {
            auto message        = std::string{error.what()};
            wrong_type_rejected = message.find("sigma") != std::string::npos &&
                                  message.find("expected 'float'") != std::string::npos;
        }
        expect(wrong_type_rejected);

        auto unequal                  = PbrtParser::parse("test/scenes/pbrt_defaults.pbrt");
        unequal.integrator.type       = IntegratorDesc::Type::Path;
        unequal.sampler.type          = SamplerDesc::Type::Independent;
        unequal.sampler.pixel_samples = 4u;
        unequal.filter.radius         = luisa::make_float2(1.0f, 1.5f);
        auto unequal_rejected         = false;
        try
        {
            (void)PbrtImporter::import(std::move(unequal));
        }
        catch (const std::runtime_error& error)
        {
            unequal_rejected = std::string{error.what()}.find("equal x/y radii") != std::string::npos;
        }
        expect(unequal_rejected);
    };

    "reject_halton_sampler"_test = []
    {
        auto parsed            = PbrtParser::parse("test/scenes/sobol_sampler_default.pbrt");
        parsed.integrator.type = IntegratorDesc::Type::Path;
        parsed.sampler.type    = SamplerDesc::Type::Halton;
        parsed.filter.type     = FilterDesc::Type::Triangle;
        auto rejected          = false;
        try
        {
            (void)PbrtImporter::import(std::move(parsed));
        }
        catch (const std::runtime_error& error)
        {
            rejected = std::string{error.what()}.find("halton") != std::string::npos;
        }
        expect(rejected);
    };

    "reject_invalid_pbrt_exposure"_test = []
    {
        struct Case
        {
            float iso;
            float shutter_open;
            float shutter_close;
            const char* expected_message;
        };
        std::array cases{
            Case{0.0f, 0.0f, 1.0f, "ISO"},
            Case{-1.0f, 0.0f, 1.0f, "ISO"},
            Case{std::numeric_limits<float>::infinity(), 0.0f, 1.0f, "ISO"},
            Case{std::numeric_limits<float>::quiet_NaN(), 0.0f, 1.0f, "ISO"},
            Case{100.0f, 1.0f, 1.0f, "shutterclose"},
            Case{100.0f, 1.0f, 0.0f, "shutterclose"},
            Case{std::numeric_limits<float>::max(), 0.0f, 2.0f, "ratio"},
        };
        for (auto test_case : cases)
        {
            auto parsed                 = PbrtParser::parse("test/scenes/import_geometry.pbrt");
            parsed.film.iso             = test_case.iso;
            parsed.camera.shutter_open  = test_case.shutter_open;
            parsed.camera.shutter_close = test_case.shutter_close;
            auto rejected               = false;
            try
            {
                (void)PbrtImporter::import(std::move(parsed));
            }
            catch (const std::runtime_error& error)
            {
                rejected = std::string{error.what()}.find(test_case.expected_message) != std::string::npos;
            }
            expect(rejected);
        }
    };

    "reject_unsupported_parameters_for_every_imported_declaration"_test = []
    {
        enum class Target
        {
            Integrator,
            Sampler,
            Filter,
            Film,
            Camera,
            Texture,
            Material,
            AreaLight,
            Shape,
            InfiniteLight,
        };
        struct Case
        {
            Target target;
            const char* type;
            const char* name;
            uint line;
        };
        std::array cases{
            Case{Target::Integrator, "float", "maxdepth", 11u},
            Case{Target::Sampler, "string", "seed", 12u},
            Case{Target::Filter, "integer", "xradius", 13u},
            Case{Target::Film, "string", "sensor", 14u},
            Case{Target::Camera, "float", "lensradius", 15u},
            Case{Target::Texture, "bool", "invert", 16u},
            Case{Target::Material, "texture", "displacement", 17u},
            Case{Target::AreaLight, "float", "scale", 18u},
            Case{Target::Shape, "float", "displacement", 19u},
            Case{Target::InfiniteLight, "float", "power", 20u},
        };
        for (auto test_case : cases)
        {
            auto scene     = PbrtParser::parse("test/scenes/strict_import_base.pbrt");
            auto parameter = test_parameter(test_case.type, test_case.name, test_case.line);
            switch (test_case.target)
            {
            case Target::Integrator:
                scene.integrator.parameters.emplace_back(std::move(parameter));
                break;
            case Target::Sampler:
                scene.sampler.parameters.emplace_back(std::move(parameter));
                break;
            case Target::Filter:
                scene.filter.parameters.emplace_back(std::move(parameter));
                break;
            case Target::Film:
                scene.film.parameters.emplace_back(std::move(parameter));
                break;
            case Target::Camera:
                scene.camera.parameters.emplace_back(std::move(parameter));
                break;
            case Target::Texture:
                scene.textures.front().parameters.emplace_back(std::move(parameter));
                break;
            case Target::Material:
                scene.named_materials.at("Floor").parameters.emplace_back(std::move(parameter));
                break;
            case Target::AreaLight:
                for (auto& shape : scene.shapes)
                {
                    if (shape.area_light)
                    {
                        shape.area_light->parameters.emplace_back(std::move(parameter));
                        break;
                    }
                }
                break;
            case Target::Shape:
                scene.shapes.front().parameters.emplace_back(std::move(parameter));
                break;
            case Target::InfiniteLight:
                scene.infinite_lights.front().parameters.emplace_back(std::move(parameter));
                break;
            }
            auto rejected = false;
            try
            {
                (void)PbrtImporter::import(std::move(scene));
            }
            catch (const std::runtime_error& error)
            {
                auto message           = std::string{error.what()};
                auto expected_location = std::string{"strict_case.pbrt:"} + std::to_string(test_case.line) + ":1";
                rejected               = message.find(expected_location) != std::string::npos &&
                                         message.find(test_case.name) != std::string::npos;
            }
            expect(rejected);
        }
    };

    "reject_duplicate_and_unused_resource_parameters"_test = []
    {
        auto duplicate = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        duplicate.integrator.parameters.emplace_back(test_parameter("integer", "maxdepth", 30u));
        auto duplicate_rejected = false;
        try
        {
            (void)PbrtImporter::import(std::move(duplicate));
        }
        catch (const std::runtime_error& error)
        {
            auto message       = std::string{error.what()};
            duplicate_rejected = message.find("strict_case.pbrt:30:1") != std::string::npos &&
                                 message.find("duplicate") != std::string::npos &&
                                 message.find("maxdepth") != std::string::npos;
        }
        expect(duplicate_rejected);

        auto unused = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        MaterialDesc unused_material;
        unused_material.source = SourceLocation{"strict_case.pbrt", 40u, 1u};
        unused_material.parameters.emplace_back(test_parameter("texture", "displacement", 41u));
        unused.named_materials.emplace("unused", std::move(unused_material));
        auto unused_rejected = false;
        try
        {
            (void)PbrtImporter::import(std::move(unused));
        }
        catch (const std::runtime_error& error)
        {
            auto message    = std::string{error.what()};
            unused_rejected = message.find("strict_case.pbrt:41:1") != std::string::npos &&
                              message.find("unused") != std::string::npos &&
                              message.find("displacement") != std::string::npos;
        }
        expect(unused_rejected);
    };

    "log_effective_scene_summary_and_resource_details"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        std::mutex mutex;
        luisa::vector<luisa::string> messages;
        auto sink = luisa::detail::create_sink_with_callback(
            [&](const char*, const char* message)
        {
            std::lock_guard lock{mutex};
            messages.emplace_back(message);
        });
        auto& logger        = luisa::detail::default_logger();
        auto original_sinks = logger.sinks();
        logger.sinks().clear();
        logger.sinks().push_back(sink);
        luisa::log_level_verbose();
        auto spec = PbrtImporter::import(std::move(parsed));
        (void)spec;
        luisa::log_flush();
        logger.sinks() = original_sinks;
        luisa::log_level_info();

        auto contains = [&](luisa::string_view expected) noexcept
        {
            return std::any_of(messages.begin(), messages.end(), [expected](auto&& message) noexcept
            {
                return message.find(expected) != luisa::string::npos;
            });
        };
        expect(contains("PBRT render config"));
        expect(contains("light_sampler=yutrel-uniform"));
        expect(contains("PBRT resources"));
        expect(contains("PBRT instance #0"));
        expect(contains("Imported PBRT scene"));
    };

    "validate_coated_diffuse_spec"_test = []
    {
        expect(!CoatedDiffuseSurfaceSpec{CoatedDiffuseSurfaceParams{}}.validate().has_value());

        auto zero_depth      = CoatedDiffuseSurfaceParams{};
        zero_depth.max_depth = 0u;
        expect(CoatedDiffuseSurfaceSpec{std::move(zero_depth)}.validate().has_value());

        auto zero_samples    = CoatedDiffuseSurfaceParams{};
        zero_samples.samples = 0u;
        expect(CoatedDiffuseSurfaceSpec{std::move(zero_samples)}.validate().has_value());
    };

    "generate_sphere_geometry"_test = []
    {
        Sphere sphere_level_0{2.0f, 0u};
        auto coarse = sphere_level_0.mesh();
        expect(coarse.vertices.size() == 12u);
        expect(coarse.triangles.size() == 20u);

        Sphere sphere_level_4{2.0f};
        auto mesh = sphere_level_4.mesh();
        expect(mesh.vertices.size() == 2562u);
        expect(mesh.triangles.size() == 5120u);
        expect((sphere_level_4.vertex_properties() & Shape::property_flag_has_vertex_normal) != 0u);
        expect((sphere_level_4.vertex_properties() & Shape::property_flag_has_vertex_uv) != 0u);

        for (auto vertex : mesh.vertices)
        {
            auto p = vertex.position();
            auto n = vertex.normal();
            expect(is_near(std::sqrt(luisa::dot(p, p)), 2.0f));
            expect(is_near(std::sqrt(luisa::dot(n, n)), 1.0f));
            expect(luisa::dot(p, n) > 0.0f);
            expect(std::isfinite(vertex.u) && std::isfinite(vertex.v));
        }
        for (auto triangle : mesh.triangles)
        {
            expect(triangle.i0 < mesh.vertices.size());
            expect(triangle.i1 < mesh.vertices.size());
            expect(triangle.i2 < mesh.vertices.size());
            auto p0 = mesh.vertices[triangle.i0].position();
            auto p1 = mesh.vertices[triangle.i1].position();
            auto p2 = mesh.vertices[triangle.i2].position();
            expect(luisa::dot(luisa::cross(p1 - p0, p2 - p0), p0) > 0.0f);
        }
    };

    "validate_sphere_spec"_test = []
    {
        expect(!SphereShapeSpec{1.0f}.validate().has_value());
        expect(SphereShapeSpec{0.0f}.validate().has_value());
        expect(SphereShapeSpec{-1.0f}.validate().has_value());
        expect(SphereShapeSpec{std::numeric_limits<float>::infinity()}.validate().has_value());
        expect(SphereShapeSpec{std::numeric_limits<float>::quiet_NaN()}.validate().has_value());
        expect(SphereShapeSpec{1.0f, Sphere::max_subdivision + 1u}.validate().has_value());
    };

    "import_fixture_ply_geometry"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        std::array<Matrix4, 3u> expected_transforms{
            parsed.shapes[0u].pbrt_transform,
            parsed.shapes[1u].pbrt_transform,
            parsed.shapes[2u].pbrt_transform,
        };
        auto spec = PbrtImporter::import(std::move(parsed));

        expect(spec.textures().size() == 3u);
        expect(spec.surfaces().size() == 3u);
        expect(spec.shapes().size() == 3u);
        expect(spec.instances().size() == 3u);

        auto instances = spec.instances();
        expect(instances[0u].surface != instances[1u].surface);
        expect(instances[0u].surface != instances[2u].surface);
        expect(instances[1u].surface != instances[2u].surface);

        std::array<std::filesystem::path, 3u> expected_paths{
            std::filesystem::absolute("test/scenes/mesh_00001.ply"),
            std::filesystem::absolute("test/scenes/mesh_00002.ply"),
            std::filesystem::absolute("test/scenes/mesh_00003.ply"),
        };
        auto shape_index = 0u;
        spec.shapes().visit_entries([&](ShapeRef, const SpecMeta&, const ShapeSpec* shape)
        {
            auto mesh = dynamic_cast<const MeshShapeSpec*>(shape);
            expect(mesh != nullptr);
            if (mesh != nullptr)
            {
                expect(shape_index < expected_paths.size());
                if (shape_index < expected_paths.size())
                {
                    expect(mesh->path().lexically_normal() == expected_paths[shape_index].lexically_normal());
                }
            }
            shape_index++;
        });
        expect(shape_index == expected_paths.size());

        for (auto instance_index = 0u; instance_index < instances.size(); instance_index++)
        {
            for (auto column = 0u; column < 4u; column++)
            {
                for (auto row = 0u; row < 4u; row++)
                {
                    expect(is_near(
                        instances[instance_index].transform[column][row],
                        expected_transforms[instance_index][row * 4u + column]));
                }
            }
        }
        auto&& first = instances[0u].transform;
        expect(is_near(first[0u].x, 0.213f));
        expect(is_near(first[1u].y, 0.213f));
        expect(is_near(first[2u].z, 0.213f));

        for (auto instance_index : {1u, 2u})
        {
            auto&& transform = instances[instance_index].transform;
            expect(is_near(transform[3u].x, 0.0f));
            expect(is_near(transform[3u].y, 2.2f));
            expect(is_near(transform[3u].z, 0.0f));
            expect(is_near(column_length(transform, 0u), 0.5f));
            expect(is_near(column_length(transform, 1u), 0.5f));
            expect(is_near(column_length(transform, 2u), 0.5f));
        }
    };

    "reuse_inherited_inline_material"_test = []
    {
        auto parsed                             = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.shapes[1u].material.inline_index = parsed.shapes[0u].material.inline_index;
        auto spec                               = PbrtImporter::import(std::move(parsed));
        expect(spec.instances()[0u].surface == spec.instances()[1u].surface);
    };

    "import_shape_alpha_as_cached_opacity_surfaces"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/shape_alpha.pbrt");
        auto spec   = PbrtImporter::import(std::move(parsed));
        auto instances = spec.instances();
        expect(instances.size() == 6u);
        if (instances.size() != 6u)
        {
            return;
        }

        expect(instances[0u].surface == instances[1u].surface);
        expect(instances[0u].light.has_value());
        expect(instances[2u].surface == instances[4u].surface);
        expect(instances[2u].surface != instances[3u].surface);

        auto mask_wrapper = dynamic_cast<const OpacitySurfaceSpec*>(
            &spec.surfaces().spec(instances[0u].surface));
        auto half_wrapper = dynamic_cast<const OpacitySurfaceSpec*>(
            &spec.surfaces().spec(instances[2u].surface));
        auto null_wrapper = dynamic_cast<const OpacitySurfaceSpec*>(
            &spec.surfaces().spec(instances[5u].surface));
        expect(mask_wrapper != nullptr);
        expect(half_wrapper != nullptr);
        expect(null_wrapper != nullptr);
        if (mask_wrapper != nullptr && half_wrapper != nullptr)
        {
            expect(mask_wrapper->base() == half_wrapper->base());
            expect(mask_wrapper->alpha() != half_wrapper->alpha());
        }
        if (null_wrapper != nullptr)
        {
            expect(dynamic_cast<const NullSurfaceSpec*>(
                       &spec.surfaces().spec(null_wrapper->base())) != nullptr);
        }
    };

    "reject_invalid_shape_alpha_texture_references"_test = []
    {
        auto nonfinite = PbrtParser::parse("test/scenes/shape_alpha.pbrt");
        nonfinite.shapes[0u].alpha_texture.reset();
        nonfinite.shapes[0u].alpha = std::numeric_limits<float>::quiet_NaN();
        auto nonfinite_rejected = false;
        try
        {
            (void)PbrtImporter::import(std::move(nonfinite));
        }
        catch (const std::runtime_error& error)
        {
            nonfinite_rejected = std::string{error.what()}.find("alpha must be finite") != std::string::npos;
        }
        expect(nonfinite_rejected);

        auto undefined = PbrtParser::parse("test/scenes/shape_alpha.pbrt");
        undefined.shapes[0u].alpha_texture.emplace("missing-alpha");
        auto undefined_rejected = false;
        try
        {
            (void)PbrtImporter::import(std::move(undefined));
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            undefined_rejected = message.find("missing-alpha") != std::string::npos &&
                                 message.find("undefined texture") != std::string::npos;
        }
        expect(undefined_rejected);

        auto spectrum = PbrtParser::parse("test/scenes/shape_alpha.pbrt");
        spectrum.shapes[0u].alpha_texture.emplace("spectrum-mask");
        auto spectrum_rejected = false;
        try
        {
            (void)PbrtImporter::import(std::move(spectrum));
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            spectrum_rejected = message.find("spectrum-mask") != std::string::npos &&
                                message.find("float texture") != std::string::npos;
        }
        expect(spectrum_rejected);
    };

    "reject_unsupported_coated_displacement"_test = []
    {
        auto parsed   = PbrtParser::parse("test/scenes/coated_diffuse_materials.pbrt");
        auto rejected = false;
        try
        {
            (void)PbrtImporter::import(std::move(parsed));
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            rejected     = message.find("coated_diffuse_materials.pbrt") != std::string::npos &&
                           message.find("displacement") != std::string::npos &&
                           message.find("unsupported parameter") != std::string::npos;
        }
        expect(rejected);
    };

    "fold_static_scale_texture"_test = []
    {
        ConstantTexture base{make_float4(2.0f, 3.0f, 4.0f, 1.0f)};
        ScaleTexture scaled{&base, make_float4(0.5f), make_float4(1.0f)};
        auto value = scaled.evaluate_static();
        expect(value.has_value());
        if (value)
        {
            expect(is_near(value->x, 2.0f));
            expect(is_near(value->y, 2.5f));
            expect(is_near(value->z, 3.0f));
            expect(is_near(value->w, 1.5f));
        }
        expect(scaled.channels() == base.channels());
    };

    "import_scaled_imagemap"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/imagemap_scale_import.pbrt");
        auto spec   = PbrtImporter::import(std::move(parsed));

        const ScaleTextureSpec* scaled = nullptr;
        spec.textures().visit_entries([&](TextureRef, const SpecMeta& meta, const TextureSpec* texture)
        {
            if (meta.name == "scaled")
            {
                scaled = dynamic_cast<const ScaleTextureSpec*>(texture);
            }
        });

        expect(scaled != nullptr);
        if (scaled != nullptr)
        {
            expect(is_near(scaled->scale().x, 0.25f));
            auto image = dynamic_cast<const ImageTextureSpec*>(&spec.textures().spec(scaled->base()));
            expect(image != nullptr);
        }
    };

    "import_imagemap_encodings"_test = []
    {
        auto parsed  = PbrtParser::parse("test/scenes/imagemap_encodings.pbrt");
        auto spec    = PbrtImporter::import(std::move(parsed));
        auto matched = 0u;
        spec.textures().visit_entries([&](TextureRef, const SpecMeta& meta, const TextureSpec* texture)
        {
            if (meta.name == "default-png" || meta.name == "explicit-srgb")
            {
                auto image = static_cast<const ImageTextureSpec*>(texture);
                expect(image->encoding() == Texture::Encoding::SRGB);
                matched++;
            }
            else if (meta.name == "default-linear" || meta.name == "explicit-linear")
            {
                auto image = static_cast<const ImageTextureSpec*>(texture);
                expect(image->encoding() == Texture::Encoding::LINEAR);
                matched++;
            }
        });
        expect(matched == 4u);
    };

    "import_trianglemesh_without_normals"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/trianglemesh_without_normals.pbrt");
        auto spec   = PbrtImporter::import(std::move(parsed));
        expect(spec.shapes().size() == 1u);
        expect(spec.instances().size() == 1u);
    };

    "import_checkerboard_textures"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/checkerboard_textures.pbrt");
        auto spec   = PbrtImporter::import(std::move(parsed));

        const CheckerBoardTextureSpec* floor     = nullptr;
        const CheckerBoardTextureSpec* defaults  = nullptr;
        const CheckerBoardTextureSpec* reference = nullptr;
        spec.textures().visit_entries([&](TextureRef, const SpecMeta& meta, const TextureSpec* texture)
        {
            auto checker = dynamic_cast<const CheckerBoardTextureSpec*>(texture);
            if (meta.name == "floor-checker")
            {
                floor = checker;
            }
            else if (meta.name == "float-defaults")
            {
                defaults = checker;
            }
            else if (meta.name == "float-reference")
            {
                reference = checker;
            }
        });

        expect(floor != nullptr);
        expect(defaults != nullptr);
        expect(reference != nullptr);
        if (floor != nullptr)
        {
            expect(is_near(floor->uv_scale().x, 20.0f));
            expect(is_near(floor->uv_scale().y, 20.0f));
            auto tex1 = dynamic_cast<const ConstantTextureSpec*>(&spec.textures().spec(floor->tex1()));
            auto tex2 = dynamic_cast<const ConstantTextureSpec*>(&spec.textures().spec(floor->tex2()));
            expect(tex1 != nullptr);
            expect(tex2 != nullptr);
            if (tex1 != nullptr && tex2 != nullptr)
            {
                expect(is_near(tex1->value().x, 0.325f));
                expect(is_near(tex2->value().x, 0.725f));
            }
        }
        if (defaults != nullptr)
        {
            auto tex1 = dynamic_cast<const ConstantTextureSpec*>(&spec.textures().spec(defaults->tex1()));
            auto tex2 = dynamic_cast<const ConstantTextureSpec*>(&spec.textures().spec(defaults->tex2()));
            expect(tex1 != nullptr);
            expect(tex2 != nullptr);
            if (tex1 != nullptr && tex2 != nullptr)
            {
                expect(is_near(tex1->value().x, 1.0f));
                expect(is_near(tex2->value().x, 0.0f));
            }
        }
        if (reference != nullptr)
        {
            auto tex1 = dynamic_cast<const ConstantTextureSpec*>(&spec.textures().spec(reference->tex1()));
            auto tex2 = dynamic_cast<const ConstantTextureSpec*>(&spec.textures().spec(reference->tex2()));
            expect(tex1 != nullptr);
            expect(tex2 != nullptr);
            if (tex1 != nullptr && tex2 != nullptr)
            {
                expect(is_near(tex1->value().x, 0.75f));
                expect(is_near(tex2->value().x, 0.25f));
            }
        }
    };

    "reject_invalid_checkerboard_references"_test = []
    {
        auto unknown = PbrtParser::parse("test/scenes/checkerboard_textures.pbrt");
        unknown.textures[0u].tex1.texture.emplace("missing-texture");
        auto rejected_unknown = false;
        try
        {
            (void)PbrtImporter::import(std::move(unknown));
        }
        catch (const std::runtime_error& error)
        {
            auto message     = std::string{error.what()};
            rejected_unknown = message.find("checkerboard_textures.pbrt") != std::string::npos &&
                               message.find("unknown tex1 texture") != std::string::npos &&
                               message.find("missing-texture") != std::string::npos;
        }
        expect(rejected_unknown);

        auto mismatch = PbrtParser::parse("test/scenes/checkerboard_textures.pbrt");
        mismatch.textures[0u].tex1.texture.emplace("float-source");
        auto rejected_mismatch = false;
        try
        {
            (void)PbrtImporter::import(std::move(mismatch));
        }
        catch (const std::runtime_error& error)
        {
            auto message      = std::string{error.what()};
            rejected_mismatch = message.find("checkerboard_textures.pbrt") != std::string::npos &&
                                message.find("tex1 'float-source'") != std::string::npos &&
                                message.find("spectrum texture") != std::string::npos;
        }
        expect(rejected_mismatch);
    };

    "reject_out_of_range_inline_material"_test = []
    {
        auto parsed                             = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.shapes[0u].material.inline_index = 99u;
        auto source_located                     = false;
        try
        {
            (void)PbrtImporter::import(std::move(parsed));
        }
        catch (const std::runtime_error& error)
        {
            auto message   = std::string{error.what()};
            source_located = message.find("test/scenes/import_geometry.pbrt") != std::string::npos &&
                             message.find("out-of-range inline material 99") != std::string::npos;
        }
        expect(source_located);
    };

    "reject_ambiguous_material_binding"_test = []
    {
        auto parsed                      = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.shapes[0u].material.named = "named";
        auto source_located              = false;
        try
        {
            (void)PbrtImporter::import(std::move(parsed));
        }
        catch (const std::runtime_error& error)
        {
            auto message   = std::string{error.what()};
            source_located = message.find("test/scenes/import_geometry.pbrt") != std::string::npos &&
                             message.find("both named and inline material bindings") != std::string::npos;
        }
        expect(source_located);
    };

    "import_inline_coated_diffuse_material"_test = []
    {
        auto parsed                      = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.materials[0u].type        = MaterialDesc::Type::CoatedDiffuse;
        parsed.materials[0u].roughness   = 0.1f;
        parsed.materials[0u].u_roughness = 0.1f;
        parsed.materials[0u].v_roughness = 0.1f;
        auto spec                        = PbrtImporter::import(std::move(parsed));
        auto found                       = false;
        spec.surfaces().visit_entries([&](SurfaceRef, const SpecMeta&, const SurfaceSpec* surface)
        {
            found |= dynamic_cast<const CoatedDiffuseSurfaceSpec*>(surface) != nullptr;
        });
        expect(found);
    };

    "import_named_material_binding"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.named_materials.emplace("named", parsed.materials[0u]);
        parsed.shapes[0u].material.named = "named";
        parsed.shapes[0u].material.inline_index.reset();
        auto spec = PbrtImporter::import(std::move(parsed));
        expect(spec.instances().size() == 3u);
        expect(spec.surfaces().size() == 4u);
    };

    "reject_missing_ply_file"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        parsed.shapes[0u].filename.emplace("missing.ply");
        auto source_located = false;
        try
        {
            (void)PbrtImporter::import(std::move(parsed));
        }
        catch (const std::runtime_error& error)
        {
            auto message   = std::string{error.what()};
            source_located = message.find("test/scenes/import_geometry.pbrt") != std::string::npos &&
                             message.find("regular file") != std::string::npos;
        }
        expect(source_located);
    };

    "import_uniform_infinite_environment"_test = []
    {
        auto parsed      = PbrtParser::parse("test/scenes/infinite_uniform.pbrt");
        auto spec        = PbrtImporter::import(std::move(parsed));
        auto environment = dynamic_cast<const UniformEnvironmentSpec*>(
            &spec.environments().spec(spec.render().environment));
        expect(environment != nullptr);
        if (environment == nullptr)
        {
            return;
        }
        auto expected_scale = 2.0f * 0.05f / std::acos(-1.0f);
        expect(is_near(environment->scale(), expected_scale, 1e-8f));
        expect(!environment->validate().has_value());

        auto emission = dynamic_cast<const ConstantTextureSpec*>(
            &spec.textures().spec(environment->emission()));
        expect(emission != nullptr);
        if (emission != nullptr)
        {
            expect(is_near(emission->value().x, 0.8f));
            expect(is_near(emission->value().y, 0.9f));
            expect(is_near(emission->value().z, 1.0f));
        }
        expect(UniformEnvironmentSpec{environment->emission(), -1.0f}
                   .validate()
                   .has_value());

        auto default_parsed = PbrtParser::parse("test/scenes/infinite_missing_filename.pbrt");
        auto default_spec   = PbrtImporter::import(std::move(default_parsed));
        auto default_environment = dynamic_cast<const UniformEnvironmentSpec*>(
            &default_spec.environments().spec(default_spec.render().environment));
        expect(default_environment != nullptr);
        if (default_environment != nullptr)
        {
            auto default_emission = dynamic_cast<const ConstantTextureSpec*>(
                &default_spec.textures().spec(default_environment->emission()));
            expect(default_emission != nullptr);
            if (default_emission != nullptr)
            {
                expect(is_near(default_emission->value().x, 1.0f));
                expect(is_near(default_emission->value().y, 1.0f));
                expect(is_near(default_emission->value().z, 1.0f));
            }
        }
    };

    "import_distant_environment"_test = []
    {
        auto parsed      = PbrtParser::parse("test/scenes/distant_basic.pbrt");
        auto spec        = PbrtImporter::import(std::move(parsed));
        auto environment = dynamic_cast<const DistantEnvironmentSpec*>(
            &spec.environments().spec(spec.render().environment));
        expect(environment != nullptr);
        if (environment == nullptr)
        {
            return;
        }
        expect(is_near(environment->scale(), 1.0f));
        expect(is_near(environment->direction().x, 1.0f));
        expect(is_near(environment->direction().y, 0.0f));
        expect(is_near(environment->direction().z, 0.0f));
        expect(!environment->validate().has_value());

        auto emission = dynamic_cast<const ConstantTextureSpec*>(
            &spec.textures().spec(environment->emission()));
        expect(emission != nullptr);
        if (emission != nullptr)
        {
            expect(is_near(emission->value().x, 8.0f));
            expect(is_near(emission->value().y, 8.0f));
            expect(is_near(emission->value().z, 8.0f));
        }
        expect(DistantEnvironmentSpec{
            environment->emission(),
            -1.0f,
            make_float3(1.0f, 0.0f, 0.0f)}
                   .validate()
                   .has_value());
        expect(DistantEnvironmentSpec{
            environment->emission(),
            1.0f,
            make_float3(2.0f, 0.0f, 0.0f)}
                   .validate()
                   .has_value());
    };

    "import_multiple_environment_lights"_test = []
    {
        auto parsed = PbrtParser::parse("test/scenes/multiple_environment_lights.pbrt");
        auto spec   = PbrtImporter::import(std::move(parsed));
        auto grouped = dynamic_cast<const GroupedEnvironmentSpec*>(
            &spec.environments().spec(spec.render().environment));
        expect(grouped != nullptr);
        if (grouped == nullptr)
        {
            return;
        }
        expect(grouped->environments().size() == 3u);
        expect(!grouped->validate().has_value());
        if (grouped->environments().size() != 3u)
        {
            return;
        }

        auto uniform = dynamic_cast<const UniformEnvironmentSpec*>(
            &spec.environments().spec(grouped->environments()[0u]));
        auto distant_a = dynamic_cast<const DistantEnvironmentSpec*>(
            &spec.environments().spec(grouped->environments()[1u]));
        auto distant_b = dynamic_cast<const DistantEnvironmentSpec*>(
            &spec.environments().spec(grouped->environments()[2u]));
        expect(uniform != nullptr);
        expect(distant_a != nullptr);
        expect(distant_b != nullptr);
        if (distant_a != nullptr)
        {
            expect(is_near(distant_a->scale(), 3.0f));
            expect(is_near(distant_a->direction().x, 1.0f));
        }
        if (distant_b != nullptr)
        {
            expect(is_near(distant_b->scale(), 1.0f));
            expect(is_near(distant_b->direction().y, 1.0f));
        }
        expect(GroupedEnvironmentSpec{luisa::vector<EnvironmentRef>{}}
                   .validate()
                   .has_value());
    };

    return 0;
}();

} // namespace

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
