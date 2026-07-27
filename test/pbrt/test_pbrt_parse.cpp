// Tests for PBRT scene parsing.
// This test covers parser defaults, parameters, and invalid declarations.

#include "ut/ut.hpp"

#include "pbrt/pbrt_parser.h"

#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace Yutrel;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace
{

[[nodiscard]] bool is_near(float a, float b) noexcept
{
    return std::abs(a - b) < 1e-4f;
}

[[nodiscard]] const RawParameter* find_parameter(luisa::span<const RawParameter> parameters, luisa::string_view name) noexcept
{
    for (auto&& parameter : parameters)
    {
        if (parameter.name == name)
        {
            return &parameter;
        }
    }
    return nullptr;
}

[[nodiscard]] bool parse_error_contains(
    const std::filesystem::path& path,
    std::initializer_list<std::string_view> expected)
{
    try
    {
        (void)PbrtParser::parse(path);
    }
    catch (const std::runtime_error& error)
    {
        auto message = std::string{error.what()};
        for (auto text : expected)
        {
            if (message.find(text) == std::string::npos)
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

static auto test_pbrt_parse_registration = []
{
    "parse_film_exposure"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/film_exposure.pbrt");
        expect(is_near(scene.film.iso, 200.0f));
        expect(is_near(scene.film.max_component_value, 10.0f));
        expect(is_near(scene.camera.shutter_open, 0.25f));
        expect(is_near(scene.camera.shutter_close, 0.75f));

        auto defaults = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        expect(is_near(defaults.film.iso, 100.0f));
        expect(std::isinf(defaults.film.max_component_value));
        expect(is_near(defaults.camera.shutter_open, 0.0f));
        expect(is_near(defaults.camera.shutter_close, 1.0f));
    };

    "parse_camera_world_up"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/camera_world_up.pbrt");
        expect(scene.camera.world_up.has_value());
        if (scene.camera.world_up)
        {
            expect(is_near(scene.camera.world_up->x, 0.0f));
            expect(is_near(scene.camera.world_up->y, 1.0f));
            expect(is_near(scene.camera.world_up->z, 0.0f));
        }

        auto fallback = PbrtParser::parse("test/scenes/import_geometry.pbrt");
        expect(!fallback.camera.world_up.has_value());
    };

    "parse_sobol_sampler"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/sobol_sampler.pbrt");
        expect(scene.sampler.type == SamplerDesc::Type::Sobol);
        expect(scene.sampler.pixel_samples == 32u);
        expect(scene.sampler.seed == 42u);

        auto defaults = PbrtParser::parse("test/scenes/sobol_sampler_default.pbrt");
        expect(defaults.sampler.type == SamplerDesc::Type::Sobol);
        expect(defaults.sampler.pixel_samples == 16u);
        expect(defaults.sampler.seed == 20120712u);
    };

    "parse_zsobol_sampler"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/zsobol_sampler.pbrt");
        expect(scene.sampler.type == SamplerDesc::Type::ZSobol);
        expect(scene.sampler.pixel_samples == 32u);
        expect(scene.sampler.seed == 42u);

        auto defaults = PbrtParser::parse("test/scenes/zsobol_sampler_default.pbrt");
        expect(defaults.sampler.type == SamplerDesc::Type::ZSobol);
        expect(defaults.sampler.pixel_samples == 16u);
        expect(defaults.sampler.seed == 20120712u);
    };

    "reject_invalid_sobol_sampler"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/sobol_sampler_zero_spp.pbrt",
            {"sobol_sampler_zero_spp.pbrt", "pixelsamples", "greater than zero"}));
        expect(parse_error_contains(
            "test/scenes/sobol_sampler_bad_randomization.pbrt",
            {"sobol_sampler_bad_randomization.pbrt", "randomization", "fastowen"}));
    };

    "reject_invalid_zsobol_sampler"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/zsobol_sampler_zero_spp.pbrt",
            {"zsobol_sampler_zero_spp.pbrt", "pixelsamples", "greater than zero"}));
        expect(parse_error_contains(
            "test/scenes/zsobol_sampler_non_power_two.pbrt",
            {"zsobol_sampler_non_power_two.pbrt", "pixelsamples", "power of two"}));
        expect(parse_error_contains(
            "test/scenes/zsobol_sampler_bad_randomization.pbrt",
            {"zsobol_sampler_bad_randomization.pbrt", "randomization", "fastowen"}));
    };

    "parse_gaussian_filter"_test = []
    {
        auto defaults = PbrtParser::parse("test/scenes/pbrt_defaults.pbrt");
        expect(defaults.filter.type == FilterDesc::Type::Gaussian);
        expect(is_near(defaults.filter.radius.x, 1.5f));
        expect(is_near(defaults.filter.radius.y, 1.5f));
        expect(is_near(defaults.filter.sigma, 0.5f));

        auto explicit_filter = PbrtParser::parse("test/scenes/gaussian_filter.pbrt");
        expect(explicit_filter.filter.type == FilterDesc::Type::Gaussian);
        expect(is_near(explicit_filter.filter.radius.x, 1.25f));
        expect(is_near(explicit_filter.filter.radius.y, 1.25f));
        expect(is_near(explicit_filter.filter.sigma, 0.25f));
    };

    "parse_imagemap_filters"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/imagemap_filters.pbrt");
        expect(scene.textures.size() == 3u);
        expect(scene.textures[0u].filter == TextureDesc::Filter::Bilinear);
        expect(scene.textures[1u].filter == TextureDesc::Filter::Point);
        expect(scene.textures[2u].filter == TextureDesc::Filter::Bilinear);
    };

    "parse_imagemap_encodings"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/imagemap_encodings.pbrt");
        expect(scene.textures.size() == 4u);
        expect(scene.textures[0u].encoding == TextureDesc::Encoding::Automatic);
        expect(scene.textures[1u].encoding == TextureDesc::Encoding::Automatic);
        expect(scene.textures[2u].encoding == TextureDesc::Encoding::Linear);
        expect(scene.textures[3u].encoding == TextureDesc::Encoding::SRGB);
    };

    "reject_invalid_imagemap_encodings"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/imagemap_invalid_encoding.pbrt",
            {"imagemap_invalid_encoding.pbrt", "unsupported imagemap encoding", "gamma 2.2", "linear", "sRGB"}));
        expect(parse_error_contains(
            "test/scenes/imagemap_invalid_encoding_type.pbrt",
            {"imagemap_invalid_encoding_type.pbrt", "unsupported parameter", "integer encoding"}));
        expect(parse_error_contains(
            "test/scenes/imagemap_duplicate_encoding.pbrt",
            {"imagemap_duplicate_encoding.pbrt", "duplicate texture parameter", "string encoding"}));
    };

    "parse_imagemap_scale"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/imagemap_scale.pbrt");
        expect(scene.textures.size() == 2u);
        expect(is_near(scene.textures[0u].image_scale, 0.25f));
        expect(is_near(scene.textures[1u].image_scale, 1.0f));
    };

    "parse_checkerboard_textures"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/checkerboard_textures.pbrt");
        expect(scene.textures.size() == 4u);

        auto&& floor = scene.textures[0u];
        expect(floor.type == TextureDesc::Type::Checkerboard);
        expect(floor.value_type == TextureDesc::ValueType::Spectrum);
        expect(is_near(floor.uv_scale.x, 20.0f));
        expect(is_near(floor.uv_scale.y, 20.0f));
        expect(!floor.tex1.texture.has_value());
        expect(!floor.tex2.texture.has_value());
        expect(is_near(floor.tex1.constant.x, 0.325f));
        expect(is_near(floor.tex1.constant.y, 0.31f));
        expect(is_near(floor.tex1.constant.z, 0.25f));
        expect(is_near(floor.tex2.constant.x, 0.725f));
        expect(is_near(floor.tex2.constant.y, 0.71f));
        expect(is_near(floor.tex2.constant.z, 0.68f));

        auto&& defaults = scene.textures[1u];
        expect(defaults.type == TextureDesc::Type::Checkerboard);
        expect(defaults.value_type == TextureDesc::ValueType::Float);
        expect(is_near(defaults.uv_scale.x, 1.0f));
        expect(is_near(defaults.uv_scale.y, 1.0f));
        expect(is_near(defaults.tex1.constant.x, 1.0f));
        expect(is_near(defaults.tex2.constant.x, 0.0f));

        auto&& reference = scene.textures[2u];
        expect(reference.type == TextureDesc::Type::Checkerboard);
        expect(reference.tex1.texture == luisa::optional<luisa::string>{"float-source"});
        expect(!reference.tex2.texture.has_value());
        expect(is_near(reference.tex2.constant.x, 0.25f));
    };

    "reject_invalid_checkerboard_parameters"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/checkerboard_dimension_3.pbrt",
            {"checkerboard_dimension_3.pbrt", "dimension 3", "only 2D"}));
        expect(parse_error_contains(
            "test/scenes/checkerboard_mapping_spherical.pbrt",
            {"checkerboard_mapping_spherical.pbrt", "mapping 'spherical'", "only 'uv'"}));
        expect(parse_error_contains(
            "test/scenes/checkerboard_conflicting_tex1.pbrt",
            {"checkerboard_conflicting_tex1.pbrt", "tex1", "both texture and rgb"}));
        expect(parse_error_contains(
            "test/scenes/checkerboard_duplicate_uscale.pbrt",
            {"checkerboard_duplicate_uscale.pbrt", "duplicate texture parameter", "float uscale"}));
        expect(parse_error_contains(
            "test/scenes/checkerboard_invalid_input_type.pbrt",
            {"checkerboard_invalid_input_type.pbrt", "unsupported parameter", "float tex1"}));
        expect(parse_error_contains(
            "test/scenes/checkerboard_nonfinite_scale.pbrt",
            {"checkerboard_nonfinite_scale.pbrt", "UV scale", "finite"}));
    };

    "reject_unsupported_imagemap_filters"_test = []
    {
        struct Case
        {
            const char* path;
            const char* filename;
            const char* expected;
        };
        std::array cases{
            Case{"test/scenes/imagemap_unsupported_filter.pbrt", "imagemap_unsupported_filter.pbrt", "trilinear"},
            Case{"test/scenes/imagemap_ewa_filter.pbrt", "imagemap_ewa_filter.pbrt", "ewa"},
            Case{"test/scenes/imagemap_anisotropic_filter.pbrt", "imagemap_anisotropic_filter.pbrt", "anisotropic"},
            Case{"test/scenes/imagemap_unknown_filter.pbrt", "imagemap_unknown_filter.pbrt", "unknown"},
        };
        for (auto test_case : cases)
        {
            expect(parse_error_contains(
                test_case.path,
                {test_case.filename, "unsupported imagemap filter", test_case.expected, "point", "bilinear"}));
        }
    };

    "reject_invalid_imagemap_filter_declarations"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/imagemap_invalid_filter_type.pbrt",
            {"imagemap_invalid_filter_type.pbrt", "unsupported parameter", "integer filter"}));
        expect(parse_error_contains(
            "test/scenes/imagemap_duplicate_filter.pbrt",
            {"imagemap_duplicate_filter.pbrt", "duplicate texture parameter", "string filter"}));
    };

    "reject_nonfinite_imagemap_scale"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/imagemap_nonfinite_scale.pbrt",
            {"imagemap_nonfinite_scale.pbrt", "imagemap texture scale", "finite"}));
    };

    "reject_imagemap_without_filename"_test = []
    {
        auto rejected = false;
        try
        {
            (void)PbrtParser::parse("test/scenes/imagemap_missing_filename.pbrt");
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            rejected     = message.find("imagemap_missing_filename.pbrt") != std::string::npos &&
                           message.find("filename") != std::string::npos;
        }
        expect(rejected);
    };

    "reject_conflicting_material_reflectance"_test = []
    {
        auto rejected = false;
        try
        {
            (void)PbrtParser::parse("test/scenes/material_reflectance_conflict.pbrt");
        }
        catch (const std::runtime_error& error)
        {
            auto message = std::string{error.what()};
            rejected     = message.find("material_reflectance_conflict.pbrt") != std::string::npos &&
                           message.find("both rgb and texture") != std::string::npos;
        }
        expect(rejected);
    };

    "parse_sphere_parameters"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/sphere_parameters.pbrt");
        expect(scene.shapes.size() == 2u);
        expect(scene.shapes[0u].type == ShapeDesc::Type::Sphere);
        expect(is_near(scene.shapes[0u].radius, 2.5f));
        expect(scene.shapes[0u].sphere_subdivision == 3u);
        expect(scene.shapes[1u].type == ShapeDesc::Type::Sphere);
        expect(is_near(scene.shapes[1u].radius, 1.0f));
        expect(scene.shapes[1u].sphere_subdivision == ShapeDesc::sphere_default_subdivision);
    };

    "parse_shape_alpha_for_all_shape_types"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/shape_alpha.pbrt");
        expect(scene.shapes.size() == 6u);
        if (scene.shapes.size() != 6u)
        {
            return;
        }
        expect(scene.shapes[0u].type == ShapeDesc::Type::TriangleMesh);
        expect(scene.shapes[0u].alpha_texture == luisa::optional<luisa::string>{"mask"});
        expect(scene.shapes[1u].type == ShapeDesc::Type::PlyMesh);
        expect(scene.shapes[1u].alpha_texture == luisa::optional<luisa::string>{"mask"});
        expect(scene.shapes[2u].type == ShapeDesc::Type::Sphere);
        expect(is_near(scene.shapes[2u].alpha, 0.5f));
        expect(is_near(scene.shapes[3u].alpha, 0.25f));
        expect(is_near(scene.shapes[4u].alpha, 0.5f));
        expect(is_near(scene.shapes[5u].alpha, 0.0f));
    };

    "reject_invalid_shape_alpha_declarations"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/shape_alpha_conflict.pbrt",
            {"shape_alpha_conflict.pbrt", "alpha", "more than once"}));
        expect(parse_error_contains(
            "test/scenes/shape_alpha_duplicate.pbrt",
            {"shape_alpha_duplicate.pbrt", "alpha", "more than once"}));
        expect(parse_error_contains(
            "test/scenes/shape_alpha_nonfinite.pbrt",
            {"shape_alpha_nonfinite.pbrt", "alpha", "finite"}));
    };

    "parse_trianglemesh_without_normals"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/trianglemesh_without_normals.pbrt");
        expect(scene.meshes.size() == 1u);
        expect(scene.meshes[0u].positions.size() == 4u);
        expect(scene.meshes[0u].normals.empty());
        expect(scene.meshes[0u].uvs.empty());
        expect(scene.meshes[0u].indices.size() == 2u);
    };

    "reject_trianglemesh_normal_count_mismatch"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/trianglemesh_normal_count_mismatch.pbrt",
            {"trianglemesh_normal_count_mismatch.pbrt", "normal N", "count", "point3 P"}));
    };

    "parse_coated_diffuse_materials"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/coated_diffuse_materials.pbrt");
        expect(scene.materials.size() == 1u);
        expect(scene.named_materials.size() == 1u);

        auto&& inline_material = scene.materials.front();
        expect(inline_material.type == MaterialDesc::Type::CoatedDiffuse);
        expect(is_near(inline_material.reflectance.x, 0.2f));
        expect(is_near(inline_material.reflectance.y, 0.4f));
        expect(is_near(inline_material.reflectance.z, 0.6f));
        expect(is_near(inline_material.roughness, 0.25f));
        expect(is_near(inline_material.u_roughness, 0.25f));
        expect(is_near(inline_material.v_roughness, 0.25f));
        expect(find_parameter(inline_material.parameters, "displacement") != nullptr);

        auto&& named_material = scene.named_materials.at("coated-default");
        expect(named_material.type == MaterialDesc::Type::CoatedDiffuse);
        expect(is_near(named_material.reflectance.x, 0.5f));
        expect(is_near(named_material.reflectance.y, 0.5f));
        expect(is_near(named_material.reflectance.z, 0.5f));
        expect(is_near(named_material.roughness, 0.0f));
        expect(named_material.remap_roughness);
    };

    "parse_dielectric_f11"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/dielectric_f11.pbrt");
        expect(scene.materials.size() == 2u);
        expect(scene.named_materials.size() == 1u);

        auto&& named = scene.named_materials.at("f11-named");
        expect(named.type == MaterialDesc::Type::Dielectric);
        expect(named.eta_spectrum == luisa::optional<luisa::string>{"glass-F11"});
        expect(!named.eta_texture.has_value());

        expect(scene.materials[0u].eta_spectrum ==
               luisa::optional<luisa::string>{"glass-F11"});
        expect(!scene.materials[1u].eta_spectrum.has_value());
        expect(is_near(scene.materials[1u].eta, 1.5f));
    };

    "reject_conflicting_dielectric_eta"_test = []
    {
        expect(parse_error_contains(
            "test/scenes/dielectric_eta_conflict.pbrt",
            {"dielectric_eta_conflict.pbrt", "eta", "spectrum", "float"}));
    };

    "parse_homogeneous_medium_and_interfaces"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/homogeneous_medium.pbrt");
        expect(scene.integrator.type == IntegratorDesc::Type::VolPath);
        expect(scene.named_media.size() == 1u);
        auto&& medium = scene.named_media.at("fog");
        expect(medium.type == MediumDesc::Type::Homogeneous);
        expect(is_near(medium.sigma_a.x, 0.1f));
        expect(is_near(medium.sigma_s.z, 0.6f));
        expect(is_near(medium.scale, 2.0f));
        expect(is_near(medium.g, 0.25f));
        expect(scene.shapes.size() == 2u);
        expect(scene.shapes[0u].medium_interface.inside == "fog");
        expect(scene.shapes[0u].medium_interface.outside.empty());
        expect(scene.shapes[1u].medium_interface.inside.empty());
        expect(scene.shapes[1u].medium_interface.outside == "fog");
        expect(scene.materials[0u].type == MaterialDesc::Type::Interface);
        expect(scene.materials[1u].type == MaterialDesc::Type::Dielectric);
    };

    "reject_invalid_homogeneous_media"_test = []
    {
        expect(parse_error_contains("test/scenes/homogeneous_medium_duplicate.pbrt", {"homogeneous_medium_duplicate.pbrt", "redefined"}));
        expect(parse_error_contains("test/scenes/homogeneous_medium_negative.pbrt", {"homogeneous_medium_negative.pbrt", "sigma_a", "non-negative"}));
        expect(parse_error_contains("test/scenes/homogeneous_medium_bad_g.pbrt", {"homogeneous_medium_bad_g.pbrt", "abs(g) < 1"}));
    };

    "reject_invalid_sphere_parameters"_test = []
    {
        for (auto path : {
                 "test/scenes/sphere_duplicate_radius.pbrt",
                 "test/scenes/sphere_invalid_radius.pbrt",
                 "test/scenes/sphere_invalid_subdivision.pbrt",
                 "test/scenes/sphere_clipped.pbrt",
             })
        {
            auto rejected = false;
            try
            {
                (void)PbrtParser::parse(path);
            }
            catch (const std::runtime_error& error)
            {
                rejected = std::string{error.what()}.find(path) != std::string::npos;
            }
            expect(rejected) << "invalid sphere parameters should produce a source-located parse error";
        }
    };

    "reject_invalid_ply_filenames"_test = []
    {
        for (auto path : {
                 "test/scenes/ply_missing_filename.pbrt",
                 "test/scenes/ply_empty_filename.pbrt",
                 "test/scenes/ply_duplicate_filename.pbrt",
             })
        {
            auto rejected = false;
            try
            {
                (void)PbrtParser::parse(path);
            }
            catch (const std::runtime_error& error)
            {
                auto message = std::string{error.what()};
                rejected     = message.find(path) != std::string::npos;
            }
            expect(rejected) << "invalid plymesh filename should produce a source-located parse error";
        }
    };

    "parse_distant_light"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/distant_basic.pbrt");
        expect(scene.distant_lights.size() == 1u);
        expect(scene.infinite_lights.empty());
        if (scene.distant_lights.empty())
        {
            return;
        }
        auto&& light = scene.distant_lights.front();
        expect(is_near(light.L.x, 8.0f));
        expect(is_near(light.L.y, 8.0f));
        expect(is_near(light.L.z, 8.0f));
        expect(is_near(light.scale, 1.0f));
        expect(is_near(light.from.z, 1.0f));
        expect(is_near(light.to.x, 0.0f));
        expect(is_near(light.pbrt_transform[2u], 1.0f));
        expect(is_near(light.pbrt_transform[8u], -1.0f));
    };

    "parse_point_light"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/point_basic.pbrt");
        expect(scene.point_lights.size() == 1u);
        if (scene.point_lights.empty()) { return; }
        auto&& light = scene.point_lights.front();
        expect(is_near(light.I.x, 2.0f));
        expect(is_near(light.I.y, 3.0f));
        expect(is_near(light.I.z, 4.0f));
        expect(is_near(light.scale, 2.0f));
        expect(is_near(light.from.x, 1.0f));
        expect(is_near(light.pbrt_transform[3u], 1.0f));
        expect(is_near(light.pbrt_transform[7u], 2.0f));
        expect(is_near(light.pbrt_transform[11u], 3.0f));

        auto defaults = PbrtParser::parse("test/scenes/point_default.pbrt");
        expect(defaults.point_lights.size() == 1u);
        if (!defaults.point_lights.empty())
        {
            auto&& point = defaults.point_lights.front();
            expect(is_near(point.I.x, 1.0f));
            expect(is_near(point.scale, 1.0f));
            expect(is_near(point.from.x, 0.0f));
        }
    };

    "reject_invalid_point_lights"_test = []
    {
        for (auto path : {
                 "test/scenes/point_invalid_scale.pbrt",
                 "test/scenes/point_invalid_intensity.pbrt",
                 "test/scenes/point_nonfinite_from.pbrt",
                 "test/scenes/point_duplicate_parameter.pbrt",
                 "test/scenes/point_unsupported_radius.pbrt",
                 "test/scenes/point_unsupported_power.pbrt",
             })
        {
            expect(parse_error_contains(path, {path, "point"}));
        }
    };

    "reject_invalid_distant_lights"_test = []
    {
        for (auto path : {
                 "test/scenes/distant_invalid_direction.pbrt",
                 "test/scenes/distant_invalid_scale.pbrt",
                 "test/scenes/distant_invalid_radiance.pbrt",
                 "test/scenes/distant_nonfinite_illuminance.pbrt",
                 "test/scenes/distant_duplicate_parameter.pbrt",
             })
        {
            expect(parse_error_contains(path, {path, "LightSource"}));
        }
    };

    "parse_uniform_infinite_light"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/infinite_uniform.pbrt");
        expect(scene.infinite_lights.size() == 1u);
        expect(scene.distant_lights.empty());
        if (scene.infinite_lights.empty())
        {
            return;
        }
        auto&& light = scene.infinite_lights.front();
        expect(light.L.has_value());
        expect(light.filename.empty());
        expect(light.illuminance.has_value());
        if (light.L)
        {
            expect(is_near(light.L->x, 0.8f));
            expect(is_near(light.L->y, 0.9f));
            expect(is_near(light.L->z, 1.0f));
        }
        expect(is_near(light.scale, 2.0f));
        if (light.illuminance)
        {
            expect(is_near(*light.illuminance, 0.05f));
        }

        auto defaults = PbrtParser::parse("test/scenes/infinite_missing_filename.pbrt");
        expect(defaults.infinite_lights.size() == 1u);
        if (!defaults.infinite_lights.empty())
        {
            auto&& default_light = defaults.infinite_lights.front();
            expect(!default_light.L.has_value());
            expect(default_light.filename.empty());
            expect(!default_light.illuminance.has_value());
            expect(is_near(default_light.scale, 1.0f));
        }
    };

    "parse_multiple_environment_lights"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/multiple_environment_lights.pbrt");
        expect(scene.infinite_lights.size() == 1u);
        expect(scene.distant_lights.size() == 2u);
        if (scene.distant_lights.size() == 2u)
        {
            expect(scene.distant_lights[0u].illuminance.has_value());
            if (scene.distant_lights[0u].illuminance)
            {
                expect(is_near(*scene.distant_lights[0u].illuminance, 3.0f));
            }
            expect(is_near(scene.distant_lights[1u].from.y, 1.0f));
        }
    };

    "reject_invalid_infinite_lights"_test = []
    {
        for (auto path : {
                 "test/scenes/infinite_invalid_scale.pbrt",
                 "test/scenes/infinite_invalid_radiance.pbrt",
                 "test/scenes/infinite_nonfinite_illuminance.pbrt",
                 "test/scenes/infinite_image_illuminance.pbrt",
                 "test/scenes/infinite_l_and_filename.pbrt",
                 "test/scenes/infinite_duplicate_parameter.pbrt",
                 "test/scenes/infinite_unknown_type.pbrt",
                 "test/scenes/infinite_unsupported_parameter.pbrt",
             })
        {
            expect(parse_error_contains(path, {path, "LightSource"}));
        }
    };

    "parse_sppm_integrator"_test = []
    {
        auto scene = PbrtParser::parse("test/scenes/sppm_basic.pbrt");
        expect(scene.integrator.type == IntegratorDesc::Type::SPPM);
        expect(scene.integrator.max_depth == 5u);
        expect(scene.integrator.photons_per_iteration == 50000u);
        expect(is_near(scene.integrator.radius, 0.5f));
    };
    return 0;
}();

} // namespace

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
