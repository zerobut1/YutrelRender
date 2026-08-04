#include "pbrt_importer.h"

#include "pbrt_parser.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include "base/film.h"
#include "base/integrator.h"
#include "cameras/pinhole.h"
#include "environments/distant.h"
#include "environments/grouped.h"
#include "environments/null.h"
#include "environments/pbrt_equal_area.h"
#include "environments/uniform.h"
#include "filters/gaussian.h"
#include "filters/triangle.h"
#include "integrators/path.h"
#include "integrators/sppm.h"
#include "integrators/vol_path.h"
#include "lights/diffuse.h"
#include "lights/point.h"
#include "media/homogeneous.h"
#include "samplers/independent.h"
#include "samplers/sobol.h"
#include "samplers/zsobol.h"
#include "scene/scene_spec_builder.h"
#include "shapes/inline_mesh.h"
#include "shapes/mesh.h"
#include "shapes/sphere.h"
#include "spectrum/hero.h"
#include "surfaces/coated_diffuse.h"
#include "surfaces/dielectric.h"
#include "surfaces/diffuse.h"
#include "surfaces/null.h"
#include "surfaces/openpbr.h"
#include "surfaces/opacity.h"
#include "textures/checker_board.h"
#include "textures/constant.h"
#include "textures/image.h"
#include "textures/scale.h"

namespace Yutrel
{
static_assert(ShapeDesc::sphere_default_subdivision == Sphere::default_subdivision);
static_assert(ShapeDesc::sphere_max_subdivision == Sphere::max_subdivision);

namespace
{

[[nodiscard]] float pbrt_vertical_fov(float fov, uint2 resolution) noexcept
{
    if (resolution.x >= resolution.y)
    {
        return fov;
    }
    constexpr auto degrees_to_radians = 0.01745329251994329577f;
    constexpr auto radians_to_degrees = 57.295779513082320876f;
    auto aspect_scale                 = static_cast<float>(resolution.y) /
                                        static_cast<float>(resolution.x);
    return 2.0f * std::atan(std::tan(0.5f * fov * degrees_to_radians) * aspect_scale) *
           radians_to_degrees;
}

[[noreturn]] void fail(luisa::string message)
{
    throw std::runtime_error{message.c_str()};
}

[[noreturn]] void fail(const SourceLocation& source, luisa::string message)
{
    fail(luisa::format("{}: {}", format_source_location(source), message));
}

struct ParameterKey
{
    luisa::string_view type;
    luisa::string_view name;
};

void validate_parameters(
    luisa::span<const RawParameter> parameters,
    luisa::string_view owner,
    luisa::span<const ParameterKey> allowed)
{
    for (auto i = 0u; i < parameters.size(); i++)
    {
        auto&& parameter  = parameters[i];
        auto name_known   = false;
        auto type_allowed = false;
        luisa::string expected_types;
        for (auto&& key : allowed)
        {
            if (key.name != parameter.name)
            {
                continue;
            }
            name_known = true;
            type_allowed |= key.type == parameter.type;
            if (!expected_types.empty())
            {
                expected_types.append(" or ");
            }
            expected_types.append(luisa::format("'{}'", key.type));
        }
        if (!name_known)
        {
            fail(parameter.source, luisa::format("PBRT {} has unsupported parameter '{} {}'.", owner, parameter.type, parameter.name));
        }
        if (!type_allowed)
        {
            fail(parameter.source, luisa::format("PBRT {} parameter '{}' has type '{}'; expected {}.", owner, parameter.name, parameter.type, expected_types));
        }
        for (auto j = 0u; j < i; j++)
        {
            if (parameters[j].name == parameter.name)
            {
                fail(parameter.source, luisa::format("PBRT {} has duplicate parameter '{} {}'.", owner, parameter.type, parameter.name));
            }
        }
    }
}

template <size_t N>
void validate_parameters(
    luisa::span<const RawParameter> parameters,
    luisa::string_view owner,
    const std::array<ParameterKey, N>& allowed)
{
    validate_parameters(parameters, owner, luisa::span<const ParameterKey>{allowed.data(), allowed.size()});
}

[[nodiscard]] luisa::string_view sampler_name(SamplerDesc::Type type) noexcept
{
    switch (type)
    {
    case SamplerDesc::Type::Independent:
        return "independent";
    case SamplerDesc::Type::Halton:
        return "halton";
    case SamplerDesc::Type::Sobol:
        return "sobol";
    case SamplerDesc::Type::ZSobol:
        return "zsobol";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view texture_type_name(TextureDesc::Type type) noexcept
{
    switch (type)
    {
    case TextureDesc::Type::ImageMap:
        return "imagemap";
    case TextureDesc::Type::Constant:
        return "constant";
    case TextureDesc::Type::Scale:
        return "scale";
    case TextureDesc::Type::Checkerboard:
        return "checkerboard";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view shape_type_name(ShapeDesc::Type type) noexcept
{
    switch (type)
    {
    case ShapeDesc::Type::TriangleMesh:
        return "trianglemesh";
    case ShapeDesc::Type::PlyMesh:
        return "plymesh";
    case ShapeDesc::Type::Sphere:
        return "sphere";
    }
    return "unknown";
}

[[nodiscard]] luisa::string format_matrix(const Matrix4& m)
{
    return luisa::format(
        "[{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}]",
        m[0u],
        m[1u],
        m[2u],
        m[3u],
        m[4u],
        m[5u],
        m[6u],
        m[7u],
        m[8u],
        m[9u],
        m[10u],
        m[11u],
        m[12u],
        m[13u],
        m[14u],
        m[15u]);
}

void validate_material(const MaterialDesc& material, luisa::string_view owner, bool named)
{
    static constexpr std::array diffuse_inline_allowed{
        ParameterKey{"rgb", "reflectance"},
        ParameterKey{"texture", "reflectance"},
    };
    static constexpr std::array diffuse_named_allowed{
        ParameterKey{"string", "type"},
        ParameterKey{"rgb", "reflectance"},
        ParameterKey{"texture", "reflectance"},
    };
    static constexpr std::array coated_inline_allowed{
        ParameterKey{"rgb", "reflectance"},
        ParameterKey{"texture", "reflectance"},
        ParameterKey{"float", "roughness"},
        ParameterKey{"texture", "roughness"},
        ParameterKey{"float", "uroughness"},
        ParameterKey{"texture", "uroughness"},
        ParameterKey{"float", "vroughness"},
        ParameterKey{"texture", "vroughness"},
        ParameterKey{"float", "thickness"},
        ParameterKey{"texture", "thickness"},
        ParameterKey{"rgb", "albedo"},
        ParameterKey{"texture", "albedo"},
        ParameterKey{"float", "g"},
        ParameterKey{"texture", "g"},
        ParameterKey{"float", "eta"},
        ParameterKey{"texture", "eta"},
        ParameterKey{"bool", "remaproughness"},
        ParameterKey{"integer", "maxdepth"},
        ParameterKey{"integer", "nsamples"},
    };
    static constexpr std::array coated_named_allowed{
        ParameterKey{"string", "type"},
        ParameterKey{"rgb", "reflectance"},
        ParameterKey{"texture", "reflectance"},
        ParameterKey{"float", "roughness"},
        ParameterKey{"texture", "roughness"},
        ParameterKey{"float", "uroughness"},
        ParameterKey{"texture", "uroughness"},
        ParameterKey{"float", "vroughness"},
        ParameterKey{"texture", "vroughness"},
        ParameterKey{"float", "thickness"},
        ParameterKey{"texture", "thickness"},
        ParameterKey{"rgb", "albedo"},
        ParameterKey{"texture", "albedo"},
        ParameterKey{"float", "g"},
        ParameterKey{"texture", "g"},
        ParameterKey{"float", "eta"},
        ParameterKey{"texture", "eta"},
        ParameterKey{"bool", "remaproughness"},
        ParameterKey{"integer", "maxdepth"},
        ParameterKey{"integer", "nsamples"},
    };
    static constexpr std::array dielectric_inline_allowed{
        ParameterKey{"float", "roughness"},
        ParameterKey{"texture", "roughness"},
        ParameterKey{"float", "uroughness"},
        ParameterKey{"texture", "uroughness"},
        ParameterKey{"float", "vroughness"},
        ParameterKey{"texture", "vroughness"},
        ParameterKey{"float", "eta"},
        ParameterKey{"texture", "eta"},
        ParameterKey{"spectrum", "eta"},
        ParameterKey{"bool", "remaproughness"},
    };
    static constexpr std::array dielectric_named_allowed{
        ParameterKey{"string", "type"},
        ParameterKey{"float", "roughness"},
        ParameterKey{"texture", "roughness"},
        ParameterKey{"float", "uroughness"},
        ParameterKey{"texture", "uroughness"},
        ParameterKey{"float", "vroughness"},
        ParameterKey{"texture", "vroughness"},
        ParameterKey{"float", "eta"},
        ParameterKey{"texture", "eta"},
        ParameterKey{"spectrum", "eta"},
        ParameterKey{"bool", "remaproughness"},
    };
    static constexpr std::array interface_named_allowed{ParameterKey{"string", "type"}};
    static constexpr std::array openpbr_inline_allowed{
        ParameterKey{"float", "base_weight"},
        ParameterKey{"texture", "base_weight"},
        ParameterKey{"rgb", "base_color"},
        ParameterKey{"texture", "base_color"},
        ParameterKey{"float", "base_metalness"},
        ParameterKey{"texture", "base_metalness"},
        ParameterKey{"float", "base_diffuse_roughness"},
        ParameterKey{"texture", "base_diffuse_roughness"},
        ParameterKey{"float", "specular_weight"},
        ParameterKey{"texture", "specular_weight"},
        ParameterKey{"rgb", "specular_color"},
        ParameterKey{"texture", "specular_color"},
        ParameterKey{"float", "specular_roughness"},
        ParameterKey{"texture", "specular_roughness"},
        ParameterKey{"float", "specular_roughness_anisotropy"},
        ParameterKey{"texture", "specular_roughness_anisotropy"},
        ParameterKey{"float", "specular_ior"},
        ParameterKey{"texture", "specular_ior"},
    };
    static constexpr std::array openpbr_named_allowed{
        ParameterKey{"string", "type"},
        ParameterKey{"float", "base_weight"},
        ParameterKey{"texture", "base_weight"},
        ParameterKey{"rgb", "base_color"},
        ParameterKey{"texture", "base_color"},
        ParameterKey{"float", "base_metalness"},
        ParameterKey{"texture", "base_metalness"},
        ParameterKey{"float", "base_diffuse_roughness"},
        ParameterKey{"texture", "base_diffuse_roughness"},
        ParameterKey{"float", "specular_weight"},
        ParameterKey{"texture", "specular_weight"},
        ParameterKey{"rgb", "specular_color"},
        ParameterKey{"texture", "specular_color"},
        ParameterKey{"float", "specular_roughness"},
        ParameterKey{"texture", "specular_roughness"},
        ParameterKey{"float", "specular_roughness_anisotropy"},
        ParameterKey{"texture", "specular_roughness_anisotropy"},
        ParameterKey{"float", "specular_ior"},
        ParameterKey{"texture", "specular_ior"},
    };
    if (material.type == MaterialDesc::Type::Diffuse)
    {
        validate_parameters(material.parameters, owner, named ? luisa::span<const ParameterKey>{diffuse_named_allowed} : luisa::span<const ParameterKey>{diffuse_inline_allowed});
        return;
    }
    if (material.type == MaterialDesc::Type::CoatedDiffuse)
    {
        validate_parameters(material.parameters, owner, named ? luisa::span<const ParameterKey>{coated_named_allowed} : luisa::span<const ParameterKey>{coated_inline_allowed});
        return;
    }
    if (material.type == MaterialDesc::Type::Dielectric)
    {
        validate_parameters(material.parameters, owner, named ? luisa::span<const ParameterKey>{dielectric_named_allowed} : luisa::span<const ParameterKey>{dielectric_inline_allowed});
        return;
    }
    if (material.type == MaterialDesc::Type::OpenPBR)
    {
        validate_parameters(material.parameters, owner,
                            named ? luisa::span<const ParameterKey>{openpbr_named_allowed}
                                  : luisa::span<const ParameterKey>{openpbr_inline_allowed});
        return;
    }
    if (named)
    {
        validate_parameters(material.parameters, owner, interface_named_allowed);
    }
    else
    {
        validate_parameters(material.parameters, owner, luisa::span<const ParameterKey>{});
    }
}

void validate_pbrt_scene(const PbrtScene& scene)
{
    if (scene.integrator.type == IntegratorDesc::Type::SPPM)
    {
        static constexpr std::array sppm_allowed{
            ParameterKey{"integer", "maxdepth"},
            ParameterKey{"integer", "photonsperiteration"},
            ParameterKey{"float", "radius"},
        };
        validate_parameters(scene.integrator.parameters, "Integrator 'sppm'", sppm_allowed);
    }
    else
    {
        static constexpr std::array integrator_allowed{ParameterKey{"integer", "maxdepth"}};
        validate_parameters(scene.integrator.parameters, scene.integrator.type == IntegratorDesc::Type::Path ? "Integrator 'path'" : "Integrator 'volpath'", integrator_allowed);
    }

    if (scene.sampler.type == SamplerDesc::Type::Halton)
    {
        fail(scene.sampler.source, luisa::format("PBRT Sampler '{}' is not implemented by Yutrel.", sampler_name(scene.sampler.type)));
    }
    if (scene.sampler.type == SamplerDesc::Type::ZSobol)
    {
        if (scene.sampler.pixel_samples == 0u)
        {
            fail(scene.sampler.source, "ZSobol pixel sample count must be greater than zero.");
        }
        if (!std::has_single_bit(scene.sampler.pixel_samples))
        {
            fail(scene.sampler.source, "ZSobol pixel sample count must be a power of two.");
        }
    }
    static constexpr std::array independent_allowed{
        ParameterKey{"integer", "pixelsamples"},
        ParameterKey{"integer", "seed"},
    };
    static constexpr std::array sobol_allowed{
        ParameterKey{"integer", "pixelsamples"},
        ParameterKey{"integer", "seed"},
        ParameterKey{"string", "randomization"},
    };
    if (scene.sampler.type == SamplerDesc::Type::Independent)
    {
        validate_parameters(scene.sampler.parameters, "Sampler 'independent'", independent_allowed);
    }
    else if (scene.sampler.type == SamplerDesc::Type::Sobol)
    {
        validate_parameters(scene.sampler.parameters, "Sampler 'sobol'", sobol_allowed);
    }
    else
    {
        validate_parameters(scene.sampler.parameters, "Sampler 'zsobol'", sobol_allowed);
    }

    static constexpr std::array triangle_filter_allowed{
        ParameterKey{"float", "xradius"},
        ParameterKey{"float", "yradius"},
    };
    static constexpr std::array gaussian_filter_allowed{
        ParameterKey{"float", "xradius"},
        ParameterKey{"float", "yradius"},
        ParameterKey{"float", "sigma"},
    };
    if (scene.filter.type == FilterDesc::Type::Gaussian)
    {
        validate_parameters(scene.filter.parameters, "PixelFilter 'gaussian'", gaussian_filter_allowed);
    }
    else
    {
        validate_parameters(scene.filter.parameters, "PixelFilter 'triangle'", triangle_filter_allowed);
    }

    static constexpr std::array film_allowed{
        ParameterKey{"integer", "xresolution"},
        ParameterKey{"integer", "yresolution"},
        ParameterKey{"float", "iso"},
        ParameterKey{"float", "maxcomponentvalue"},
        ParameterKey{"string", "filename"},
    };
    validate_parameters(scene.film.parameters, "Film 'rgb'", film_allowed);

    static constexpr std::array camera_allowed{
        ParameterKey{"float", "fov"},
        ParameterKey{"float", "shutteropen"},
        ParameterKey{"float", "shutterclose"},
    };
    validate_parameters(scene.camera.parameters, "Camera 'perspective'", camera_allowed);

    for (auto&& texture : scene.textures)
    {
        auto owner = luisa::format("Texture '{}' ({})", texture.name, texture_type_name(texture.type));
        switch (texture.type)
        {
        case TextureDesc::Type::ImageMap:
        {
            static constexpr std::array allowed{
                ParameterKey{"string", "filename"},
                ParameterKey{"string", "filter"},
                ParameterKey{"string", "encoding"},
                ParameterKey{"float", "uscale"},
                ParameterKey{"float", "vscale"},
                ParameterKey{"float", "scale"},
            };
            validate_parameters(texture.parameters, owner, allowed);
            break;
        }
        case TextureDesc::Type::Constant:
        {
            static constexpr std::array allowed{ParameterKey{"float", "value"}};
            validate_parameters(texture.parameters, owner, allowed);
            break;
        }
        case TextureDesc::Type::Scale:
        {
            static constexpr std::array allowed{
                ParameterKey{"texture", "tex"},
                ParameterKey{"texture", "scale"},
            };
            validate_parameters(texture.parameters, owner, allowed);
            break;
        }
        case TextureDesc::Type::Checkerboard:
        {
            static constexpr std::array float_allowed{
                ParameterKey{"texture", "tex1"},
                ParameterKey{"float", "tex1"},
                ParameterKey{"texture", "tex2"},
                ParameterKey{"float", "tex2"},
                ParameterKey{"float", "uscale"},
                ParameterKey{"float", "vscale"},
                ParameterKey{"integer", "dimension"},
                ParameterKey{"string", "mapping"},
            };
            static constexpr std::array spectrum_allowed{
                ParameterKey{"texture", "tex1"},
                ParameterKey{"rgb", "tex1"},
                ParameterKey{"texture", "tex2"},
                ParameterKey{"rgb", "tex2"},
                ParameterKey{"float", "uscale"},
                ParameterKey{"float", "vscale"},
                ParameterKey{"integer", "dimension"},
                ParameterKey{"string", "mapping"},
            };
            if (texture.value_type == TextureDesc::ValueType::Float)
            {
                validate_parameters(texture.parameters, owner, float_allowed);
            }
            else
            {
                validate_parameters(texture.parameters, owner, spectrum_allowed);
            }
            break;
        }
        }
    }

    luisa::vector<std::pair<luisa::string_view, const MaterialDesc*>> named_materials;
    named_materials.reserve(scene.named_materials.size());
    for (auto&& [name, material] : scene.named_materials)
    {
        named_materials.emplace_back(name, &material);
    }
    std::sort(named_materials.begin(), named_materials.end(), [](auto a, auto b) noexcept
    {
        return a.first < b.first;
    });
    for (auto&& [name, material] : named_materials)
    {
        validate_material(*material, luisa::format("MakeNamedMaterial '{}'", name), true);
    }
    for (auto i = 0u; i < scene.materials.size(); i++)
    {
        validate_material(scene.materials[i], luisa::format("inline Material #{}", i), false);
    }

    static constexpr std::array homogeneous_medium_allowed{
        ParameterKey{"string", "type"},
        ParameterKey{"rgb", "sigma_a"},
        ParameterKey{"rgb", "sigma_s"},
        ParameterKey{"float", "scale"},
        ParameterKey{"float", "g"},
    };
    for (auto&& [name, medium] : scene.named_media)
    {
        validate_parameters(medium.parameters, luisa::format("MakeNamedMedium '{}'", name), homogeneous_medium_allowed);
    }

    static constexpr std::array area_light_allowed{ParameterKey{"rgb", "L"}};
    static constexpr std::array point_light_allowed{
        ParameterKey{"rgb", "I"},
        ParameterKey{"float", "scale"},
        ParameterKey{"point3", "from"},
    };
    static constexpr std::array infinite_light_allowed{
        ParameterKey{"rgb", "L"},
        ParameterKey{"string", "filename"},
        ParameterKey{"float", "scale"},
        ParameterKey{"float", "illuminance"},
    };
    static constexpr std::array distant_light_allowed{
        ParameterKey{"rgb", "L"},
        ParameterKey{"float", "scale"},
        ParameterKey{"float", "illuminance"},
        ParameterKey{"point3", "from"},
        ParameterKey{"point3", "to"},
    };
    for (auto&& light : scene.point_lights)
    {
        validate_parameters(light.parameters, "LightSource 'point'", point_light_allowed);
        auto finite = [](float3 v) noexcept
        {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        };
        if (!finite(light.I) || light.I.x < 0.0f || light.I.y < 0.0f || light.I.z < 0.0f)
        {
            fail(light.source, "PBRT point LightSource intensity must be finite and non-negative.");
        }
        if (!std::isfinite(light.scale) || light.scale < 0.0f)
        {
            fail(light.source, "PBRT point LightSource scale must be finite and non-negative.");
        }
        if (!finite(light.from))
        {
            fail(light.source, "PBRT point LightSource position must be finite.");
        }
    }
    for (auto&& light : scene.infinite_lights)
    {
        validate_parameters(light.parameters, "LightSource 'infinite'", infinite_light_allowed);
        auto finite = [](float3 v) noexcept
        {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        };
        if (light.L && (!finite(*light.L) || light.L->x < 0.0f || light.L->y < 0.0f || light.L->z < 0.0f))
        {
            fail(light.source, "PBRT infinite LightSource radiance must be finite and non-negative.");
        }
        if (!std::isfinite(light.scale) || light.scale < 0.0f)
        {
            fail(light.source, "PBRT infinite LightSource scale must be finite and non-negative.");
        }
        if (light.illuminance && !std::isfinite(*light.illuminance))
        {
            fail(light.source, "PBRT infinite LightSource illuminance must be finite.");
        }
        if (light.L && !light.filename.empty())
        {
            fail(light.source, "PBRT infinite LightSource cannot specify both L and filename.");
        }
        if (!light.filename.empty() && light.illuminance)
        {
            fail(light.source, "PBRT image infinite LightSource does not support illuminance.");
        }
    }
    for (auto&& light : scene.distant_lights)
    {
        validate_parameters(light.parameters, "LightSource 'distant'", distant_light_allowed);
        auto finite = [](float3 v) noexcept
        {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        };
        if (!finite(light.L) || light.L.x < 0.0f || light.L.y < 0.0f || light.L.z < 0.0f)
        {
            fail(light.source, "PBRT distant LightSource radiance must be finite and non-negative.");
        }
        if (!std::isfinite(light.scale) || light.scale < 0.0f)
        {
            fail(light.source, "PBRT distant LightSource scale must be finite and non-negative.");
        }
        if (light.illuminance && !std::isfinite(*light.illuminance))
        {
            fail(light.source, "PBRT distant LightSource illuminance must be finite.");
        }
    }

    for (auto i = 0u; i < scene.shapes.size(); i++)
    {
        auto&& shape = scene.shapes[i];
        auto owner   = luisa::format("Shape #{} ({})", i, shape_type_name(shape.type));
        if (!std::isfinite(shape.alpha))
        {
            fail(shape.source, luisa::format("PBRT {} alpha must be finite.", owner));
        }
        switch (shape.type)
        {
        case ShapeDesc::Type::TriangleMesh:
        {
            static constexpr std::array allowed{
                ParameterKey{"point3", "P"},
                ParameterKey{"normal", "N"},
                ParameterKey{"point2", "uv"},
                ParameterKey{"integer", "indices"},
                ParameterKey{"float", "alpha"},
                ParameterKey{"texture", "alpha"},
            };
            validate_parameters(shape.parameters, owner, allowed);
            break;
        }
        case ShapeDesc::Type::PlyMesh:
        {
            static constexpr std::array allowed{
                ParameterKey{"string", "filename"},
                ParameterKey{"float", "alpha"},
                ParameterKey{"texture", "alpha"},
            };
            validate_parameters(shape.parameters, owner, allowed);
            break;
        }
        case ShapeDesc::Type::Sphere:
        {
            static constexpr std::array allowed{
                ParameterKey{"float", "radius"},
                ParameterKey{"integer", "subdivision"},
                ParameterKey{"float", "alpha"},
                ParameterKey{"texture", "alpha"},
            };
            validate_parameters(shape.parameters, owner, allowed);
            break;
        }
        }
        if (shape.area_light)
        {
            validate_parameters(shape.area_light->parameters, luisa::format("AreaLightSource for Shape #{}", i), area_light_allowed);
        }
    }
}

[[nodiscard]] float& at(Matrix4& m, uint32_t row, uint32_t column) noexcept { return m[row * 4u + column]; }
[[nodiscard]] float at(const Matrix4& m, uint32_t row, uint32_t column) noexcept { return m[row * 4u + column]; }

[[nodiscard]] Matrix4 inverse(Matrix4 m, const SourceLocation& source)
{
    Matrix4 inv{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    for (auto column = 0u; column < 4u; column++)
    {
        auto pivot     = column;
        auto pivot_abs = std::abs(at(m, pivot, column));
        for (auto row = column + 1u; row < 4u; row++)
        {
            auto candidate_abs = std::abs(at(m, row, column));
            if (candidate_abs > pivot_abs)
            {
                pivot     = row;
                pivot_abs = candidate_abs;
            }
        }
        if (pivot_abs < 1e-8f)
        {
            fail(source, "PBRT camera transform is singular.");
        }
        if (pivot != column)
        {
            for (auto i = 0u; i < 4u; i++)
            {
                std::swap(at(m, column, i), at(m, pivot, i));
                std::swap(at(inv, column, i), at(inv, pivot, i));
            }
        }
        auto inv_pivot = 1.0f / at(m, column, column);
        for (auto i = 0u; i < 4u; i++)
        {
            at(m, column, i) *= inv_pivot;
            at(inv, column, i) *= inv_pivot;
        }
        for (auto row = 0u; row < 4u; row++)
        {
            if (row == column)
            {
                continue;
            }
            auto factor = at(m, row, column);
            for (auto i = 0u; i < 4u; i++)
            {
                at(m, row, i) -= factor * at(m, column, i);
                at(inv, row, i) -= factor * at(inv, column, i);
            }
        }
    }
    return inv;
}

[[nodiscard]] float3 transform_vector(const Matrix4& m, float3 v) noexcept
{
    return make_float3(
        at(m, 0u, 0u) * v.x + at(m, 0u, 1u) * v.y + at(m, 0u, 2u) * v.z,
        at(m, 1u, 0u) * v.x + at(m, 1u, 1u) * v.y + at(m, 1u, 2u) * v.z,
        at(m, 2u, 0u) * v.x + at(m, 2u, 1u) * v.y + at(m, 2u, 2u) * v.z);
}

[[nodiscard]] float3 transform_point(const Matrix4& m, float3 p) noexcept
{
    return make_float3(
        at(m, 0u, 0u) * p.x + at(m, 0u, 1u) * p.y + at(m, 0u, 2u) * p.z + at(m, 0u, 3u),
        at(m, 1u, 0u) * p.x + at(m, 1u, 1u) * p.y + at(m, 1u, 2u) * p.z + at(m, 1u, 3u),
        at(m, 2u, 0u) * p.x + at(m, 2u, 1u) * p.y + at(m, 2u, 2u) * p.z + at(m, 2u, 3u));
}

[[nodiscard]] float4x4 instance_transform(const std::array<float, 16u>& raw) noexcept
{
    return make_float4x4(
        make_float4(raw[0u], raw[4u], raw[8u], raw[12u]),
        make_float4(raw[1u], raw[5u], raw[9u], raw[13u]),
        make_float4(raw[2u], raw[6u], raw[10u], raw[14u]),
        make_float4(raw[3u], raw[7u], raw[11u], raw[15u]));
}

[[nodiscard]] float4x4 camera_to_world(
    const Matrix4& camera_from_world,
    const SourceLocation& source)
{
    for (auto value : camera_from_world)
    {
        if (!std::isfinite(value))
        {
            fail(source, "PBRT camera transform entries must be finite.");
        }
    }
    constexpr auto affine_epsilon = 1e-6f;
    if (std::abs(at(camera_from_world, 3u, 0u)) > affine_epsilon ||
        std::abs(at(camera_from_world, 3u, 1u)) > affine_epsilon ||
        std::abs(at(camera_from_world, 3u, 2u)) > affine_epsilon ||
        std::abs(at(camera_from_world, 3u, 3u) - 1.0f) > affine_epsilon)
    {
        fail(source, "PBRT camera transform must be affine.");
    }

    auto world_from_pbrt_camera = inverse(camera_from_world, source);
    // PBRT camera rays face +Z while Yutrel camera rays face -Z.
    // Right-multiply by diag(1, 1, -1, 1) by negating the third column.
    for (auto row = 0u; row < 4u; row++)
    {
        at(world_from_pbrt_camera, row, 2u) *= -1.0f;
    }
    return instance_transform(world_from_pbrt_camera);
}

[[nodiscard]] float3x3 environment_transform(const Matrix4& raw) noexcept
{
    return make_float3x3(
        make_float3(raw[0u], raw[4u], raw[8u]),
        make_float3(raw[1u], raw[5u], raw[9u]),
        make_float3(raw[2u], raw[6u], raw[10u]));
}

[[nodiscard]] bool is_exr_path(const std::filesystem::path& path)
{
    auto ext = path.extension().string();
    for (auto& c : ext)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".exr";
}

[[nodiscard]] bool is_png_path(const std::filesystem::path& path)
{
    auto ext = path.extension().string();
    for (auto& c : ext)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".png";
}

[[nodiscard]] Texture::Encoding resolve_image_encoding(TextureDesc::Encoding encoding, const std::filesystem::path& path) noexcept
{
    switch (encoding)
    {
    case TextureDesc::Encoding::Automatic:
        return is_png_path(path) ? Texture::Encoding::SRGB : Texture::Encoding::LINEAR;
    case TextureDesc::Encoding::Linear:
        return Texture::Encoding::LINEAR;
    case TextureDesc::Encoding::SRGB:
        return Texture::Encoding::SRGB;
    }
    return Texture::Encoding::LINEAR;
}

[[nodiscard]] luisa::string_view image_encoding_name(Texture::Encoding encoding) noexcept
{
    switch (encoding)
    {
    case Texture::Encoding::LINEAR:
        return "linear";
    case Texture::Encoding::SRGB:
        return "sRGB";
    case Texture::Encoding::GAMMA:
        return "gamma";
    }
    return "unknown";
}

[[nodiscard]] std::filesystem::path resolve_relative_to_scene(const std::filesystem::path& scene_path, const std::filesystem::path& path)
{
    if (path.empty())
    {
        return "pbrt.exr";
    }
    if (path.is_absolute())
    {
        return path;
    }
    return std::filesystem::absolute(scene_path.parent_path() / path);
}

} // namespace

SceneSpec PbrtImporter::import(
    const std::filesystem::path& path,
    PbrtImportOptions options)
{
    auto scene = PbrtParser::parse(path);
    if (options.spp)
    {
        scene.sampler.pixel_samples = *options.spp;
    }
    if (options.seed)
    {
        scene.sampler.seed = *options.seed;
    }
    if (options.resolution)
    {
        scene.film.resolution = *options.resolution;
    }
    if (options.output)
    {
        scene.film.filename = std::move(*options.output);
    }
    if (options.world_up)
    {
        scene.camera.world_up = *options.world_up;
    }
    return import(std::move(scene));
}

SceneSpec PbrtImporter::import(PbrtScene scene)
{
    Clock import_clock;
    validate_pbrt_scene(scene);
    SceneSpecBuilder builder;

    auto environment = [&]() -> EnvironmentRef
    {
        luisa::vector<EnvironmentRef> environments;
        environments.reserve(scene.infinite_lights.size() + scene.distant_lights.size());

        for (auto&& desc : scene.infinite_lights)
        {
            if (desc.filename.empty())
            {
                auto L       = desc.L.value_or(make_float3(1.0f));
                auto texture = builder.add_anonymous_texture<ConstantTextureSpec>(
                    desc.source,
                    make_float4(L, 1.0f));
                // Yutrel's illuminant decoding and spectrum-to-RGB conversion already
                // normalize D65. PBRT performs the corresponding normalization in the
                // light because its PixelSensor integrates the spectrum without the
                // CIE Y divisor; applying it here would make the result too dark.
                auto scale = desc.scale;
                if (desc.illuminance && *desc.illuminance > 0.0f)
                {
                    scale *= *desc.illuminance * inv_pi;
                }
                environments.emplace_back(builder.add_anonymous_environment<UniformEnvironmentSpec>(
                    desc.source,
                    texture,
                    scale));
                continue;
            }
            auto path    = resolve_relative_to_scene(desc.source.file, desc.filename);
            auto texture = builder.add_anonymous_texture<ImageTextureSpec>(
                desc.source,
                std::move(path),
                TextureSampler::point_edge(),
                Texture::Encoding::LINEAR);
            environments.emplace_back(builder.add_anonymous_environment<PBRTEqualAreaEnvironmentSpec>(
                desc.source,
                texture,
                desc.scale,
                environment_transform(desc.pbrt_transform)));
        }

        for (auto&& desc : scene.distant_lights)
        {
            auto direction      = transform_vector(desc.pbrt_transform, desc.from - desc.to);
            auto length_squared = dot(direction, direction);
            if (!std::isfinite(length_squared) || length_squared < 1e-16f)
            {
                fail(desc.source, "PBRT distant LightSource transform produced a non-finite or zero-length direction.");
            }
            direction *= 1.0f / std::sqrt(length_squared);
            auto texture = builder.add_anonymous_texture<ConstantTextureSpec>(
                desc.source,
                make_float4(desc.L, 1.0f));
            auto scale = desc.scale;
            if (desc.illuminance && *desc.illuminance > 0.0f)
            {
                scale *= *desc.illuminance;
            }
            environments.emplace_back(builder.add_anonymous_environment<DistantEnvironmentSpec>(
                desc.source,
                texture,
                scale,
                direction));
        }

        if (environments.empty())
        {
            return builder.add_environment<NullEnvironmentSpec>(
                SpecMeta{.name = "pbrt_null_environment", .source = SourceLocation{scene.source_path}});
        }
        if (environments.size() == 1u)
        {
            return environments.front();
        }
        return builder.add_environment<GroupedEnvironmentSpec>(
            SpecMeta{.name = "pbrt_grouped_environment", .source = SourceLocation{scene.source_path}},
            std::move(environments));
    }();

    for (auto&& desc : scene.point_lights)
    {
        auto position = transform_point(desc.pbrt_transform, desc.from);
        auto intensity = builder.add_anonymous_texture<ConstantTextureSpec>(
            desc.source, make_float4(desc.I, 1.0f));
        auto light = builder.add_anonymous_light<PointLightSpec>(
            desc.source, intensity, position, desc.scale);
        builder.add_standalone_light(light);
    }

    luisa::unordered_map<luisa::string, const TextureDesc*> texture_declarations;
    texture_declarations.reserve(scene.textures.size());
    for (auto&& texture : scene.textures)
    {
        texture_declarations.emplace(texture.name, &texture);
    }

    for (auto&& texture : scene.textures)
    {
        auto meta = SpecMeta{.name = texture.name, .source = texture.source};
        if (texture.type == TextureDesc::Type::ImageMap)
        {
            auto path     = resolve_relative_to_scene(texture.source.file, texture.filename);
            auto encoding = resolve_image_encoding(texture.encoding, path);
            auto sampler  = texture.filter == TextureDesc::Filter::Point
                                ? TextureSampler::point_repeat()
                                : TextureSampler::linear_point_repeat();
            if (texture.image_scale == 1.0f)
            {
                (void)builder.add_texture<ImageTextureSpec>(
                    std::move(meta),
                    std::move(path),
                    sampler,
                    encoding,
                    texture.uv_scale);
            }
            else
            {
                auto image = builder.add_anonymous_texture<ImageTextureSpec>(
                    texture.source,
                    std::move(path),
                    sampler,
                    encoding,
                    texture.uv_scale);
                (void)builder.add_texture<ScaleTextureSpec>(
                    std::move(meta),
                    image,
                    make_float4(texture.image_scale),
                    make_float4(0.0f));
            }
            continue;
        }
        if (texture.type == TextureDesc::Type::Constant)
        {
            (void)builder.add_texture<ConstantTextureSpec>(
                std::move(meta),
                make_float4(texture.constant_value));
            continue;
        }
        if (texture.type == TextureDesc::Type::Scale)
        {
            auto base_iter = texture_declarations.find(texture.tex);
            if (base_iter == texture_declarations.end())
            {
                fail(luisa::format(
                    "PBRT scale texture '{}' references unknown base texture '{}' at {}.",
                    texture.name,
                    texture.tex,
                    format_source_location(texture.source)));
            }
            if (base_iter->second->value_type != TextureDesc::ValueType::Float)
            {
                fail(luisa::format(
                    "PBRT scale texture '{}' requires '{}' to be a float texture at {}.",
                    texture.name,
                    texture.tex,
                    format_source_location(texture.source)));
            }
            auto scale_iter = texture_declarations.find(texture.scale);
            if (scale_iter == texture_declarations.end())
            {
                fail(luisa::format(
                    "PBRT scale texture '{}' references unknown scale texture '{}' at {}.",
                    texture.name,
                    texture.scale,
                    format_source_location(texture.source)));
            }
            auto scale = scale_iter->second;
            if (scale->value_type != TextureDesc::ValueType::Float ||
                scale->type != TextureDesc::Type::Constant)
            {
                fail(luisa::format(
                    "PBRT scale texture '{}' requires '{}' to be a float constant texture; dynamic multiplication is unsupported at {}.",
                    texture.name,
                    texture.scale,
                    format_source_location(texture.source)));
            }
            auto base = builder.reference_texture(texture.tex, texture.source);
            (void)builder.add_texture<ScaleTextureSpec>(
                std::move(meta),
                base,
                make_float4(scale->constant_value),
                make_float4(0.0f));
            continue;
        }
        if (texture.type == TextureDesc::Type::Checkerboard)
        {
            auto resolve_input = [&](const TextureInputDesc& input, luisa::string_view parameter) -> TextureRef
            {
                if (!input.texture)
                {
                    return builder.add_anonymous_texture<ConstantTextureSpec>(texture.source, input.constant);
                }
                auto iter = texture_declarations.find(*input.texture);
                if (iter == texture_declarations.end())
                {
                    fail(luisa::format(
                        "PBRT checkerboard texture '{}' references unknown {} texture '{}' at {}.",
                        texture.name,
                        parameter,
                        *input.texture,
                        format_source_location(texture.source)));
                }
                if (iter->second->value_type != texture.value_type)
                {
                    auto expected = texture.value_type == TextureDesc::ValueType::Float ? "float" : "spectrum";
                    fail(luisa::format(
                        "PBRT checkerboard texture '{}' requires {} '{}' to be a {} texture at {}.",
                        texture.name,
                        parameter,
                        *input.texture,
                        expected,
                        format_source_location(texture.source)));
                }
                return builder.reference_texture(*input.texture, texture.source);
            };
            auto tex1 = resolve_input(texture.tex1, "tex1");
            auto tex2 = resolve_input(texture.tex2, "tex2");
            (void)builder.add_texture<CheckerBoardTextureSpec>(
                std::move(meta),
                texture.uv_scale,
                tex1,
                tex2);
            continue;
        }
        fail(luisa::format(
            "Unsupported PBRT texture '{}' at {}.",
            texture.name,
            format_source_location(texture.source)));
    }

    auto make_material_surface = [&](const MaterialDesc& material, luisa::string_view name) -> SurfaceRef
    {
        auto add_constant_texture = [&](luisa::string_view parameter, float4 value) -> TextureRef
        {
            if (name.empty())
            {
                return builder.add_anonymous_texture<ConstantTextureSpec>(material.source, value);
            }
            return builder.add_texture<ConstantTextureSpec>(
                SpecMeta{.name = luisa::format("{}::{}", name, parameter), .source = material.source},
                value);
        };

        auto resolve_texture = [&](luisa::string_view parameter, float4 value, const luisa::optional<luisa::string>& texture) -> TextureRef
        {
            if (texture)
            {
                return builder.reference_texture(*texture, material.source);
            }
            return add_constant_texture(parameter, value);
        };

        if (material.type == MaterialDesc::Type::Interface)
        {
            if (name.empty())
            {
                return builder.add_anonymous_surface<NullSurfaceSpec>(material.source, true);
            }
            return builder.add_surface<NullSurfaceSpec>(
                SpecMeta{.name = luisa::string{name}, .source = material.source},
                true);
        }

        if (material.type == MaterialDesc::Type::Dielectric)
        {
            luisa::optional<TextureRef> eta;
            luisa::optional<CauchyEta> cauchy_eta;
            if (material.eta_spectrum)
            {
                if (*material.eta_spectrum != "glass-F11")
                {
                    fail(material.source, luisa::format(
                                              "Unsupported PBRT dielectric eta spectrum '{}'; only 'glass-F11' is supported.",
                                              *material.eta_spectrum));
                }
                cauchy_eta.emplace(glass_f11_cauchy_eta());
            }
            else
            {
                eta.emplace(resolve_texture("eta", make_float4(material.eta), material.eta_texture));
            }
            DielectricSurfaceParams params{
                .roughness       = luisa::nullopt,
                .u_roughness     = resolve_texture("uroughness", make_float4(material.u_roughness), material.u_roughness_texture),
                .v_roughness     = resolve_texture("vroughness", make_float4(material.v_roughness), material.v_roughness_texture),
                .eta             = eta,
                .cauchy_eta      = cauchy_eta,
                .remap_roughness = material.remap_roughness,
                .two_sided       = false,
            };
            if (name.empty())
            {
                return builder.add_anonymous_surface<DielectricSurfaceSpec>(material.source, std::move(params));
            }
            return builder.add_surface<DielectricSurfaceSpec>(
                SpecMeta{.name = luisa::string{name}, .source = material.source},
                std::move(params));
        }

        if (material.type == MaterialDesc::Type::OpenPBR)
        {
            OpenPBRSurfaceParams params{
                .base_weight = resolve_texture("base_weight", make_float4(material.base_weight), material.base_weight_texture),
                .base_color = resolve_texture("base_color", make_float4(material.base_color, 1.0f), material.base_color_texture),
                .base_metalness = resolve_texture("base_metalness", make_float4(material.base_metalness), material.base_metalness_texture),
                .base_diffuse_roughness = resolve_texture("base_diffuse_roughness", make_float4(material.base_diffuse_roughness), material.base_diffuse_roughness_texture),
                .specular_weight = resolve_texture("specular_weight", make_float4(material.specular_weight), material.specular_weight_texture),
                .specular_color = resolve_texture("specular_color", make_float4(material.specular_color, 1.0f), material.specular_color_texture),
                .specular_roughness = resolve_texture("specular_roughness", make_float4(material.specular_roughness), material.specular_roughness_texture),
                .specular_roughness_anisotropy = resolve_texture("specular_roughness_anisotropy", make_float4(material.specular_roughness_anisotropy), material.specular_roughness_anisotropy_texture),
                .specular_ior = resolve_texture("specular_ior", make_float4(material.specular_ior), material.specular_ior_texture),
                .two_sided = true,
            };
            if (name.empty())
            {
                return builder.add_anonymous_surface<OpenPBRSurfaceSpec>(material.source, std::move(params));
            }
            return builder.add_surface<OpenPBRSurfaceSpec>(
                SpecMeta{.name = luisa::string{name}, .source = material.source},
                std::move(params));
        }

        auto reflectance = resolve_texture(
            "reflectance",
            make_float4(material.reflectance.x, material.reflectance.y, material.reflectance.z, 1.0f),
            material.reflectance_texture);

        if (material.type == MaterialDesc::Type::Diffuse)
        {
            if (name.empty())
            {
                return builder.add_anonymous_surface<DiffuseSurfaceSpec>(material.source, reflectance, true);
            }
            return builder.add_surface<DiffuseSurfaceSpec>(
                SpecMeta{.name = luisa::string{name}, .source = material.source},
                reflectance,
                true);
        }

        CoatedDiffuseSurfaceParams params{
            .reflectance     = reflectance,
            .roughness       = luisa::nullopt,
            .u_roughness     = resolve_texture("uroughness", make_float4(material.u_roughness), material.u_roughness_texture),
            .v_roughness     = resolve_texture("vroughness", make_float4(material.v_roughness), material.v_roughness_texture),
            .thickness       = resolve_texture("thickness", make_float4(material.thickness), material.thickness_texture),
            .albedo          = resolve_texture("albedo", make_float4(material.albedo.x, material.albedo.y, material.albedo.z, 1.0f), material.albedo_texture),
            .g               = resolve_texture("g", make_float4(material.g), material.g_texture),
            .eta             = resolve_texture("eta", make_float4(material.eta), material.eta_texture),
            .remap_roughness = material.remap_roughness,
            .max_depth       = material.max_depth,
            .samples         = material.samples,
        };
        if (name.empty())
        {
            return builder.add_anonymous_surface<CoatedDiffuseSurfaceSpec>(material.source, std::move(params));
        }
        return builder.add_surface<CoatedDiffuseSurfaceSpec>(
            SpecMeta{.name = luisa::string{name}, .source = material.source},
            std::move(params));
    };

    for (auto& [name, material] : scene.named_materials)
    {
        (void)make_material_surface(material, name);
    }

    for (auto& [name, medium] : scene.named_media)
    {
        auto sigma_a = builder.add_texture<ConstantTextureSpec>(
            SpecMeta{.name = luisa::format("{}::sigma_a", name), .source = medium.source},
            make_float4(medium.sigma_a.x, medium.sigma_a.y, medium.sigma_a.z, 1.0f));
        auto sigma_s = builder.add_texture<ConstantTextureSpec>(
            SpecMeta{.name = luisa::format("{}::sigma_s", name), .source = medium.source},
            make_float4(medium.sigma_s.x, medium.sigma_s.y, medium.sigma_s.z, 1.0f));
        (void)builder.add_medium<HomogeneousMediumSpec>(
            SpecMeta{.name = name, .source = medium.source},
            HomogeneousMediumParams{.sigma_a = sigma_a, .sigma_s = sigma_s, .scale = medium.scale, .g = medium.g});
    }

    luisa::vector<SurfaceRef> inline_surface_refs;
    inline_surface_refs.reserve(scene.materials.size());
    for (auto& material : scene.materials)
    {
        inline_surface_refs.emplace_back(make_material_surface(material, {}));
    }

    luisa::optional<SurfaceRef> default_surface;
    auto get_default_surface = [&](const SourceLocation& source) -> SurfaceRef
    {
        if (!default_surface)
        {
            auto texture    = builder.add_anonymous_texture<ConstantTextureSpec>(source, make_float4(0.5f, 0.5f, 0.5f, 1.0f));
            default_surface = builder.add_anonymous_surface<DiffuseSurfaceSpec>(source, texture, true);
        }
        return *default_surface;
    };

    std::map<uint32_t, TextureRef> alpha_constant_cache;
    std::map<std::pair<SurfaceRef, TextureRef>, SurfaceRef> opacity_surface_cache;
    auto resolve_alpha_texture = [&](const ShapeDesc& shape) -> luisa::optional<TextureRef>
    {
        if (shape.alpha_texture)
        {
            auto iter = texture_declarations.find(*shape.alpha_texture);
            if (iter == texture_declarations.end())
            {
                fail(shape.source, luisa::format("PBRT shape alpha references undefined texture '{}'.", *shape.alpha_texture));
            }
            if (iter->second->value_type != TextureDesc::ValueType::Float)
            {
                fail(shape.source, luisa::format("PBRT shape alpha texture '{}' must be a float texture.", *shape.alpha_texture));
            }
            return builder.reference_texture(*shape.alpha_texture, shape.source);
        }
        if (shape.alpha >= 1.0f)
        {
            return luisa::nullopt;
        }
        auto bits = std::bit_cast<uint32_t>(shape.alpha);
        if (auto iter = alpha_constant_cache.find(bits); iter != alpha_constant_cache.end())
        {
            return iter->second;
        }
        auto texture = builder.add_anonymous_texture<ConstantTextureSpec>(
            shape.source,
            make_float4(shape.alpha));
        alpha_constant_cache.emplace(bits, texture);
        return texture;
    };

    luisa::vector<ShapeRef> inline_mesh_refs;
    inline_mesh_refs.reserve(scene.meshes.size());
    for (auto& mesh : scene.meshes)
    {
        inline_mesh_refs.emplace_back(builder.add_shape<InlineMeshShapeSpec>(
            SpecMeta{.name = luisa::format("mesh_{}", inline_mesh_refs.size()), .source = mesh.source},
            std::move(mesh.positions),
            std::move(mesh.normals),
            std::move(mesh.uvs),
            std::move(mesh.indices)));
    }

    for (auto shape_index = 0u; shape_index < scene.shapes.size(); shape_index++)
    {
        auto& shape    = scene.shapes[shape_index];
        auto shape_ref = [&]() -> ShapeRef
        {
            switch (shape.type)
            {
            case ShapeDesc::Type::TriangleMesh:
                if (!shape.mesh_index || *shape.mesh_index >= inline_mesh_refs.size())
                {
                    fail(luisa::format("PBRT shape references an out-of-range mesh at {}.", format_source_location(shape.source)));
                }
                return inline_mesh_refs[*shape.mesh_index];
            case ShapeDesc::Type::PlyMesh:
                if (!shape.filename || shape.filename->empty())
                {
                    fail(luisa::format("PBRT plymesh has no filename at {}.", format_source_location(shape.source)));
                }
                return builder.add_shape<MeshShapeSpec>(
                    SpecMeta{.name = luisa::format("plymesh_{}", shape_index), .source = shape.source},
                    resolve_relative_to_scene(scene.source_path, *shape.filename));
            case ShapeDesc::Type::Sphere:
                return builder.add_shape<SphereShapeSpec>(
                    SpecMeta{.name = luisa::format("sphere_{}", shape_index), .source = shape.source},
                    shape.radius,
                    shape.sphere_subdivision);
            }
            fail("Unsupported PBRT shape type.");
        }();
        auto surface = [&]() -> SurfaceRef
        {
            auto has_named  = !shape.material.named.empty();
            auto has_inline = shape.material.inline_index.has_value();
            if (has_named && has_inline)
            {
                fail(luisa::format("PBRT shape has both named and inline material bindings at {}.", format_source_location(shape.source)));
            }
            if (has_named)
            {
                return builder.reference_surface(shape.material.named, shape.source);
            }
            if (has_inline)
            {
                auto index = *shape.material.inline_index;
                if (index >= inline_surface_refs.size())
                {
                    fail(luisa::format(
                        "PBRT shape references out-of-range inline material {} at {}.",
                        index,
                        format_source_location(shape.source)));
                }
                return inline_surface_refs[index];
            }
            return get_default_surface(shape.source);
        }();
        if (auto alpha_texture = resolve_alpha_texture(shape))
        {
            auto key = std::pair{surface, *alpha_texture};
            if (auto iter = opacity_surface_cache.find(key); iter != opacity_surface_cache.end())
            {
                surface = iter->second;
            }
            else
            {
                auto opacity_surface = builder.add_anonymous_surface<OpacitySurfaceSpec>(
                    shape.source,
                    surface,
                    *alpha_texture);
                opacity_surface_cache.emplace(key, opacity_surface);
                surface = opacity_surface;
            }
        }
        luisa::optional<LightRef> light;
        if (shape.area_light)
        {
            if (shape.area_light->type != AreaLightDesc::Type::Diffuse)
            {
                fail("Unsupported PBRT area light type.");
            }
            auto emission = builder.add_anonymous_texture<ConstantTextureSpec>(
                shape.area_light->source,
                make_float4(shape.area_light->emission.x, shape.area_light->emission.y, shape.area_light->emission.z, 1.0f));
            light = builder.add_anonymous_light<DiffuseLightSpec>(shape.area_light->source, emission, 1.0f, false);
        }
        builder.add_instance(ShapeInstanceSpec{
            .source         = shape.source,
            .shape          = shape_ref,
            .surface        = surface,
            .light          = light,
            .inside_medium  = shape.medium_interface.inside.empty()
                                  ? luisa::nullopt
                                  : luisa::optional<MediumRef>{builder.reference_medium(shape.medium_interface.inside, shape.source)},
            .outside_medium = shape.medium_interface.outside.empty()
                                  ? luisa::nullopt
                                  : luisa::optional<MediumRef>{builder.reference_medium(shape.medium_interface.outside, shape.source)},
            .transform      = instance_transform(shape.pbrt_transform),
        });
    }

    auto camera_transform = camera_to_world(scene.camera.pbrt_transform, scene.camera.source);
    auto camera_world_up  = scene.camera.world_up;
    if (!camera_world_up)
    {
        camera_world_up = normalize(make_float3(camera_transform[1]));
        LUISA_WARNING(
            "PBRT camera has no world-up metadata; falling back to its local up axis for interactive navigation.");
    }
    auto filename = resolve_relative_to_scene(scene.source_path, scene.film.filename);
    if (!is_exr_path(filename))
    {
        fail(luisa::format("Yutrel only supports EXR film output, got '{}'.", filename.string()));
    }
    auto shutter_span  = make_float2(scene.camera.shutter_open, scene.camera.shutter_close);
    auto exposure_time = shutter_span.y - shutter_span.x;
    if (!std::isfinite(exposure_time) || exposure_time <= 0.0f)
    {
        fail("PBRT shutterclose must be greater than shutteropen.");
    }
    if (!std::isfinite(scene.film.iso) || scene.film.iso <= 0.0f)
    {
        fail("PBRT Film ISO must be finite and positive.");
    }
    if (std::isnan(scene.film.max_component_value) || scene.film.max_component_value <= 0.0f)
    {
        fail("PBRT Film maxcomponentvalue must be positive or infinity.");
    }
    auto imaging_ratio = exposure_time * scene.film.iso / 100.0f;
    if (!std::isfinite(imaging_ratio))
    {
        fail("PBRT Film exposure ratio is not finite.");
    }
    auto vertical_fov = pbrt_vertical_fov(scene.camera.fov, scene.film.resolution);
    auto camera       = builder.add_camera<PinholeCameraSpec>(
        SpecMeta{.name = "pbrt_camera", .source = scene.camera.source},
        camera_transform,
        *camera_world_up,
        shutter_span,
        0u,
        vertical_fov);
    auto film = builder.add_film<RGBFilmSpec>(
        SpecMeta{.name = "pbrt_film", .source = scene.film.source},
        scene.film.resolution,
        false,
        filename,
        imaging_ratio,
        scene.film.max_component_value);
    if (std::abs(scene.filter.radius.x - scene.filter.radius.y) > 1e-6f)
    {
        fail("PBRT pixel filter expects equal x/y radii for now.");
    }
    auto filter = [&]() -> FilterRef
    {
        switch (scene.filter.type)
        {
        case FilterDesc::Type::Triangle:
            return builder.add_filter<TriangleFilterSpec>(SpecMeta{.name = "pbrt_filter", .source = scene.filter.source}, scene.filter.radius.x);
        case FilterDesc::Type::Gaussian:
            return builder.add_filter<GaussianFilterSpec>(SpecMeta{.name = "pbrt_filter", .source = scene.filter.source}, scene.filter.radius.x, scene.filter.sigma);
        }
        fail("Unsupported PBRT pixel filter.");
    }();
    auto spectrum = builder.add_spectrum<HeroWavelengthSpectrumSpec>(SpecMeta{.name = "pbrt_spectrum", .source = SourceLocation{scene.source_path}});
    auto sampler  = [&]() -> SamplerRef
    {
        switch (scene.sampler.type)
        {
        case SamplerDesc::Type::Independent:
            return builder.add_sampler<IndependentSamplerSpec>(SpecMeta{.name = "pbrt_sampler", .source = scene.sampler.source}, scene.sampler.pixel_samples, scene.sampler.seed);
        case SamplerDesc::Type::Sobol:
            return builder.add_sampler<SobolSamplerSpec>(SpecMeta{.name = "pbrt_sampler", .source = scene.sampler.source}, scene.sampler.pixel_samples, scene.sampler.seed);
        case SamplerDesc::Type::ZSobol:
            return builder.add_sampler<ZSobolSamplerSpec>(SpecMeta{.name = "pbrt_sampler", .source = scene.sampler.source}, scene.sampler.pixel_samples, scene.sampler.seed);
        case SamplerDesc::Type::Halton:
            break;
        }
        fail("Unsupported PBRT sampler type.");
    }();
    auto integrator = [&]() {
        switch (scene.integrator.type)
        {
        case IntegratorDesc::Type::Path:
            return builder.add_integrator<PathIntegratorSpec>(SpecMeta{.name = "pbrt_integrator", .source = scene.integrator.source}, scene.integrator.max_depth);
        case IntegratorDesc::Type::VolPath:
            return builder.add_integrator<VolPathIntegratorSpec>(SpecMeta{.name = "pbrt_integrator", .source = scene.integrator.source}, scene.integrator.max_depth);
        case IntegratorDesc::Type::SPPM:
        {
            auto photons = scene.integrator.photons_per_iteration;
            if (photons == 0u) { photons = scene.film.resolution.x * scene.film.resolution.y; }
            return builder.add_integrator<SPPMIntegratorSpec>(SpecMeta{.name = "pbrt_integrator", .source = scene.integrator.source}, scene.integrator.max_depth, photons, scene.integrator.radius);
        }
        }
        fail("Unsupported PBRT integrator type.");
    }();
    builder.set_render(RenderSpec{
        .spectrum    = spectrum,
        .environment = environment,
        .camera      = camera,
        .film        = film,
        .filter      = filter,
        .sampler     = sampler,
        .integrator  = integrator,
    });
    auto result = builder.finish();

    auto randomization  = scene.sampler.type == SamplerDesc::Type::Sobol ||
                                  scene.sampler.type == SamplerDesc::Type::ZSobol
                              ? "fastowen"
                              : "n/a";
    auto filter_summary = scene.filter.type == FilterDesc::Type::Gaussian
                              ? luisa::format("gaussian, radius=({}, {}), sigma={}", scene.filter.radius.x, scene.filter.radius.y, scene.filter.sigma)
                              : luisa::format("triangle, radius=({}, {})", scene.filter.radius.x, scene.filter.radius.y);
    LUISA_INFO(
        "PBRT render config: integrator={}, max_depth={}, rr=pbrt-v4, light_sampler=yutrel-uniform; sampler={}, spp={}, seed={}, randomization={}; filter={}.",
        scene.integrator.type == IntegratorDesc::Type::Path ? "path" : (scene.integrator.type == IntegratorDesc::Type::SPPM ? "sppm" : "volpath"),
        scene.integrator.max_depth,
        sampler_name(scene.sampler.type),
        scene.sampler.pixel_samples,
        scene.sampler.seed,
        randomization,
        filter_summary);
    LUISA_INFO(
        "PBRT film: type=rgb, resolution={}x{}, ISO={}, imaging_ratio={}, output='{}'.",
        scene.film.resolution.x,
        scene.film.resolution.y,
        scene.film.iso,
        imaging_ratio,
        filename.string());
    LUISA_INFO(
        "PBRT camera: type=perspective, fov={}, vertical_fov={}, shutter=[{}, {}], camera_to_world_columns=[({}, {}, {}, {}), ({}, {}, {}, {}), ({}, {}, {}, {}), ({}, {}, {}, {})], linear_determinant={}.",
        scene.camera.fov,
        vertical_fov,
        scene.camera.shutter_open,
        scene.camera.shutter_close,
        camera_transform[0].x,
        camera_transform[0].y,
        camera_transform[0].z,
        camera_transform[0].w,
        camera_transform[1].x,
        camera_transform[1].y,
        camera_transform[1].z,
        camera_transform[1].w,
        camera_transform[2].x,
        camera_transform[2].y,
        camera_transform[2].z,
        camera_transform[2].w,
        camera_transform[3].x,
        camera_transform[3].y,
        camera_transform[3].z,
        camera_transform[3].w,
        camera_linear_determinant(camera_transform));
    if (scene.infinite_lights.empty() && scene.distant_lights.empty())
    {
        LUISA_INFO("PBRT environment: none.");
    }
    else
    {
        LUISA_INFO(
            "PBRT environments: infinite={}, distant={}, total={}.",
            scene.infinite_lights.size(),
            scene.distant_lights.size(),
            scene.infinite_lights.size() + scene.distant_lights.size());
    }
    for (auto&& light : scene.infinite_lights)
    {
        if (light.filename.empty())
        {
            auto L = light.L.value_or(make_float3(1.0f));
            LUISA_INFO(
                "PBRT environment: type=uniform-infinite, L=({}, {}, {}), scale={}, illuminance={}.",
                L.x,
                L.y,
                L.z,
                light.scale,
                light.illuminance.value_or(-1.0f));
        }
        else
        {
            LUISA_INFO(
                "PBRT environment: type=infinite, file='{}', scale={}.",
                resolve_relative_to_scene(light.source.file, light.filename).string(),
                light.scale);
        }
        LUISA_VERBOSE(
            "PBRT environment transform={}, source={}.",
            format_matrix(light.pbrt_transform),
            format_source_location(light.source));
    }
    for (auto&& light : scene.distant_lights)
    {
        LUISA_INFO(
            "PBRT environment: type=distant, L=({}, {}, {}), scale={}, illuminance={}, from=({}, {}, {}), to=({}, {}, {}).",
            light.L.x,
            light.L.y,
            light.L.z,
            light.scale,
            light.illuminance.value_or(-1.0f),
            light.from.x,
            light.from.y,
            light.from.z,
            light.to.x,
            light.to.y,
            light.to.z);
        LUISA_VERBOSE(
            "PBRT distant transform={}, source={}.",
            format_matrix(light.pbrt_transform),
            format_source_location(light.source));
    }
    auto area_light_count = static_cast<size_t>(std::count_if(
        scene.shapes.begin(),
        scene.shapes.end(),
        [](auto&& shape) noexcept
    {
        return shape.area_light.has_value();
    }));
    LUISA_INFO(
        "PBRT resources: textures={}, materials={} (named={}, inline={}), shapes={}, area_lights={}; SceneSpec textures={}, surfaces={}, lights={}, shapes={}, instances={}.",
        scene.textures.size(),
        scene.named_materials.size() + scene.materials.size(),
        scene.named_materials.size(),
        scene.materials.size(),
        scene.shapes.size(),
        area_light_count,
        result.textures().size(),
        result.surfaces().size(),
        result.lights().size(),
        result.shapes().size(),
        result.instances().size());

    for (auto i = 0u; i < scene.textures.size(); i++)
    {
        auto&& texture = scene.textures[i];
        switch (texture.type)
        {
        case TextureDesc::Type::ImageMap:
        {
            auto path     = resolve_relative_to_scene(texture.source.file, texture.filename);
            auto encoding = resolve_image_encoding(texture.encoding, path);
            LUISA_VERBOSE(
                "PBRT texture #{} '{}' type=imagemap, value_type={}, file='{}', filter={}, encoding={}, uv_scale=({}, {}), scale={}, source={}.",
                i,
                texture.name,
                texture.value_type == TextureDesc::ValueType::Float ? "float" : "spectrum",
                path.string(),
                texture.filter == TextureDesc::Filter::Point ? "point" : "bilinear",
                image_encoding_name(encoding),
                texture.uv_scale.x,
                texture.uv_scale.y,
                texture.image_scale,
                format_source_location(texture.source));
            break;
        }
        case TextureDesc::Type::Constant:
            LUISA_VERBOSE(
                "PBRT texture #{} '{}' type=constant, value={}, source={}.",
                i,
                texture.name,
                texture.constant_value,
                format_source_location(texture.source));
            break;
        case TextureDesc::Type::Scale:
            LUISA_VERBOSE(
                "PBRT texture #{} '{}' type=scale, tex='{}', scale='{}', source={}.",
                i,
                texture.name,
                texture.tex,
                texture.scale,
                format_source_location(texture.source));
            break;
        case TextureDesc::Type::Checkerboard:
            LUISA_VERBOSE(
                "PBRT texture #{} '{}' type=checkerboard, tex1={}, tex2={}, uv_scale=({}, {}), source={}.",
                i,
                texture.name,
                texture.tex1.texture ? *texture.tex1.texture : "<constant>",
                texture.tex2.texture ? *texture.tex2.texture : "<constant>",
                texture.uv_scale.x,
                texture.uv_scale.y,
                format_source_location(texture.source));
            break;
        }
    }

    luisa::vector<std::pair<luisa::string_view, const MaterialDesc*>> sorted_materials;
    sorted_materials.reserve(scene.named_materials.size());
    for (auto&& [name, material] : scene.named_materials)
    {
        sorted_materials.emplace_back(name, &material);
    }
    std::sort(sorted_materials.begin(), sorted_materials.end(), [](auto a, auto b) noexcept
    {
        return a.first < b.first;
    });
    for (auto&& [name, material] : sorted_materials)
    {
        if (material->reflectance_texture)
        {
            LUISA_VERBOSE(
                "PBRT material '{}' type=diffuse, reflectance=texture '{}', source={}.",
                name,
                *material->reflectance_texture,
                format_source_location(material->source));
        }
        else
        {
            LUISA_VERBOSE(
                "PBRT material '{}' type=diffuse, reflectance=({}, {}, {}), source={}.",
                name,
                material->reflectance.x,
                material->reflectance.y,
                material->reflectance.z,
                format_source_location(material->source));
        }
    }
    for (auto i = 0u; i < scene.materials.size(); i++)
    {
        auto&& material = scene.materials[i];
        if (material.reflectance_texture)
        {
            LUISA_VERBOSE(
                "PBRT inline material #{} type=diffuse, reflectance=texture '{}', source={}.",
                i,
                *material.reflectance_texture,
                format_source_location(material.source));
        }
        else
        {
            LUISA_VERBOSE(
                "PBRT inline material #{} type=diffuse, reflectance=({}, {}, {}), source={}.",
                i,
                material.reflectance.x,
                material.reflectance.y,
                material.reflectance.z,
                format_source_location(material.source));
        }
    }
    for (auto i = 0u; i < scene.shapes.size(); i++)
    {
        auto&& shape           = scene.shapes[i];
        auto material          = !shape.material.named.empty()
                                     ? luisa::format("named '{}'", shape.material.named)
                                 : shape.material.inline_index
                                     ? luisa::format("inline #{}", *shape.material.inline_index)
                                     : luisa::string{"default diffuse"};
        auto shape_detail      = shape.filename
                                     ? luisa::string{shape.filename->string().c_str()}
                                     : luisa::string{"inline"};
        auto area_light_detail = shape.area_light
                                     ? luisa::format(
                                           "diffuse L=({}, {}, {})",
                                           shape.area_light->emission.x,
                                           shape.area_light->emission.y,
                                           shape.area_light->emission.z)
                                     : luisa::string{"none"};
        LUISA_VERBOSE(
            "PBRT instance #{} shape={} ({}), material={}, area_light={}, transform={}, source={}.",
            i,
            shape_type_name(shape.type),
            shape_detail,
            material,
            area_light_detail,
            format_matrix(shape.pbrt_transform),
            format_source_location(shape.source));
    }
    LUISA_INFO("Imported PBRT scene '{}' in {} ms.", scene.source_path.string(), import_clock.toc());
    return result;
}

} // namespace Yutrel
