#include "usd_importer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <luisa/core/logging.h>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdLux/boundableLightBase.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/nonboundableLightBase.h>
#include <pxr/usd/usdLux/shapingAPI.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/tokens.h>

#include "base/film.h"
#include "cameras/pinhole.h"
#include "environments/null.h"
#include "filters/gaussian.h"
#include "integrators/path.h"
#include "lights/diffuse.h"
#include "samplers/zsobol.h"
#include "scene/scene_spec_builder.h"
#include "shapes/inline_mesh.h"
#include "shapes/sphere.h"
#include "spectrum/hero.h"
#include "surfaces/coated_diffuse.h"
#include "surfaces/diffuse.h"
#include "surfaces/opacity.h"
#include "textures/constant.h"
#include "textures/image.h"
#include "textures/scale.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace Yutrel
{
namespace
{

constexpr auto default_resolution = luisa::uint2{1280u, 720u};
constexpr auto default_spp        = 16u;
constexpr auto default_seed       = 20120712u;
constexpr auto default_max_depth  = 5u;
constexpr auto float_epsilon      = 1e-6f;

[[noreturn]] void fail(
    const std::filesystem::path& scene_path,
    const UsdPrim& prim,
    const std::string& message)
{
    auto prim_path = prim ? prim.GetPath().GetString() : std::string{"/"};
    auto formatted = luisa::format(
        "{} [{}]: {}",
        scene_path.generic_string(),
        prim_path,
        message);
    throw std::runtime_error{formatted.c_str()};
}

[[nodiscard]] SourceLocation source_location(const std::filesystem::path& scene_path)
{
    return SourceLocation{scene_path};
}

[[nodiscard]] SpecMeta spec_meta(
    const std::filesystem::path& scene_path,
    std::string_view name)
{
    return SpecMeta{
        .name   = luisa::string{name.data(), name.size()},
        .source = source_location(scene_path),
    };
}

[[nodiscard]] luisa::float4x4 to_luisa_matrix(const GfMatrix4d& matrix) noexcept
{
    return luisa::make_float4x4(
        luisa::make_float4(
            static_cast<float>(matrix[0][0]),
            static_cast<float>(matrix[0][1]),
            static_cast<float>(matrix[0][2]),
            static_cast<float>(matrix[0][3])),
        luisa::make_float4(
            static_cast<float>(matrix[1][0]),
            static_cast<float>(matrix[1][1]),
            static_cast<float>(matrix[1][2]),
            static_cast<float>(matrix[1][3])),
        luisa::make_float4(
            static_cast<float>(matrix[2][0]),
            static_cast<float>(matrix[2][1]),
            static_cast<float>(matrix[2][2]),
            static_cast<float>(matrix[2][3])),
        luisa::make_float4(
            static_cast<float>(matrix[3][0]),
            static_cast<float>(matrix[3][1]),
            static_cast<float>(matrix[3][2]),
            static_cast<float>(matrix[3][3])));
}

[[nodiscard]] bool is_near(float a, float b, float epsilon = float_epsilon) noexcept
{
    return std::abs(a - b) <= epsilon;
}

[[nodiscard]] luisa::float3 to_float3(const GfVec3f& value) noexcept
{
    return luisa::make_float3(value[0], value[1], value[2]);
}

[[nodiscard]] luisa::float2 to_float2(const GfVec2f& value) noexcept
{
    return luisa::make_float2(value[0], value[1]);
}

[[nodiscard]] bool is_renderable(const UsdPrim& prim)
{
    if (!prim || !prim.IsActive() || !prim.IsDefined() || prim.IsAbstract())
    {
        return false;
    }
    auto imageable = UsdGeomImageable{prim};
    if (!imageable)
    {
        return true;
    }
    if (imageable.ComputeVisibility() == UsdGeomTokens->invisible)
    {
        return false;
    }
    auto purpose = imageable.ComputePurpose();
    return purpose == UsdGeomTokens->default_ || purpose == UsdGeomTokens->render;
}

template <typename T>
[[nodiscard]] T attribute_or(
    const UsdAttribute& attribute,
    T fallback,
    const std::filesystem::path& scene_path,
    const UsdPrim& prim,
    const char* name)
{
    if (!attribute)
    {
        return fallback;
    }
    T value{};
    if (!attribute.Get(&value, UsdTimeCode::Default()))
    {
        fail(scene_path, prim, luisa::format("Failed to read USD attribute '{}'.", name).c_str());
    }
    return value;
}

[[nodiscard]] std::filesystem::path resolve_asset_path(
    const SdfAssetPath& asset,
    const std::filesystem::path& scene_path,
    const UsdPrim& prim,
    const char* owner)
{
    std::filesystem::path result;
    if (!asset.GetResolvedPath().empty())
    {
        result = asset.GetResolvedPath();
    }
    else if (!asset.GetAssetPath().empty())
    {
        auto authored = std::filesystem::path{asset.GetAssetPath()};
        result        = authored.is_absolute() ? authored : scene_path.parent_path() / authored;
    }
    if (result.empty())
    {
        fail(scene_path, prim, luisa::format("{} has an empty asset path.", owner).c_str());
    }
    result = std::filesystem::absolute(result).lexically_normal();
    std::error_code error;
    if (!std::filesystem::is_regular_file(result, error))
    {
        fail(scene_path, prim, luisa::format("{} asset '{}' is not a regular file.", owner, result.generic_string()).c_str());
    }
    return result;
}

[[nodiscard]] Texture::Encoding texture_encoding(
    const TfToken& color_space,
    const std::filesystem::path& path,
    const std::filesystem::path& scene_path,
    const UsdPrim& prim)
{
    auto name = color_space.GetString();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) noexcept
    {
        return static_cast<char>(std::tolower(c));
    });
    if (name.empty() || name == "auto")
    {
        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) noexcept
        {
            return static_cast<char>(std::tolower(c));
        });
        return extension == ".png" ? Texture::Encoding::SRGB : Texture::Encoding::LINEAR;
    }
    if (name == "raw" || name == "linear" || name == "lin_rec709_scene")
    {
        return Texture::Encoding::LINEAR;
    }
    if (name == "srgb")
    {
        return Texture::Encoding::SRGB;
    }
    fail(scene_path, prim, luisa::format("Unsupported USD texture color space '{}'.", name).c_str());
}

[[nodiscard]] size_t interpolation_index(
    const TfToken& interpolation,
    size_t face_index,
    size_t face_vertex_index,
    size_t point_index,
    const std::filesystem::path& scene_path,
    const UsdPrim& prim,
    const char* attribute)
{
    if (interpolation == UsdGeomTokens->constant)
    {
        return 0u;
    }
    if (interpolation == UsdGeomTokens->uniform)
    {
        return face_index;
    }
    if (interpolation == UsdGeomTokens->vertex || interpolation == UsdGeomTokens->varying)
    {
        return point_index;
    }
    if (interpolation == UsdGeomTokens->faceVarying)
    {
        return face_vertex_index;
    }
    fail(scene_path, prim, luisa::format("Unsupported interpolation '{}' for {}.", interpolation.GetString(), attribute).c_str());
}

template <typename T>
[[nodiscard]] const T& checked_value(
    const VtArray<T>& values,
    size_t index,
    const std::filesystem::path& scene_path,
    const UsdPrim& prim,
    const char* attribute)
{
    if (index >= values.size())
    {
        fail(scene_path, prim, luisa::format("{} interpolation index {} exceeds value count {}.", attribute, index, values.size()).c_str());
    }
    return values[index];
}

struct MeshData
{
    luisa::vector<luisa::float3> positions;
    luisa::vector<luisa::float3> normals;
    luisa::vector<luisa::float2> uvs;
    luisa::vector<luisa::uint3> indices;
};

[[nodiscard]] MeshData import_mesh_data(
    const UsdGeomMesh& mesh,
    const std::filesystem::path& scene_path)
{
    auto prim        = mesh.GetPrim();
    auto subdivision = attribute_or(
        mesh.GetSubdivisionSchemeAttr(),
        UsdGeomTokens->catmullClark,
        scene_path,
        prim,
        "subdivisionScheme");
    if (subdivision != UsdGeomTokens->none)
    {
        fail(scene_path, prim, luisa::format("Unsupported subdivision scheme '{}'.", subdivision.GetString()).c_str());
    }

    VtVec3fArray points;
    VtIntArray face_counts;
    VtIntArray face_indices;
    if (!mesh.GetPointsAttr().Get(&points) ||
        !mesh.GetFaceVertexCountsAttr().Get(&face_counts) ||
        !mesh.GetFaceVertexIndicesAttr().Get(&face_indices))
    {
        fail(scene_path, prim, "USD mesh is missing points or topology.");
    }
    if (points.empty() || face_counts.empty() || face_indices.empty())
    {
        fail(scene_path, prim, "USD mesh has empty points or topology.");
    }

    VtVec3fArray normals;
    auto normals_attribute     = mesh.GetNormalsAttr();
    auto normals_interpolation = mesh.GetNormalsInterpolation();
    auto has_normals           = normals_attribute && normals_attribute.HasAuthoredValueOpinion();
    if (has_normals && !UsdGeomPrimvar{normals_attribute}.ComputeFlattened(&normals))
    {
        fail(scene_path, prim, "Failed to flatten indexed USD mesh normals.");
    }
    has_normals &= !normals.empty();

    auto st = UsdGeomPrimvarsAPI{prim}.GetPrimvar(TfToken{"st"});
    VtVec2fArray uvs;
    auto has_uvs = st && st.HasAuthoredValue();
    if (has_uvs && !st.ComputeFlattened(&uvs))
    {
        fail(scene_path, prim, "Failed to flatten indexed primvars:st.");
    }
    has_uvs &= !uvs.empty();
    auto uv_interpolation = st.GetInterpolation();

    VtIntArray holes;
    if (auto holes_attribute = mesh.GetHoleIndicesAttr(); holes_attribute)
    {
        (void)holes_attribute.Get(&holes);
    }

    auto orientation = attribute_or(
        mesh.GetOrientationAttr(),
        UsdGeomTokens->rightHanded,
        scene_path,
        prim,
        "orientation");
    if (orientation != UsdGeomTokens->rightHanded && orientation != UsdGeomTokens->leftHanded)
    {
        fail(scene_path, prim, luisa::format("Unsupported mesh orientation '{}'.", orientation.GetString()).c_str());
    }

    MeshData result;
    result.positions.reserve(face_indices.size());
    if (has_normals)
    {
        result.normals.reserve(face_indices.size());
    }
    if (has_uvs)
    {
        result.uvs.reserve(face_indices.size());
    }

    size_t face_vertex_offset = 0u;
    for (size_t face_index = 0u; face_index < face_counts.size(); face_index++)
    {
        auto count = face_counts[face_index];
        if (count < 3)
        {
            fail(scene_path, prim, luisa::format("Face {} has fewer than three vertices.", face_index).c_str());
        }
        auto face_vertex_count = static_cast<size_t>(count);
        if (face_vertex_offset + face_vertex_count > face_indices.size())
        {
            fail(scene_path, prim, "faceVertexCounts exceeds faceVertexIndices length.");
        }
        auto is_hole = std::find(holes.begin(), holes.end(), static_cast<int>(face_index)) != holes.end();
        if (is_hole)
        {
            face_vertex_offset += face_vertex_count;
            continue;
        }

        auto base_vertex = static_cast<luisa::uint>(result.positions.size());
        for (size_t corner = 0u; corner < face_vertex_count; corner++)
        {
            auto face_vertex_index = face_vertex_offset + corner;
            auto authored_index    = face_indices[face_vertex_index];
            if (authored_index < 0 || static_cast<size_t>(authored_index) >= points.size())
            {
                fail(scene_path, prim, luisa::format("faceVertexIndices[{}] is out of bounds.", face_vertex_index).c_str());
            }
            auto point_index = static_cast<size_t>(authored_index);
            result.positions.emplace_back(to_float3(points[point_index]));
            if (has_normals)
            {
                auto index = interpolation_index(normals_interpolation, face_index, face_vertex_index, point_index, scene_path, prim, "normals");
                result.normals.emplace_back(to_float3(checked_value(normals, index, scene_path, prim, "normals")));
            }
            if (has_uvs)
            {
                auto index = interpolation_index(uv_interpolation, face_index, face_vertex_index, point_index, scene_path, prim, "primvars:st");
                result.uvs.emplace_back(to_float2(checked_value(uvs, index, scene_path, prim, "primvars:st")));
            }
        }
        for (luisa::uint corner = 1u; corner + 1u < face_vertex_count; corner++)
        {
            if (orientation == UsdGeomTokens->rightHanded)
            {
                result.indices.emplace_back(luisa::make_uint3(base_vertex, base_vertex + corner, base_vertex + corner + 1u));
            }
            else
            {
                result.indices.emplace_back(luisa::make_uint3(base_vertex, base_vertex + corner + 1u, base_vertex + corner));
            }
        }
        face_vertex_offset += face_vertex_count;
    }
    if (face_vertex_offset != face_indices.size())
    {
        fail(scene_path, prim, "faceVertexCounts does not consume all faceVertexIndices.");
    }
    if (result.indices.empty())
    {
        fail(scene_path, prim, "USD mesh contains no renderable triangles.");
    }
    return result;
}

[[nodiscard]] TextureSampler texture_sampler(
    const UsdShadeShader& shader,
    const std::filesystem::path& scene_path)
{
    auto prim   = shader.GetPrim();
    auto wrap_s = attribute_or(shader.GetInput(TfToken{"wrapS"}).GetAttr(), TfToken{"repeat"}, scene_path, prim, "wrapS");
    auto wrap_t = attribute_or(shader.GetInput(TfToken{"wrapT"}).GetAttr(), TfToken{"repeat"}, scene_path, prim, "wrapT");
    if (wrap_s != wrap_t)
    {
        fail(scene_path, prim, "Yutrel requires identical USD wrapS and wrapT modes.");
    }
    auto wrap = wrap_s.GetString();
    if (wrap == "repeat" || wrap == "useMetadata")
    {
        return TextureSampler::linear_point_repeat();
    }
    if (wrap == "clamp" || wrap == "black")
    {
        return TextureSampler::linear_point_edge();
    }
    if (wrap == "mirror")
    {
        return TextureSampler::linear_point_mirror();
    }
    fail(scene_path, prim, luisa::format("Unsupported USD texture wrap mode '{}'.", wrap).c_str());
}

class ImportContext
{
private:
    std::filesystem::path _scene_path;
    UsdStageRefPtr _stage;
    SceneSpecBuilder _builder;
    UsdGeomXformCache _xform_cache{UsdTimeCode::Default()};
    std::unordered_map<std::string, SurfaceRef> _material_cache;
    std::unordered_map<std::string, TextureRef> _texture_cache;
    luisa::optional<SurfaceRef> _default_surface;
    luisa::optional<SurfaceRef> _light_surface;
    luisa::optional<CameraRef> _camera;

public:
    ImportContext(std::filesystem::path scene_path, UsdStageRefPtr stage) noexcept
        : _scene_path{std::move(scene_path)}, _stage{std::move(stage)} {}

    [[nodiscard]] SceneSpec import(UsdImportOptions options)
    {
        for (auto&& prim : _stage->Traverse())
        {
            if (!is_renderable(prim))
            {
                continue;
            }
            if (prim.IsInstanceProxy() || prim.IsInstance())
            {
                fail(_scene_path, prim, "USD instancing is not supported in the first importer version.");
            }
            if (prim.IsA<UsdGeomMesh>())
            {
                _import_mesh(UsdGeomMesh{prim});
                continue;
            }
            if (prim.IsA<UsdGeomSubset>())
            {
                fail(_scene_path, prim, "UsdGeomSubset material assignment is not supported.");
            }
            if (prim.IsA<UsdGeomPointInstancer>())
            {
                fail(_scene_path, prim, "UsdGeomPointInstancer is not supported.");
            }
            if (prim.IsA<UsdLuxSphereLight>())
            {
                _import_sphere_light(UsdLuxSphereLight{prim});
                continue;
            }
            if (prim.IsA<UsdLuxDomeLight>())
            {
                // The first version intentionally ignores dome lights and uses a null environment.
                continue;
            }
            if (prim.IsA<UsdGeomCamera>())
            {
                _import_camera(UsdGeomCamera{prim});
                continue;
            }
            if (prim.IsA<UsdGeomGprim>())
            {
                fail(_scene_path, prim, luisa::format("Unsupported renderable USD prim type '{}'.", prim.GetTypeName().GetString()).c_str());
            }
            if (prim.IsA<UsdLuxBoundableLightBase>() || prim.IsA<UsdLuxNonboundableLightBase>())
            {
                fail(_scene_path, prim, luisa::format("Unsupported USD light type '{}'.", prim.GetTypeName().GetString()).c_str());
            }
        }

        if (!_camera)
        {
            fail(_scene_path, _stage->GetPseudoRoot(), "USD scene must contain exactly one visible perspective camera.");
        }

        auto environment = _builder.add_environment<NullEnvironmentSpec>(
            spec_meta(_scene_path, "usd_null_environment"));

        auto resolution = options.resolution.value_or(default_resolution);
        auto output     = options.output.value_or(
            std::filesystem::absolute(_scene_path.parent_path() / (_scene_path.stem().string() + ".exr")));
        auto film = _builder.add_film<RGBFilmSpec>(
            spec_meta(_scene_path, "usd_film"),
            resolution,
            false,
            std::move(output));
        auto filter = _builder.add_filter<GaussianFilterSpec>(
            spec_meta(_scene_path, "usd_filter"),
            1.5f,
            0.5f);
        auto spectrum = _builder.add_spectrum<HeroWavelengthSpectrumSpec>(
            spec_meta(_scene_path, "usd_spectrum"));
        auto sampler = _builder.add_sampler<ZSobolSamplerSpec>(
            spec_meta(_scene_path, "usd_sampler"),
            options.spp.value_or(default_spp),
            options.seed.value_or(default_seed));
        auto integrator = _builder.add_integrator<PathIntegratorSpec>(
            spec_meta(_scene_path, "usd_integrator"),
            default_max_depth);
        _builder.set_render(RenderSpec{
            .spectrum    = spectrum,
            .environment = environment,
            .camera      = *_camera,
            .film        = film,
            .filter      = filter,
            .sampler     = sampler,
            .integrator  = integrator,
        });
        return _builder.finish();
    }

private:
    [[nodiscard]] TextureRef _constant_texture(
        const UsdPrim&,
        std::string name,
        luisa::float4 value)
    {
        return _builder.add_texture<ConstantTextureSpec>(
            spec_meta(_scene_path, name),
            value);
    }

    [[nodiscard]] SurfaceRef _get_default_surface(const UsdPrim& prim)
    {
        if (!_default_surface)
        {
            auto reflectance = _constant_texture(prim, "usd_default_reflectance", luisa::make_float4(0.5f, 0.5f, 0.5f, 1.0f));
            _default_surface = _builder.add_surface<DiffuseSurfaceSpec>(
                spec_meta(_scene_path, "usd_default_surface"),
                reflectance,
                true);
        }
        return *_default_surface;
    }

    [[nodiscard]] SurfaceRef _get_light_surface(const UsdPrim& prim)
    {
        if (!_light_surface)
        {
            auto reflectance = _constant_texture(prim, "usd_light_black", luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f));
            _light_surface   = _builder.add_surface<DiffuseSurfaceSpec>(
                spec_meta(_scene_path, "usd_light_surface"),
                reflectance,
                false);
        }
        return *_light_surface;
    }

    [[nodiscard]] TextureRef _import_uv_texture(
        const UsdShadeConnectionSourceInfo& source,
        const UsdPrim& owner)
    {
        if (source.sourceType != UsdShadeAttributeType::Output || source.sourceName != TfToken{"rgb"})
        {
            fail(_scene_path, owner, "diffuseColor must connect to an UsdUVTexture rgb output.");
        }
        auto shader = UsdShadeShader{source.source.GetPrim()};
        TfToken id;
        if (!shader || !shader.GetIdAttr().Get(&id) || id != TfToken{"UsdUVTexture"})
        {
            fail(_scene_path, owner, "diffuseColor source is not an UsdUVTexture shader.");
        }
        auto shader_path = shader.GetPath().GetString();
        if (auto iter = _texture_cache.find(shader_path); iter != _texture_cache.end())
        {
            return iter->second;
        }

        auto st_sources = shader.GetInput(TfToken{"st"}).GetConnectedSources();
        if (st_sources.size() != 1u)
        {
            fail(_scene_path, shader.GetPrim(), "UsdUVTexture st must have exactly one source.");
        }
        auto st_shader = UsdShadeShader{st_sources.front().source.GetPrim()};
        TfToken st_id;
        if (!st_shader || !st_shader.GetIdAttr().Get(&st_id) || st_id != TfToken{"UsdPrimvarReader_float2"})
        {
            fail(_scene_path, shader.GetPrim(), "UsdUVTexture st source must be UsdPrimvarReader_float2.");
        }
        TfToken varname;
        if (!st_shader.GetInput(TfToken{"varname"}).Get(&varname) || varname != TfToken{"st"})
        {
            fail(_scene_path, st_shader.GetPrim(), "UsdPrimvarReader_float2 must read primvars:st.");
        }

        SdfAssetPath file;
        if (!shader.GetInput(TfToken{"file"}).Get(&file))
        {
            fail(_scene_path, shader.GetPrim(), "UsdUVTexture is missing inputs:file.");
        }
        auto path        = resolve_asset_path(file, _scene_path, shader.GetPrim(), "UsdUVTexture");
        auto color_space = attribute_or(
            shader.GetInput(TfToken{"sourceColorSpace"}).GetAttr(),
            TfToken{"auto"},
            _scene_path,
            shader.GetPrim(),
            "sourceColorSpace");
        auto encoding = texture_encoding(color_space, path, _scene_path, shader.GetPrim());
        auto name     = shader_path;
        auto texture  = _builder.add_texture<ImageTextureSpec>(
            spec_meta(_scene_path, name),
            std::move(path),
            texture_sampler(shader, _scene_path),
            encoding);

        auto scale = attribute_or(
            shader.GetInput(TfToken{"scale"}).GetAttr(),
            GfVec4f{1.0f},
            _scene_path,
            shader.GetPrim(),
            "scale");
        auto bias = attribute_or(
            shader.GetInput(TfToken{"bias"}).GetAttr(),
            GfVec4f{0.0f},
            _scene_path,
            shader.GetPrim(),
            "bias");
        auto has_scale = !is_near(scale[0], 1.0f) || !is_near(scale[1], 1.0f) || !is_near(scale[2], 1.0f) || !is_near(scale[3], 1.0f);
        auto has_bias  = !is_near(bias[0], 0.0f) || !is_near(bias[1], 0.0f) || !is_near(bias[2], 0.0f) || !is_near(bias[3], 0.0f);
        if (has_scale || has_bias)
        {
            texture = _builder.add_texture<ScaleTextureSpec>(
                spec_meta(_scene_path, name + "::scale_bias"),
                texture,
                luisa::make_float4(scale[0], scale[1], scale[2], scale[3]),
                luisa::make_float4(bias[0], bias[1], bias[2], bias[3]));
        }
        _texture_cache.emplace(std::move(shader_path), texture);
        return texture;
    }

    [[nodiscard]] TextureRef _import_diffuse_color(
        const UsdShadeShader& shader,
        const UsdPrim& owner)
    {
        auto input   = shader.GetInput(TfToken{"diffuseColor"});
        auto sources = input.GetConnectedSources();
        if (sources.size() > 1u)
        {
            fail(_scene_path, owner, "diffuseColor has multiple connected sources.");
        }
        if (sources.size() == 1u)
        {
            return _import_uv_texture(sources.front(), owner);
        }
        GfVec3f value{0.18f};
        if (input && !input.Get(&value))
        {
            fail(_scene_path, owner, "Failed to read UsdPreviewSurface diffuseColor.");
        }
        auto name = owner.GetPath().GetString() + "::diffuseColor";
        return _constant_texture(owner, name, luisa::make_float4(value[0], value[1], value[2], 1.0f));
    }

    template <typename T>
    [[nodiscard]] T _constant_input(
        const UsdShadeShader& shader,
        const char* name,
        T fallback)
    {
        auto input = shader.GetInput(TfToken{name});
        if (input && input.HasConnectedSource())
        {
            fail(_scene_path, shader.GetPrim(), luisa::format("UsdPreviewSurface {} must be constant.", name).c_str());
        }
        if (input && !input.Get(&fallback))
        {
            fail(_scene_path, shader.GetPrim(), luisa::format("Failed to read UsdPreviewSurface {}.", name).c_str());
        }
        return fallback;
    }

    [[nodiscard]] SurfaceRef _import_material(const UsdPrim& geometry)
    {
        auto material = UsdShadeMaterialBindingAPI{geometry}.ComputeBoundMaterial();
        if (!material)
        {
            return _get_default_surface(geometry);
        }
        auto material_path = material.GetPath().GetString();
        if (auto iter = _material_cache.find(material_path); iter != _material_cache.end())
        {
            return iter->second;
        }

        auto shader = material.ComputeSurfaceSource();
        TfToken id;
        if (!shader || !shader.GetIdAttr().Get(&id) || id != TfToken{"UsdPreviewSurface"})
        {
            fail(_scene_path, material.GetPrim(), "USD material surface must resolve to UsdPreviewSurface.");
        }
        if (material.ComputeDisplacementSource())
        {
            fail(_scene_path, material.GetPrim(), "USD material displacement output is not supported.");
        }
        if (material.ComputeVolumeSource())
        {
            fail(_scene_path, material.GetPrim(), "USD material volume output is not supported.");
        }
        auto metallic = _constant_input(shader, "metallic", 0.0f);
        if (!std::isfinite(metallic) || !is_near(metallic, 0.0f))
        {
            fail(_scene_path, shader.GetPrim(), "Metallic UsdPreviewSurface is not supported.");
        }
        auto use_specular_workflow = _constant_input(shader, "useSpecularWorkflow", 0);
        if (use_specular_workflow != 0)
        {
            fail(_scene_path, shader.GetPrim(), "UsdPreviewSurface specular workflow is not supported.");
        }
        auto clearcoat = _constant_input(shader, "clearcoat", 0.0f);
        if (!std::isfinite(clearcoat) || !is_near(clearcoat, 0.0f))
        {
            fail(_scene_path, shader.GetPrim(), "UsdPreviewSurface clearcoat is not supported.");
        }
        for (auto name : {"normal", "displacement", "occlusion"})
        {
            auto input = shader.GetInput(TfToken{name});
            if (input && input.HasConnectedSource())
            {
                fail(_scene_path, shader.GetPrim(), luisa::format("Connected UsdPreviewSurface {} is not supported.", name).c_str());
            }
        }
        auto emissive = _constant_input(shader, "emissiveColor", GfVec3f{0.0f});
        if (!is_near(emissive[0], 0.0f) || !is_near(emissive[1], 0.0f) || !is_near(emissive[2], 0.0f))
        {
            fail(_scene_path, shader.GetPrim(), "Emissive UsdPreviewSurface is not supported.");
        }

        auto roughness = _constant_input(shader, "roughness", 0.5f);
        auto ior       = _constant_input(shader, "ior", 1.5f);
        auto opacity   = _constant_input(shader, "opacity", 1.0f);
        auto threshold = _constant_input(shader, "opacityThreshold", 0.0f);
        if (!std::isfinite(roughness) || roughness < 0.0f || roughness > 1.0f ||
            !std::isfinite(ior) || ior <= 0.0f ||
            !std::isfinite(opacity) || opacity < 0.0f || opacity > 1.0f)
        {
            fail(_scene_path, shader.GetPrim(), "UsdPreviewSurface roughness, ior, or opacity is invalid.");
        }
        if (!is_near(threshold, 0.0f))
        {
            fail(_scene_path, shader.GetPrim(), "UsdPreviewSurface opacityThreshold is not supported.");
        }

        auto reflectance       = _import_diffuse_color(shader, material.GetPrim());
        auto roughness_texture = _constant_texture(
            material.GetPrim(),
            material_path + "::roughness",
            luisa::make_float4(roughness));
        auto ior_texture = _constant_texture(
            material.GetPrim(),
            material_path + "::ior",
            luisa::make_float4(ior));
        auto surface = _builder.add_surface<CoatedDiffuseSurfaceSpec>(
            spec_meta(_scene_path, material_path),
            CoatedDiffuseSurfaceParams{
                .reflectance     = reflectance,
                .roughness       = roughness_texture,
                .eta             = ior_texture,
                .remap_roughness = true,
            });
        if (!is_near(opacity, 1.0f))
        {
            auto alpha = _constant_texture(
                material.GetPrim(),
                material_path + "::opacity",
                luisa::make_float4(opacity));
            surface = _builder.add_surface<OpacitySurfaceSpec>(
                spec_meta(_scene_path, material_path + "::opacity_surface"),
                surface,
                alpha);
        }
        _material_cache.emplace(material_path, surface);
        return surface;
    }

    void _import_mesh(const UsdGeomMesh& mesh)
    {
        auto prim  = mesh.GetPrim();
        auto data  = import_mesh_data(mesh, _scene_path);
        auto name  = prim.GetPath().GetString();
        auto shape = _builder.add_shape<InlineMeshShapeSpec>(
            spec_meta(_scene_path, name),
            std::move(data.positions),
            std::move(data.normals),
            std::move(data.uvs),
            std::move(data.indices));
        _builder.add_instance(ShapeInstanceSpec{
            .source    = source_location(_scene_path),
            .shape     = shape,
            .surface   = _import_material(prim),
            .transform = to_luisa_matrix(_xform_cache.GetLocalToWorldTransform(prim)),
        });
    }

    void _validate_light_controls(const UsdLuxLightAPI& light, const UsdPrim& prim)
    {
        auto enable_temperature = attribute_or(
            light.GetEnableColorTemperatureAttr(),
            false,
            _scene_path,
            prim,
            "inputs:enableColorTemperature");
        if (enable_temperature)
        {
            fail(_scene_path, prim, "USD light color temperature is not supported.");
        }
    }

    void _import_sphere_light(const UsdLuxSphereLight& light)
    {
        auto prim = light.GetPrim();
        if (prim.HasAPI<UsdLuxShapingAPI>())
        {
            fail(_scene_path, prim, "UsdLuxShapingAPI is not supported on SphereLight.");
        }
        auto treat_as_point = attribute_or(light.GetTreatAsPointAttr(), false, _scene_path, prim, "treatAsPoint");
        if (treat_as_point)
        {
            fail(_scene_path, prim, "SphereLight treatAsPoint is not supported.");
        }
        _validate_light_controls(UsdLuxLightAPI{prim}, prim);

        auto radius    = attribute_or(light.GetRadiusAttr(), 0.5f, _scene_path, prim, "inputs:radius");
        auto intensity = attribute_or(light.GetIntensityAttr(), 1.0f, _scene_path, prim, "inputs:intensity");
        auto exposure  = attribute_or(light.GetExposureAttr(), 0.0f, _scene_path, prim, "inputs:exposure");
        auto normalize = attribute_or(light.GetNormalizeAttr(), false, _scene_path, prim, "inputs:normalize");
        auto color     = attribute_or(light.GetColorAttr(), GfVec3f{1.0f}, _scene_path, prim, "inputs:color");
        if (!std::isfinite(radius) || radius <= 0.0f || !std::isfinite(intensity) || intensity < 0.0f || !std::isfinite(exposure))
        {
            fail(_scene_path, prim, "SphereLight radius, intensity, or exposure is invalid.");
        }
        if (!std::isfinite(color[0]) || color[0] < 0.0f ||
            !std::isfinite(color[1]) || color[1] < 0.0f ||
            !std::isfinite(color[2]) || color[2] < 0.0f)
        {
            fail(_scene_path, prim, "SphereLight color must be finite and non-negative.");
        }

        auto transform = to_luisa_matrix(_xform_cache.GetLocalToWorldTransform(prim));
        auto sx        = luisa::length(luisa::make_float3(transform[0]));
        auto sy        = luisa::length(luisa::make_float3(transform[1]));
        auto sz        = luisa::length(luisa::make_float3(transform[2]));
        auto scale     = (sx + sy + sz) / 3.0f;
        auto tolerance = 1e-4f * std::max(1.0f, scale);
        if (!std::isfinite(scale) || scale <= float_epsilon ||
            std::abs(sx - scale) > tolerance || std::abs(sy - scale) > tolerance || std::abs(sz - scale) > tolerance)
        {
            fail(_scene_path, prim, "SphereLight requires a uniform world-space scale.");
        }

        auto emission_scale = intensity * std::exp2(exposure);
        if (normalize)
        {
            auto world_radius = radius * scale;
            emission_scale /= 4.0f * luisa::pi * world_radius * world_radius;
        }
        auto name     = prim.GetPath().GetString();
        auto emission = _constant_texture(
            prim,
            name + "::emission",
            luisa::make_float4(color[0] * emission_scale, color[1] * emission_scale, color[2] * emission_scale, 1.0f));
        auto light_ref = _builder.add_light<DiffuseLightSpec>(
            spec_meta(_scene_path, name + "::light"),
            emission,
            1.0f,
            false);
        auto shape = _builder.add_shape<SphereShapeSpec>(
            spec_meta(_scene_path, name),
            radius);
        _builder.add_instance(ShapeInstanceSpec{
            .source    = source_location(_scene_path),
            .shape     = shape,
            .surface   = _get_light_surface(prim),
            .light     = light_ref,
            .transform = transform,
        });
    }

    void _import_camera(const UsdGeomCamera& camera)
    {
        auto prim = camera.GetPrim();
        if (_camera)
        {
            fail(_scene_path, prim, "USD scene contains multiple visible cameras.");
        }
        auto projection = attribute_or(camera.GetProjectionAttr(), UsdGeomTokens->perspective, _scene_path, prim, "projection");
        if (projection != UsdGeomTokens->perspective)
        {
            fail(_scene_path, prim, "Only perspective USD cameras are supported.");
        }
        auto focal_length      = attribute_or(camera.GetFocalLengthAttr(), 50.0f, _scene_path, prim, "focalLength");
        auto vertical_aperture = attribute_or(camera.GetVerticalApertureAttr(), 15.2908f, _scene_path, prim, "verticalAperture");
        auto horizontal_offset = attribute_or(camera.GetHorizontalApertureOffsetAttr(), 0.0f, _scene_path, prim, "horizontalApertureOffset");
        auto vertical_offset   = attribute_or(camera.GetVerticalApertureOffsetAttr(), 0.0f, _scene_path, prim, "verticalApertureOffset");
        if (!std::isfinite(focal_length) || focal_length <= 0.0f ||
            !std::isfinite(vertical_aperture) || vertical_aperture <= 0.0f)
        {
            fail(_scene_path, prim, "USD camera focal length or aperture is invalid.");
        }
        if (!is_near(horizontal_offset, 0.0f) || !is_near(vertical_offset, 0.0f))
        {
            fail(_scene_path, prim, "USD camera aperture offsets are not supported.");
        }
        auto fov = 2.0f * std::atan(0.5f * vertical_aperture / focal_length) * 180.0f / luisa::pi;
        _camera  = _builder.add_camera<PinholeCameraSpec>(
            spec_meta(_scene_path, prim.GetPath().GetString()),
            to_luisa_matrix(_xform_cache.GetLocalToWorldTransform(prim)),
            luisa::make_float2(0.0f),
            0u,
            fov);

        auto f_stop = camera.GetFStopAttr();
        if (f_stop && f_stop.HasAuthoredValueOpinion())
        {
            LUISA_WARNING("USD camera '{}' depth of field is ignored by the first importer version.", prim.GetPath().GetString());
        }
    }
};

} // namespace

SceneSpec UsdImporter::import(
    const std::filesystem::path& path,
    UsdImportOptions options)
{
    if (path.empty())
    {
        throw std::runtime_error{"USD scene path cannot be empty."};
    }
    auto scene_path = std::filesystem::absolute(path).lexically_normal();
    std::error_code error;
    if (!std::filesystem::is_regular_file(scene_path, error))
    {
        auto message = luisa::format("USD scene '{}' is not a regular file.", scene_path.generic_string());
        throw std::runtime_error{message.c_str()};
    }
    auto stage = UsdStage::Open(scene_path.string(), UsdStage::LoadAll);
    if (!stage)
    {
        auto message = luisa::format("Failed to open USD scene '{}'.", scene_path.generic_string());
        throw std::runtime_error{message.c_str()};
    }
    return ImportContext{std::move(scene_path), std::move(stage)}.import(std::move(options));
}

} // namespace Yutrel
