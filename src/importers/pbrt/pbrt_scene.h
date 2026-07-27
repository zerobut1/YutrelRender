#pragma once

#include <array>
#include <filesystem>
#include <limits>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl.h>

#include "scene/source_location.h"

namespace Yutrel
{
using namespace luisa;

using Matrix4 = std::array<float, 16u>;

inline constexpr Matrix4 identity_matrix4{
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

struct RawValue
{
    SourceLocation source;
    luisa::string text;
    bool quoted{};
};

struct RawParameter
{
    SourceLocation source;
    luisa::string type;
    luisa::string name;
    luisa::vector<RawValue> values;
    bool bracketed{};
};

struct CameraDesc
{
    SourceLocation source;
    enum class Type
    {
        Perspective,
    };
    Type type{Type::Perspective};
    float fov{90.0f};
    float shutter_open{0.0f};
    float shutter_close{1.0f};
    Matrix4 pbrt_transform{identity_matrix4};
    luisa::optional<float3> world_up;
    luisa::vector<RawParameter> parameters;
};

struct FilmDesc
{
    SourceLocation source;
    enum class Type
    {
        RGB,
    };
    Type type{Type::RGB};
    uint2 resolution{1280u, 720u};
    float iso{100.0f};
    float max_component_value{std::numeric_limits<float>::infinity()};
    std::filesystem::path filename{"pbrt.exr"};
    luisa::vector<RawParameter> parameters;
};

struct IntegratorDesc
{
    SourceLocation source;
    enum class Type
    {
        Path,
        VolPath,
        SPPM,
    };
    Type type{Type::VolPath};
    uint max_depth{5u};
    uint photons_per_iteration{0u}; // 0 means use pixel_count
    float radius{1.0f};
    luisa::vector<RawParameter> parameters;
};

struct SamplerDesc
{
    SourceLocation source;
    enum class Type
    {
        Independent,
        Halton,
        Sobol,
        ZSobol,
    };
    Type type{Type::ZSobol};
    uint pixel_samples{16u};
    uint seed{20120712u};
    luisa::vector<RawParameter> parameters;
};

struct FilterDesc
{
    SourceLocation source;
    enum class Type
    {
        Triangle,
        Gaussian,
    };
    Type type{Type::Gaussian};
    float2 radius{1.5f, 1.5f};
    float sigma{0.5f};
    luisa::vector<RawParameter> parameters;
};

struct TextureInputDesc
{
    luisa::optional<luisa::string> texture;
    float4 constant{0.0f};
};

struct TextureDesc
{
    SourceLocation source;
    enum class ValueType
    {
        Float,
        Spectrum,
    };
    enum class Type
    {
        ImageMap,
        Constant,
        Scale,
        Checkerboard,
    };
    enum class Filter
    {
        Point,
        Bilinear,
    };
    enum class Encoding
    {
        Automatic,
        Linear,
        SRGB,
    };
    luisa::string name;
    ValueType value_type{ValueType::Float};
    Type type{Type::Constant};
    Filter filter{Filter::Bilinear};
    Encoding encoding{Encoding::Automatic};
    std::filesystem::path filename;
    float2 uv_scale{1.0f, 1.0f};
    float image_scale{1.0f};
    float constant_value{0.0f};
    luisa::string tex;
    luisa::string scale;
    TextureInputDesc tex1;
    TextureInputDesc tex2;
    luisa::vector<RawParameter> parameters;
};

struct MaterialDesc
{
    SourceLocation source;
    enum class Type
    {
        Diffuse,
        CoatedDiffuse,
        Dielectric,
        Interface,
    };
    Type type{Type::Diffuse};
    float3 reflectance{0.5f, 0.5f, 0.5f};
    luisa::optional<luisa::string> reflectance_texture;
    float roughness{0.0f};
    luisa::optional<luisa::string> roughness_texture;
    float u_roughness{0.0f};
    luisa::optional<luisa::string> u_roughness_texture;
    float v_roughness{0.0f};
    luisa::optional<luisa::string> v_roughness_texture;
    float thickness{0.01f};
    luisa::optional<luisa::string> thickness_texture;
    float3 albedo{0.0f, 0.0f, 0.0f};
    luisa::optional<luisa::string> albedo_texture;
    float g{0.0f};
    luisa::optional<luisa::string> g_texture;
    float eta{1.5f};
    luisa::optional<luisa::string> eta_texture;
    luisa::optional<luisa::string> eta_spectrum;
    bool remap_roughness{true};
    uint max_depth{10u};
    uint samples{1u};
    luisa::vector<RawParameter> parameters;
};

struct MediumDesc
{
    SourceLocation source;
    enum class Type
    {
        Homogeneous,
    };
    Type type{Type::Homogeneous};
    luisa::float3 sigma_a{0.0f};
    luisa::float3 sigma_s{0.0f};
    float scale{1.0f};
    float g{0.0f};
    luisa::vector<RawParameter> parameters;
};

struct MediumInterfaceDesc
{
    luisa::string inside;
    luisa::string outside;
};

struct MaterialBinding
{
    luisa::string named;
    luisa::optional<uint> inline_index;
};

struct AreaLightDesc
{
    SourceLocation source;
    enum class Type
    {
        Diffuse,
    };
    Type type{Type::Diffuse};
    float3 emission{0.0f, 0.0f, 0.0f};
    luisa::vector<RawParameter> parameters;
};

struct InfiniteLightDesc
{
    SourceLocation source;
    luisa::optional<float3> L;
    std::filesystem::path filename;
    float scale{1.0f};
    luisa::optional<float> illuminance;
    Matrix4 pbrt_transform{identity_matrix4};
    luisa::vector<RawParameter> parameters;
};

struct DistantLightDesc
{
    SourceLocation source;
    float3 L{1.0f};
    float scale{1.0f};
    luisa::optional<float> illuminance;
    float3 from{0.0f};
    float3 to{0.0f, 0.0f, 1.0f};
    Matrix4 pbrt_transform{identity_matrix4};
    luisa::vector<RawParameter> parameters;
};

struct PointLightDesc
{
    SourceLocation source;
    float3 I{1.0f};
    float scale{1.0f};
    float3 from{0.0f};
    Matrix4 pbrt_transform{identity_matrix4};
    luisa::vector<RawParameter> parameters;
};

struct MeshDesc
{
    SourceLocation source;
    luisa::vector<float3> positions;
    luisa::vector<float3> normals;
    luisa::vector<float2> uvs;
    luisa::vector<uint3> indices;
};

struct ShapeDesc
{
    static constexpr uint sphere_default_subdivision = 4u;
    static constexpr uint sphere_max_subdivision     = 8u;

    SourceLocation source;
    enum class Type
    {
        TriangleMesh,
        PlyMesh,
        Sphere,
    };
    Type type{Type::TriangleMesh};
    luisa::optional<uint> mesh_index;
    luisa::optional<std::filesystem::path> filename;
    float radius{1.0f};
    uint sphere_subdivision{sphere_default_subdivision};
    float alpha{1.0f};
    luisa::optional<luisa::string> alpha_texture;
    luisa::vector<RawParameter> parameters;
    MaterialBinding material;
    luisa::optional<AreaLightDesc> area_light;
    MediumInterfaceDesc medium_interface;
    Matrix4 pbrt_transform{identity_matrix4};
};

struct PbrtScene
{
    std::filesystem::path source_path;
    CameraDesc camera;
    FilmDesc film;
    IntegratorDesc integrator;
    SamplerDesc sampler;
    FilterDesc filter;
    luisa::vector<TextureDesc> textures;
    luisa::vector<MaterialDesc> materials;
    luisa::unordered_map<luisa::string, MaterialDesc> named_materials;
    luisa::unordered_map<luisa::string, MediumDesc> named_media;
    luisa::vector<MeshDesc> meshes;
    luisa::vector<ShapeDesc> shapes;
    luisa::vector<PointLightDesc> point_lights;
    luisa::vector<InfiniteLightDesc> infinite_lights;
    luisa::vector<DistantLightDesc> distant_lights;
};

} // namespace Yutrel
