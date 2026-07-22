#include "mesh.h"

#include <luisa/core/clock.h>

#include <assimp/Importer.hpp>
#include <assimp/Subdivision.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "scene/scene_builder.h"

namespace Yutrel
{
Mesh::Mesh(std::filesystem::path path) noexcept
    : m_loader(MeshLoader::load(std::move(path))) {}

const Shape* MeshShapeSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Shape, Mesh>(_path);
}

luisa::shared_ptr<MeshLoader> MeshLoader::load(std::filesystem::path path,
                                               uint subdiv_level,
                                               bool flip_uv,
                                               bool drop_normal,
                                               bool drop_uv) noexcept
{
    static luisa::lru_cache<uint64_t, luisa::shared_ptr<MeshLoader>> loaded_meshes{256u};

    auto abs_path = std::filesystem::canonical(path).string();
    auto key      = luisa::hash_value(abs_path, luisa::hash_value(subdiv_level));

    if (auto m = loaded_meshes.at(key))
    {
        return *m;
    }

    Clock clock;
    auto path_string = path.string();

    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
    importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 45.f);
    auto import_flags = aiProcess_RemoveComponent | aiProcess_SortByPType |
                        aiProcess_ValidateDataStructure | aiProcess_ImproveCacheLocality |
                        aiProcess_PreTransformVertices | aiProcess_FindInvalidData |
                        aiProcess_JoinIdenticalVertices;
    auto remove_flags = aiComponent_ANIMATIONS | aiComponent_BONEWEIGHTS |
                        aiComponent_CAMERAS | aiComponent_LIGHTS |
                        aiComponent_MATERIALS | aiComponent_TEXTURES |
                        aiComponent_COLORS | aiComponent_TANGENTS_AND_BITANGENTS;
    if (drop_uv)
    {
        remove_flags |= aiComponent_TEXCOORDS;
    }
    else
    {
        if (!flip_uv)
        {
            import_flags |= aiProcess_FlipUVs;
        }
        import_flags |= aiProcess_GenUVCoords | aiProcess_TransformUVCoords;
    }
    if (drop_normal)
    {
        import_flags |= aiProcess_DropNormals;
        remove_flags |= aiComponent_NORMALS;
    }
    else
    {
        import_flags |= aiProcess_GenSmoothNormals;
    }
    if (subdiv_level == 0u)
    {
        import_flags |= aiProcess_Triangulate;
    }
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, static_cast<int>(remove_flags));

    auto model = importer.ReadFile(path_string.c_str(), import_flags);
    if (model == nullptr ||
        (model->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ||
        model->mRootNode == nullptr ||
        model->mRootNode->mNumMeshes == 0) [[unlikely]]
    {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load mesh '{}': {}.",
            path_string,
            importer.GetErrorString());
    }
    if (auto err = importer.GetErrorString();
        err != nullptr && err[0] != '\0') [[unlikely]]
    {
        LUISA_WARNING_WITH_LOCATION(
            "Mesh '{}' has warnings: {}.",
            path_string,
            err);
    }
    LUISA_ASSERT(model->mNumMeshes == 1u, "Only single mesh is supported.");

    auto mesh = model->mMeshes[0];
    if (subdiv_level > 0u)
    {
        auto subdiv         = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
        aiMesh* subdiv_mesh = nullptr;
        subdiv->Subdivide(mesh, subdiv_mesh, subdiv_level, true);
        model->mMeshes[0] = nullptr;
        mesh              = subdiv_mesh;
        delete subdiv;
    }

    if (mesh->mTextureCoords[0] == nullptr ||
        mesh->mNumUVComponents[0] != 2) [[unlikely]]
    {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid texture coordinates in mesh '{}': "
            "address = {}, components = {}.",
            path_string,
            static_cast<void*>(mesh->mTextureCoords[0]),
            mesh->mNumUVComponents[0]);
    }

    auto vertex_count = mesh->mNumVertices;
    auto ai_positions = mesh->mVertices;
    auto ai_normals   = mesh->mNormals;
    auto ai_uvs       = mesh->mTextureCoords[0];
    auto loader       = luisa::make_shared<MeshLoader>();
    loader->m_vertices.resize(vertex_count);
    if (ai_normals)
    {
        loader->m_properties |= Shape::property_flag_has_vertex_normal;
    }
    if (ai_uvs)
    {
        loader->m_properties |= Shape::property_flag_has_vertex_uv;
    }
    for (auto i = 0; i < vertex_count; i++)
    {
        auto p  = make_float3(ai_positions[i].x, ai_positions[i].y, ai_positions[i].z);
        auto n  = ai_normals
                      ? normalize(make_float3(ai_normals[i].x, ai_normals[i].y, ai_normals[i].z))
                      : make_float3(0.f, 0.f, 1.f);
        auto uv = ai_uvs
                      ? make_float2(ai_uvs[i].x, ai_uvs[i].y)
                      : make_float2(0.f, 0.f);

        loader->m_vertices[i] = Vertex::encode(p, n, uv);
    }
    if (subdiv_level == 0u)
    {
        auto ai_triangles = mesh->mFaces;
        loader->m_triangles.resize(mesh->mNumFaces);
        for (auto i = 0; i < mesh->mNumFaces; i++)
        {
            auto&& face = ai_triangles[i];
            LUISA_ASSERT(face.mNumIndices == 3u);
            loader->m_triangles[i] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
        }
    }
    else
    {
        auto ai_triangles = mesh->mFaces;
        loader->m_triangles.resize(mesh->mNumFaces * 2u);
        for (auto i = 0; i < mesh->mNumFaces; i++)
        {
            auto&& face = ai_triangles[i];
            LUISA_ASSERT(face.mNumIndices == 4u);
            loader->m_triangles[i * 2u]      = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
            loader->m_triangles[i * 2u + 1u] = {face.mIndices[2], face.mIndices[3], face.mIndices[0]};
        }
    }

    LUISA_INFO("Loaded triangle mesh '{}' in {} ms.", path_string, clock.toc());

    return loader;
}
} // namespace Yutrel
