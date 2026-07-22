#pragma once

#include <filesystem>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/optional.h>

#include "scene/scene_spec.h"

namespace Yutrel
{

struct PbrtScene;

struct PbrtImportOptions
{
    luisa::optional<luisa::uint> spp;
    luisa::optional<luisa::uint> seed;
    luisa::optional<luisa::uint2> resolution;
    luisa::optional<std::filesystem::path> output;
    luisa::optional<luisa::float3> world_up;
};

class PbrtImporter
{
public:
    [[nodiscard]] static SceneSpec import(
        const std::filesystem::path& path,
        PbrtImportOptions options = {});
    [[nodiscard]] static SceneSpec import(PbrtScene scene);
};

} // namespace Yutrel
