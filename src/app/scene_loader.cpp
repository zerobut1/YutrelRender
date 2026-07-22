#include "scene_loader.h"

#include <cctype>
#include <stdexcept>
#include <string>

#include "pbrt/pbrt_importer.h"
#include "usd/usd_importer.h"

namespace Yutrel
{
namespace
{

[[nodiscard]] std::string lower_extension(const std::filesystem::path& path)
{
    auto extension = path.extension().string();
    for (auto& c : extension)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return extension;
}

} // namespace

SceneSpec load_scene(
    const std::filesystem::path& path,
    const CliOverrides& overrides)
{
    auto extension = lower_extension(path);
    if (extension == ".pbrt")
    {
        return PbrtImporter::import(
            path,
            PbrtImportOptions{
                .spp        = overrides.spp,
                .seed       = overrides.seed,
                .resolution = overrides.resolution,
                .output     = overrides.output,
            });
    }
    if (extension == ".usd" ||
        extension == ".usda" ||
        extension == ".usdc" ||
        extension == ".usdz")
    {
        return UsdImporter::import(
            path,
            UsdImportOptions{
                .spp        = overrides.spp,
                .seed       = overrides.seed,
                .resolution = overrides.resolution,
                .output     = overrides.output,
            });
    }
    throw std::runtime_error{
        "Unsupported scene format '" + extension + "' for '" + path.string() + "'."};
}

} // namespace Yutrel
