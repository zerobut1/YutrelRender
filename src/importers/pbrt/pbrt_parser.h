#pragma once

#include <filesystem>

#include "pbrt/pbrt_scene.h"

namespace Yutrel
{

class PbrtParser
{
public:
    [[nodiscard]] static PbrtScene parse(const std::filesystem::path& path);
};

} // namespace Yutrel
