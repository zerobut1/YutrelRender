#pragma once

#include <filesystem>

#include "cli_options.h"
#include "scene/scene_spec.h"

namespace Yutrel
{

[[nodiscard]] SceneSpec load_scene(
    const std::filesystem::path& path,
    const CliOverrides& overrides);

} // namespace Yutrel
