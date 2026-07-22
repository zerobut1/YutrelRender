#pragma once

#include <filesystem>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl.h>

namespace Yutrel
{
using namespace luisa;

struct CliOverrides
{
    luisa::optional<uint> spp;
    luisa::optional<uint> seed;
    luisa::optional<uint2> resolution;
    luisa::optional<std::filesystem::path> output;
    luisa::optional<float3> world_up;
};

struct CommandLineOptions
{
    luisa::string backend;
    std::filesystem::path scene_path;
    bool interactive{false};
    bool headless{false};
    CliOverrides overrides;
};

[[nodiscard]] luisa::string command_line_usage(luisa::string_view bin);
[[nodiscard]] CommandLineOptions parse_command_line(int argc, char* argv[]);

} // namespace Yutrel
