#pragma once

#include <cstdint>
#include <filesystem>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>

namespace Yutrel
{

struct SourceLocation
{
    std::filesystem::path file;
    uint32_t line{1u};
    uint32_t column{1u};
};

struct SpecMeta
{
    luisa::string name;
    SourceLocation source;
};

[[nodiscard]] inline luisa::string format_source_location(const SourceLocation& location)
{
    auto file = location.file.empty() ? luisa::string{"<unknown>"} : luisa::string{location.file.generic_string()};
    return luisa::format("{}:{}:{}", file, location.line, location.column);
}

} // namespace Yutrel
