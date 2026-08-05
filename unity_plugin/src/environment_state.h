#pragma once

#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/optional.h>

namespace yutrel::unity {

constexpr uint32_t max_environment_path_byte_count = 32768u;

struct EnvironmentData {
    const char *path_utf8;
    uint32_t path_byte_count;
    float intensity;
    uint32_t enabled;
};

struct EnvironmentSnapshot {
    luisa::filesystem::path path;
    float intensity{};
    bool enabled{};
    uint64_t revision{};
};

static_assert(sizeof(EnvironmentData) == 24u);

[[nodiscard]] luisa::optional<EnvironmentSnapshot>
copy_environment(const EnvironmentData &data, uint64_t revision) noexcept;

} // namespace yutrel::unity
