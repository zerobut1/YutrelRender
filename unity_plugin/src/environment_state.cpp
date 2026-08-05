#include "environment_state.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace yutrel::unity {

luisa::optional<EnvironmentSnapshot>
copy_environment(const EnvironmentData &data, uint64_t revision) noexcept {
    if (data.enabled > 1u || !std::isfinite(data.intensity) || data.intensity < 0.0f) {
        return luisa::nullopt;
    }
    if (data.enabled == 0u) {
        if (data.path_utf8 != nullptr || data.path_byte_count != 0u || data.intensity != 0.0f) {
            return luisa::nullopt;
        }
        return EnvironmentSnapshot{.revision = revision};
    }
    if (data.path_utf8 == nullptr || data.path_byte_count == 0u ||
        data.path_byte_count > max_environment_path_byte_count || data.intensity == 0.0f) {
        return luisa::nullopt;
    }

    auto path_begin = data.path_utf8;
    auto path_end = path_begin + data.path_byte_count;
    if (std::find(path_begin, path_end, '\0') != path_end) {
        return luisa::nullopt;
    }
    try {
        auto path = std::filesystem::u8path(path_begin, path_end);
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error) || error) {
            return luisa::nullopt;
        }
        return EnvironmentSnapshot{
            .path = std::move(path),
            .intensity = data.intensity,
            .enabled = true,
            .revision = revision,
        };
    } catch (...) {
        return luisa::nullopt;
    }
}

} // namespace yutrel::unity
