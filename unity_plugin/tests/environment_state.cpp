#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include "environment_state.h"

using namespace yutrel::unity;

namespace {

class TemporaryFile {
private:
    std::filesystem::path _path;

public:
    TemporaryFile()
        : _path{std::filesystem::temp_directory_path() /
                ("yutrel_environment_state_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 ".hdr")} {
        std::ofstream{_path, std::ios::binary}.put('\0');
    }

    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(_path, error);
    }

    [[nodiscard]] const auto &path() const noexcept { return _path; }
};

[[nodiscard]] std::string path_to_utf8(const std::filesystem::path &path) {
    auto value = path.u8string();
    return {reinterpret_cast<const char *>(value.data()), value.size()};
}

} // namespace

int main() {
    static_assert(sizeof(EnvironmentData) == 24u);

    auto disabled = copy_environment(EnvironmentData{}, 7u);
    if (!disabled || disabled->enabled || disabled->intensity != 0.0f ||
        !disabled->path.empty() || disabled->revision != 7u) {
        return 1;
    }

    auto noncanonical_disabled = EnvironmentData{
        .intensity = 1.0f,
        .enabled = 0u,
    };
    if (copy_environment(noncanonical_disabled, 8u)) {
        return 2;
    }

    TemporaryFile file;
    auto path_utf8 = path_to_utf8(file.path());
    auto valid = EnvironmentData{
        .path_utf8 = path_utf8.data(),
        .path_byte_count = static_cast<uint32_t>(path_utf8.size()),
        .intensity = 20000.0f,
        .enabled = 1u,
    };
    auto first = copy_environment(valid, 9u);
    if (!first || !first->enabled || first->path != file.path() ||
        first->intensity != 20000.0f || first->revision != 9u) {
        return 3;
    }

    valid.intensity = 10000.0f;
    auto second = copy_environment(valid, 10u);
    path_utf8.assign(path_utf8.size(), 'x');
    if (!second || second->path != file.path() || second->intensity != 10000.0f ||
        second->revision != 10u) {
        return 4;
    }

    auto embedded_nul = path_to_utf8(file.path());
    embedded_nul.insert(embedded_nul.begin() + embedded_nul.size() / 2u, '\0');
    valid.path_utf8 = embedded_nul.data();
    valid.path_byte_count = static_cast<uint32_t>(embedded_nul.size());
    if (copy_environment(valid, 11u)) {
        return 5;
    }

    const char invalid_utf8[]{static_cast<char>(0xff)};
    valid.path_utf8 = invalid_utf8;
    valid.path_byte_count = 1u;
    if (copy_environment(valid, 12u)) {
        return 6;
    }

    auto missing_file = file.path();
    missing_file += ".missing";
    auto missing_path = path_to_utf8(missing_file);
    valid.path_utf8 = missing_path.data();
    valid.path_byte_count = static_cast<uint32_t>(missing_path.size());
    if (copy_environment(valid, 13u)) {
        return 7;
    }

    auto original_path = path_to_utf8(file.path());
    valid.path_utf8 = original_path.data();
    valid.path_byte_count = static_cast<uint32_t>(original_path.size());
    valid.intensity = 0.0f;
    if (copy_environment(valid, 14u)) {
        return 8;
    }
    valid.intensity = std::numeric_limits<float>::infinity();
    if (copy_environment(valid, 15u)) {
        return 9;
    }
    valid.intensity = 1.0f;
    valid.enabled = 2u;
    if (copy_environment(valid, 16u)) {
        return 10;
    }

    std::string oversized(max_environment_path_byte_count + 1u, 'a');
    valid.path_utf8 = oversized.data();
    valid.path_byte_count = static_cast<uint32_t>(oversized.size());
    valid.enabled = 1u;
    if (copy_environment(valid, 17u)) {
        return 11;
    }
    return 0;
}
