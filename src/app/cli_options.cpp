#include "cli_options.h"

#include <charconv>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <luisa/core/logging.h>

namespace Yutrel
{
namespace
{

[[noreturn]] void fail(luisa::string message)
{
    throw std::runtime_error{message.c_str()};
}

[[nodiscard]] bool has_extension(std::filesystem::path const& path, std::string_view expected)
{
    auto extension = path.extension().string();
    for (auto& c : extension)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return extension == expected;
}

[[nodiscard]] uint parse_uint(std::string_view text, luisa::string_view option, bool require_positive)
{
    uint64_t value{};
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() ||
        value > std::numeric_limits<uint>::max() || (require_positive && value == 0u))
    {
        fail(luisa::format("Invalid value '{}' for {}.", text, option));
    }
    return static_cast<uint>(value);
}

[[nodiscard]] uint2 parse_resolution(std::string_view text)
{
    auto separator = text.find_first_of("xX");
    if (separator == std::string_view::npos ||
        text.find_first_of("xX", separator + 1u) != std::string_view::npos)
    {
        fail(luisa::format("Invalid resolution '{}'. Expected WIDTHxHEIGHT.", text));
    }
    auto width  = parse_uint(text.substr(0u, separator), "--resolution", true);
    auto height = parse_uint(text.substr(separator + 1u), "--resolution", true);
    if (static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > std::numeric_limits<uint>::max())
    {
        fail(luisa::format("Resolution '{}' exceeds the supported 32-bit pixel count.", text));
    }
    return make_uint2(width, height);
}

} // namespace

luisa::string command_line_usage(luisa::string_view bin)
{
    return luisa::format(
        "Usage: {} <backend> <scene> [--interactive|-i] [--headless] "
        "[--spp N] [--seed N] [--resolution WIDTHxHEIGHT] [--output PATH.exr]. "
        "<backend>: cuda, dx, vk, metal",
        bin);
}

CommandLineOptions parse_command_line(int argc, char* argv[])
{
    if (argc <= 1)
    {
        fail(command_line_usage(argc == 1 ? argv[0] : "Yutrel"));
    }

    CommandLineOptions options{.backend = argv[1]};
    bool has_scene_path = false;

    auto require_value = [&](int& i, luisa::string_view option) -> std::string_view
    {
        if (++i >= argc)
        {
            fail(luisa::format("Missing value for {}.", option));
        }
        auto value = std::string_view{argv[i]};
        if (value.empty() || value.front() == '-')
        {
            fail(luisa::format("Missing value for {}.", option));
        }
        return value;
    };

    for (int i = 2; i < argc; i++)
    {
        auto arg = std::string_view{argv[i]};
        if (arg == "--interactive" || arg == "-i")
        {
            if (options.interactive) { fail("Duplicate --interactive option."); }
            options.interactive = true;
            continue;
        }
        if (arg == "--headless")
        {
            if (options.headless) { fail("Duplicate --headless option."); }
            options.headless = true;
            continue;
        }
        if (arg == "--spp")
        {
            if (options.overrides.spp) { fail("Duplicate --spp option."); }
            options.overrides.spp = parse_uint(require_value(i, arg), "--spp", true);
            continue;
        }
        if (arg == "--seed")
        {
            if (options.overrides.seed) { fail("Duplicate --seed option."); }
            options.overrides.seed = parse_uint(require_value(i, arg), "--seed", false);
            continue;
        }
        if (arg == "--resolution")
        {
            if (options.overrides.resolution) { fail("Duplicate --resolution option."); }
            options.overrides.resolution = parse_resolution(require_value(i, arg));
            continue;
        }
        if (arg == "--output")
        {
            if (options.overrides.output) { fail("Duplicate --output option."); }
            auto output = std::filesystem::path{require_value(i, arg)};
            if (!has_extension(output, ".exr"))
            {
                fail(luisa::format("Output path '{}' must use the .exr extension.", output.string()));
            }
            options.overrides.output = std::filesystem::absolute(output).lexically_normal();
            continue;
        }
        if (!arg.empty() && arg.front() == '-')
        {
            fail(luisa::format("Unknown option '{}'. {}", arg, command_line_usage(argv[0])));
        }

        auto candidate = std::filesystem::path{arg};
        if (has_scene_path)
        {
            fail(luisa::format("Multiple scene files specified: '{}' and '{}'.",
                               options.scene_path.string(), candidate.string()));
        }
        options.scene_path = std::move(candidate);
        has_scene_path     = true;
    }

    if (!has_scene_path) { fail(command_line_usage(argv[0])); }
    if (options.interactive && options.headless)
    {
        fail("--interactive and --headless cannot be used together.");
    }
    return options;
}

} // namespace Yutrel
