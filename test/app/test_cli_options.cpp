// Test for Yutrel command-line option parsing and validation.

#include "ut/ut.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli_options.h"

using namespace Yutrel;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace
{

[[nodiscard]] CommandLineOptions parse(std::initializer_list<const char*> values)
{
    std::vector<std::string> storage;
    storage.reserve(values.size());
    for (auto value : values)
    {
        storage.emplace_back(value);
    }
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& value : storage)
    {
        argv.emplace_back(value.data());
    }
    return parse_command_line(static_cast<int>(argv.size()), argv.data());
}

[[nodiscard]] bool parse_fails(std::initializer_list<const char*> values)
{
    try
    {
        (void)parse(values);
        return false;
    }
    catch (const std::runtime_error&)
    {
        return true;
    }
}

} // namespace

static auto test_cli_options_registration = []
{
    "parse_all_overrides"_test = []
    {
        auto options = parse({"Yutrel", "dx", "scene.pbrt", "--headless", "--spp", "32", "--seed", "7", "--resolution", "640x480", "--output", "result.exr", "--world-up", "-1,2,0"});
        expect(options.backend == "dx");
        expect(options.scene_path == std::filesystem::path{"scene.pbrt"});
        expect(options.headless);
        expect(options.overrides.spp && *options.overrides.spp == 32u);
        expect(options.overrides.seed && *options.overrides.seed == 7u);
        expect(static_cast<bool>(options.overrides.resolution));
        expect(luisa::all(*options.overrides.resolution == luisa::make_uint2(640u, 480u)));
        expect(options.overrides.output && options.overrides.output->is_absolute());
        expect(static_cast<bool>(options.overrides.world_up));
        if (options.overrides.world_up)
        {
            expect(std::abs(options.overrides.world_up->x + 0.4472136f) < 1e-5f);
            expect(std::abs(options.overrides.world_up->y - 0.8944272f) < 1e-5f);
            expect(std::abs(options.overrides.world_up->z) < 1e-5f);
        }
    };

    "accept_scene_format_for_dispatch"_test = []
    {
        auto options = parse({"Yutrel", "dx", "scene.gltf", "--headless"});
        expect(options.scene_path == std::filesystem::path{"scene.gltf"});
        expect(options.headless);
    };

    "reject_invalid_options"_test = []
    {
        expect(parse_fails({"Yutrel", "dx"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--spp", "0"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--seed", "4294967296"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--resolution", "0x16"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--resolution", "65536x65536"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--output", "result.png"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--spp", "1", "--spp", "2"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--world-up", "0,0,0"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--world-up", "1,0"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--world-up", "1,0,0,0"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--world-up", "x,0,0"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--world-up", "inf,0,0"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--world-up", "1,0,0", "--world-up", "0,1,0"}));
        expect(parse_fails({"Yutrel", "dx", "scene.pbrt", "--unknown"}));
    };
    return 0;
}();

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
