#include "ut/ut.hpp"

#include <array>
#include <stdexcept>
#include <string>

#include "scene_loader.h"

using namespace Yutrel;
using namespace boost::ut;
using namespace boost::ut::literals;

static auto test_scene_loader_registration = []
{
    "reject_unsupported_scene_format"_test = []
    {
        auto rejected = false;
        try
        {
            (void)load_scene("scene.gltf", {});
        }
        catch (const std::runtime_error& error)
        {
            rejected = std::string{error.what()}.find(".gltf") != std::string::npos;
        }
        expect(rejected);
    };

    "dispatch_all_usd_extensions_case_insensitively"_test = []
    {
        constexpr std::array extensions{".usd", ".USDA", ".UsDc", ".USDZ"};
        for (auto extension : extensions)
        {
            auto dispatched = false;
            try
            {
                (void)load_scene(std::string{"test/app/missing"} + extension, {});
            }
            catch (const std::runtime_error& error)
            {
                auto message = std::string{error.what()};
                dispatched   = message.find("not a regular file") != std::string::npos &&
                               message.find("Unsupported scene format") == std::string::npos;
            }
            expect(dispatched);
        }
    };

    "load_usdc_scene"_test = []
    {
        auto spec = load_scene("scene/blender-basic/scene.usdc", {});
        expect(spec.instances().size() == 2u);
        expect(spec.lights().size() == 1u);
        expect(spec.cameras().size() == 1u);
    };
    return 0;
}();

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
