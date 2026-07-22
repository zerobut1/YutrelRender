#include "ut/ut.hpp"

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
    return 0;
}();

int main(int argc, char* argv[])
{
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
}
