#pragma once

#include <luisa/gui/window.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;
class Scene;
class SceneSpec;

struct ApplicationOptions
{
    luisa::string_view bin;
    luisa::string_view backend;
    bool interactive{false};
    bool headless{false};
};

class Application final
{
private:
    Context m_context;
    Device m_device;
    Stream m_stream;

    bool m_interactive{false};
    bool m_headless{false};

    luisa::unique_ptr<Scene> m_scene;
    luisa::unique_ptr<Renderer> m_renderer;

public:
    Application(ApplicationOptions options, const SceneSpec& scene);
    ~Application() noexcept;

    Application() noexcept                     = delete;
    Application(const Application&)            = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&)                 = delete;
    Application& operator=(Application&&)      = delete;

public:
    void run();
};
} // namespace Yutrel
