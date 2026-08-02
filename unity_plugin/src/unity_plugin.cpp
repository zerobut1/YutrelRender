#include "unity_device_config.h"

#include <IUnityGraphics.h>
#include <IUnityInterface.h>

#include <Windows.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>

#include <luisa/backends/ext/native_resource_ext.hpp>
#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>

namespace yutrel::unity {

using namespace luisa;
using namespace luisa::compute;

constexpr uint32_t clear_event_abi_version = 1u;
constexpr int clear_event_id = 0;

struct ClearEventData {
    uint32_t abi_version;
    uint32_t struct_size;
    ID3D12Resource *output;
};

struct NativeTextureDesc {
    D3D12_RESOURCE_STATES init_state;
    DXGI_FORMAT custom_format;
    bool allow_uav;
};

class Plugin final {
private:
    Context _context;
    Device _device;
    Stream _stream;
    UnityDeviceConfig *_unity_config{nullptr};
    Shader2D<Image<float>> _clear_shader;

public:
    Plugin(luisa::string module_path, luisa::unique_ptr<UnityDeviceConfig> config)
        : _context{module_path},
          _unity_config{config.get()} {
        DeviceConfig device_config{
            .extension = std::move(config),
            .inqueue_buffer_limit = false,
        };
        _device = _context.create_device("dx", &device_config);
        _stream = _device.create_stream(StreamTag::GRAPHICS);

        Kernel2D clear = [](ImageFloat output) noexcept {
            output.write(
                dispatch_id().xy(),
                make_float4(0.18f, 0.36f, 0.72f, 1.0f));
        };
        _clear_shader = _device.compile(clear);
    }

    ~Plugin() noexcept {
        try {
            if (_stream) {
                _stream << synchronize();
            }
        } catch (...) {
            LUISA_WARNING_WITH_LOCATION("Failed to synchronize YutrelUnityPlugin during shutdown.");
        }
    }

    void clear(const ClearEventData &data) noexcept {
        try {
            clear_unchecked(data);
        } catch (const std::exception &exception) {
            LUISA_WARNING_WITH_LOCATION(
                "Luisa fixed-color render event failed: {}",
                exception.what());
        } catch (...) {
            LUISA_WARNING_WITH_LOCATION("Luisa fixed-color render event failed with an unknown error.");
        }
    }

private:
    void clear_unchecked(const ClearEventData &data) {
        if (data.output == nullptr) {
            LUISA_WARNING_WITH_LOCATION("Unity fixed-color output resource is null.");
            return;
        }

        auto description = data.output->GetDesc();
        if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
            description.Format != DXGI_FORMAT_R16G16B16A16_FLOAT ||
            description.Width == 0u ||
            description.Width > std::numeric_limits<uint32_t>::max() ||
            description.Height == 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "Unsupported Unity fixed-color output: dimension={}, format={}, size={}x{}.",
                static_cast<uint32_t>(description.Dimension),
                static_cast<uint32_t>(description.Format),
                description.Width,
                description.Height);
            return;
        }

        auto native_resources = _device.extension<NativeResourceExt>();
        if (native_resources == nullptr) {
            LUISA_WARNING_WITH_LOCATION("Luisa DX NativeResourceExt is unavailable.");
            return;
        }

        NativeTextureDesc native_description{
            .init_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            .custom_format = DXGI_FORMAT_UNKNOWN,
            .allow_uav = true,
        };
        _unity_config->register_resource(
            data.output,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        auto output = native_resources->create_native_image<float>(
            data.output,
            static_cast<uint32_t>(description.Width),
            description.Height,
            PixelStorage::HALF4,
            1u,
            &native_description);

        auto commands = CommandList::create(1u, 1u);
        commands << _clear_shader(output).dispatch(output.size());
        commands.add_callback([output = std::move(output)]() mutable {});
        _stream << commands.commit() << synchronize();
    }
};

namespace {

IUnityGraphicsD3D12v8 *unity_graphics = nullptr;
luisa::unique_ptr<Plugin> plugin;
std::mutex plugin_mutex;

[[nodiscard]] luisa::string current_module_path() {
    HMODULE module = nullptr;
    auto address = reinterpret_cast<LPCSTR>(&current_module_path);
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address,
            &module)) {
        return {};
    }

    luisa::vector<char> path(32768u, '\0');
    auto length = GetModuleFileNameA(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0u || length >= path.size()) {
        return {};
    }
    return luisa::string{path.data(), length};
}

void UNITY_INTERFACE_API on_render_event(int event_id, void *event_data) {
    auto data = std::unique_ptr<ClearEventData>{
        static_cast<ClearEventData *>(event_data)};
    if (event_id != clear_event_id || data == nullptr) {
        return;
    }
    if (data->abi_version != clear_event_abi_version ||
        data->struct_size != sizeof(ClearEventData)) {
        return;
    }

    std::scoped_lock lock{plugin_mutex};
    if (plugin != nullptr) {
        plugin->clear(*data);
    }
}

}// namespace

}// namespace yutrel::unity

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces *interfaces) {
    using namespace yutrel::unity;
    std::scoped_lock lock{plugin_mutex};
    unity_graphics = interfaces == nullptr
        ? nullptr
        : interfaces->Get<IUnityGraphicsD3D12v8>();
    if (unity_graphics == nullptr) {
        return;
    }

    UnityD3D12PluginEventConfig config{};
    config.graphicsQueueAccess = kUnityD3D12GraphicsQueueAccess_Allow;
    config.flags = kUnityD3D12EventConfigFlag_FlushCommandBuffers |
                   kUnityD3D12EventConfigFlag_SyncWorkerThreads |
                   kUnityD3D12EventConfigFlag_ModifiesCommandBuffersState;
    config.ensureActiveRenderTextureIsBound = false;
    unity_graphics->ConfigureEvent(clear_event_id, &config);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
UnityPluginUnload() {
    using namespace yutrel::unity;
    std::scoped_lock lock{plugin_mutex};
    plugin = nullptr;
    unity_graphics = nullptr;
}

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API
YutrelUnityInitialize() {
    using namespace yutrel::unity;
    std::scoped_lock lock{plugin_mutex};
    if (plugin != nullptr) {
        return 0;
    }
    if (unity_graphics == nullptr) {
        return 1;
    }

    try {
        auto config = luisa::make_unique<UnityDeviceConfig>(unity_graphics);
        if (!config->valid()) {
            return 2;
        }
        auto module_path = current_module_path();
        if (module_path.empty()) {
            return 3;
        }

        plugin = luisa::make_unique<Plugin>(
            std::move(module_path),
            std::move(config));
        return 0;
    } catch (const std::exception &exception) {
        LUISA_WARNING_WITH_LOCATION(
            "YutrelUnityPlugin initialization failed: {}",
            exception.what());
    } catch (...) {
        LUISA_WARNING_WITH_LOCATION("YutrelUnityPlugin initialization failed with an unknown error.");
    }
    plugin = nullptr;
    return 4;
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
YutrelUnityShutdown() {
    using namespace yutrel::unity;
    std::scoped_lock lock{plugin_mutex};
    plugin = nullptr;
}

extern "C" UNITY_INTERFACE_EXPORT UnityRenderingEventAndData UNITY_INTERFACE_API
YutrelUnityGetRenderEventFunc() {
    return yutrel::unity::on_render_event;
}

extern "C" UNITY_INTERFACE_EXPORT void *UNITY_INTERFACE_API
YutrelUnityCreateClearEvent(const yutrel::unity::ClearEventData *data) {
    using namespace yutrel::unity;
    if (data == nullptr ||
        data->abi_version != clear_event_abi_version ||
        data->struct_size != sizeof(ClearEventData) ||
        data->output == nullptr) {
        return nullptr;
    }
    return new (std::nothrow) ClearEventData{*data};
}
