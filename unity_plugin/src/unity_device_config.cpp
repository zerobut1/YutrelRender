#include "unity_device_config.h"

#include <cstring>

namespace yutrel::unity {

using Microsoft::WRL::ComPtr;
using luisa::compute::DirectXDeviceConfigExt;

UnityDeviceConfig::UnityDeviceConfig(IUnityGraphicsD3D12v8 *unity_graphics) noexcept
    : _unity_graphics{unity_graphics} {
    if (_unity_graphics == nullptr) {
        return;
    }

    _device = _unity_graphics->GetDevice();
    if (_device == nullptr ||
        FAILED(CreateDXGIFactory2(0u, IID_PPV_ARGS(_factory.GetAddressOf())))) {
        _device = nullptr;
        return;
    }

    auto device_luid = _device->GetAdapterLuid();
    for (auto adapter_index = 0u;; adapter_index++) {
        ComPtr<IDXGIAdapter1> candidate;
        if (_factory->EnumAdapters1(adapter_index, candidate.GetAddressOf()) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(candidate->GetDesc1(&description)) &&
            std::memcmp(&description.AdapterLuid, &device_luid, sizeof(LUID)) == 0) {
            _adapter = std::move(candidate);
            break;
        }
    }

    if (_adapter == nullptr) {
        _device = nullptr;
    }
}

bool UnityDeviceConfig::valid() const noexcept {
    return _unity_graphics != nullptr &&
           _device != nullptr &&
           _adapter != nullptr &&
           _factory != nullptr;
}

luisa::optional<DirectXDeviceConfigExt::ExternalDevice>
UnityDeviceConfig::CreateExternalDevice() noexcept {
    if (!valid()) {
        return luisa::nullopt;
    }
    return ExternalDevice{
        .device = _device,
        .adapter = _adapter.Get(),
        .factory = _factory.Get(),
    };
}

ID3D12CommandQueue *UnityDeviceConfig::CreateQueue(D3D12_COMMAND_LIST_TYPE type) noexcept {
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT || _unity_graphics == nullptr) {
        return nullptr;
    }
    return _unity_graphics->GetCommandQueue();
}

bool UnityDeviceConfig::ExecuteCommandList(
    ID3D12CommandQueue *queue,
    ID3D12GraphicsCommandList *command_list) noexcept {
    if (_unity_graphics == nullptr ||
        command_list == nullptr ||
        queue != _unity_graphics->GetCommandQueue()) {
        return false;
    }

    _unity_graphics->ExecuteCommandList(
        command_list,
        static_cast<int>(_resource_states.size()),
        _resource_states.data());
    _resource_states.clear();
    return true;
}

void UnityDeviceConfig::register_resource(
    ID3D12Resource *resource,
    D3D12_RESOURCE_STATES state) noexcept {
    _resource_states.emplace_back(UnityGraphicsD3D12ResourceState{
        .resource = resource,
        .expected = state,
        .current = state,
    });
}

}// namespace yutrel::unity
