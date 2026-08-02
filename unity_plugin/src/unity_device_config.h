#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <IUnityGraphicsD3D12.h>

#include <luisa/backends/ext/dx_config_ext.h>
#include <luisa/core/stl/vector.h>

namespace yutrel::unity {

class UnityDeviceConfig final : public luisa::compute::DirectXDeviceConfigExt {
private:
    IUnityGraphicsD3D12v8 *_unity_graphics;
    Microsoft::WRL::ComPtr<IDXGIFactory4> _factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> _adapter;
    ID3D12Device *_device{nullptr};
    luisa::vector<UnityGraphicsD3D12ResourceState> _resource_states;

public:
    explicit UnityDeviceConfig(IUnityGraphicsD3D12v8 *unity_graphics) noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] luisa::optional<ExternalDevice> CreateExternalDevice() noexcept override;
    [[nodiscard]] ID3D12CommandQueue *CreateQueue(D3D12_COMMAND_LIST_TYPE type) noexcept override;
    [[nodiscard]] bool ExecuteCommandList(
        ID3D12CommandQueue *queue,
        ID3D12GraphicsCommandList *command_list) noexcept override;

    void register_resource(
        ID3D12Resource *resource,
        D3D12_RESOURCE_STATES state) noexcept;
};

}// namespace yutrel::unity
