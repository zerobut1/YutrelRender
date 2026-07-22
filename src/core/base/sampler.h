#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/dsl/syntax.h>

#include "utils/command_buffer.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;

class Sampler
{
public:
    class Instance
    {
    private:
        const Renderer& _renderer;
        const Sampler* _sampler;

    public:
        Instance(const Renderer& renderer, const Sampler* sampler) noexcept;
        virtual ~Instance() noexcept = default;

        Instance() noexcept                           = delete;
        Instance(const Instance&) noexcept            = delete;
        Instance(Instance&&) noexcept                 = delete;
        Instance& operator=(const Instance&) noexcept = delete;
        Instance& operator=(Instance&&) noexcept      = delete;

        [[nodiscard]] const Renderer& renderer() const noexcept { return _renderer; }

        template <typename T = Sampler>
            requires std::is_base_of_v<Sampler, T>
        [[nodiscard]] const T* base() const noexcept
        {
            return static_cast<const T*>(_sampler);
        }

        virtual void reset(CommandBuffer& command_buffer, uint2 resolution, uint state_count) noexcept = 0;
        virtual void start(UInt2 pixel, UInt index) noexcept                                           = 0;
        [[nodiscard]] virtual Float generate_1d() noexcept                                             = 0;
        [[nodiscard]] virtual Float2 generate_2d() noexcept                                            = 0;
        [[nodiscard]] virtual Float2 generate_pixel_2d() noexcept { return generate_2d(); }
    };

private:
    uint _spp;

public:
    explicit Sampler(uint spp) noexcept : _spp{spp} {}
    virtual ~Sampler() noexcept = default;

    Sampler() noexcept                          = delete;
    Sampler(const Sampler&) noexcept            = delete;
    Sampler(Sampler&&) noexcept                 = delete;
    Sampler& operator=(const Sampler&) noexcept = delete;
    Sampler& operator=(Sampler&&) noexcept      = delete;

    [[nodiscard]] uint spp() const noexcept { return _spp; }
    [[nodiscard]] virtual uint seed() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(const Renderer& renderer) const noexcept = 0;
};
} // namespace Yutrel
