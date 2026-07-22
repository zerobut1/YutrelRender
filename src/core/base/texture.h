#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/dsl/syntax.h>

#include "base/spectrum.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;
using TextureSampler = compute::Sampler;

class Renderer;
class CommandBuffer;
class Interaction;

class Texture
{
public:
    enum class Encoding : uint
    {
        LINEAR,
        SRGB,
        GAMMA,
    };

public:
    class Instance
    {
    private:
        const Renderer& m_renderer;
        const Texture* m_texture;

    public:
        explicit Instance(const Renderer& renderer, const Texture* texture) noexcept
            : m_renderer(renderer), m_texture(texture) {}
        virtual ~Instance() noexcept = default;

        template <typename T = Texture>
            requires std::is_base_of_v<Texture, T>
        [[nodiscard]] auto base() const noexcept
        {
            return static_cast<const T*>(m_texture);
        }
        Instance()                           = delete;
        Instance(const Instance&)            = delete;
        Instance& operator=(const Instance&) = delete;
        Instance(Instance&&)                 = delete;
        Instance& operator=(Instance&&)      = delete;

    public:
        [[nodiscard]] auto& renderer() const noexcept { return m_renderer; }

        [[nodiscard]] virtual Float4 evaluate(const Interaction& it, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Spectrum::Decode evaluate_albedo_spectrum(
            const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Spectrum::Decode evaluate_unbounded_spectrum(
            const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Spectrum::Decode evaluate_illuminant_spectrum(
            const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept;

    protected:
        [[nodiscard]] Spectrum::Decode evaluate_static_albedo_spectrum_impl(
            const SampledWavelengths& swl, float4 v) const noexcept;
        [[nodiscard]] Spectrum::Decode evaluate_static_unbounded_spectrum_impl(
            const SampledWavelengths& swl, float4 v) const noexcept;
        [[nodiscard]] Spectrum::Decode evaluate_static_illuminant_spectrum_impl(
            const SampledWavelengths& swl, float4 v) const noexcept;
    };

public:
    Texture() noexcept          = default;
    virtual ~Texture() noexcept = default;

    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&)                 = delete;
    Texture& operator=(Texture&&)      = delete;

public:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept = 0;

    [[nodiscard]] virtual luisa::optional<float4> evaluate_static() const noexcept { return luisa::nullopt; }
    [[nodiscard]] virtual uint channels() const noexcept { return 4u; }
    [[nodiscard]] virtual uint2 resolution() const noexcept { return make_uint2(1u); }
    // TODO
    // is black
    // is constant
};
} // namespace Yutrel
