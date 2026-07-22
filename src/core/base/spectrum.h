#pragma once

#include <luisa/core/stl.h>
#include <luisa/dsl/syntax.h>

#include "base/spd.h"
#include "utils/spectra.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;
class CommandBuffer;

class Spectrum
{
public:
    struct Decode
    {
        SampledSpectrum value;
        Float strength;
        [[nodiscard]] static auto constant(uint dim, float value) noexcept
        {
            return Decode{.value = {dim, value}, .strength = value};
        }
        [[nodiscard]] static auto one(uint dim) noexcept { return constant(dim, 1.f); }
        [[nodiscard]] static auto zero(uint dim) noexcept { return constant(dim, 0.f); }
    };

public:
    class Instance
    {
    private:
        const Renderer& m_renderer;
        const Spectrum* m_spectrum;

        SPD m_cie_x;
        SPD m_cie_y;
        SPD m_cie_z;

    public:
        explicit Instance(Renderer& renderer, CommandBuffer& command_buffer, const Spectrum* spectrum) noexcept;
        ~Instance() noexcept = default;

    public:
        template <typename T = Spectrum>
            requires std::is_base_of_v<Spectrum, T>
        [[nodiscard]] auto base() const noexcept
        {
            return static_cast<const T*>(m_spectrum);
        }
        [[nodiscard]] auto& renderer() const noexcept { return m_renderer; }

        [[nodiscard]] virtual SampledWavelengths sample(Expr<float> u) const noexcept                                = 0;
        [[nodiscard]] virtual Float4 encode_srgb_albedo(Expr<float3> rgb) const noexcept                             = 0;
        [[nodiscard]] virtual Float4 encode_srgb_unbounded(Expr<float3> rgb) const noexcept                          = 0;
        [[nodiscard]] virtual Float4 encode_srgb_illuminant(Expr<float3> rgb) const noexcept                         = 0;
        [[nodiscard]] virtual Decode decode_albedo(const SampledWavelengths& swl, Expr<float4> v) const noexcept     = 0;
        [[nodiscard]] virtual Decode decode_unbounded(const SampledWavelengths& swl, Expr<float4> v) const noexcept  = 0;
        [[nodiscard]] virtual Decode decode_illuminant(const SampledWavelengths& swl, Expr<float4> v) const noexcept = 0;
        [[nodiscard]] virtual Float cie_y(const SampledWavelengths& swl, const SampledSpectrum& sp) const noexcept;
        [[nodiscard]] virtual Float3 cie_xyz(const SampledWavelengths& swl, const SampledSpectrum& sp) const noexcept;
        [[nodiscard]] virtual Float3 srgb(const SampledWavelengths& swl, const SampledSpectrum& sp) const noexcept;
    };

public:
    Spectrum() noexcept          = default;
    virtual ~Spectrum() noexcept = default;

public:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept = 0;

    [[nodiscard]] virtual bool is_fixed() const noexcept                                  = 0;
    [[nodiscard]] virtual uint dimension() const noexcept                                 = 0;
    [[nodiscard]] virtual float4 encode_static_srgb_albedo(float3 rgb) const noexcept     = 0;
    [[nodiscard]] virtual float4 encode_static_srgb_unbounded(float3 rgb) const noexcept  = 0;
    [[nodiscard]] virtual float4 encode_static_srgb_illuminant(float3 rgb) const noexcept = 0;
};

} // namespace Yutrel
