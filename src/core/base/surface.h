#pragma once

#include <luisa/core/stl.h>
#include <luisa/core/stl/optional.h>
#include <luisa/dsl/syntax.h>

#include "base/texture.h"
#include "utils/command_buffer.h"
#include "utils/polymorphic_closure.h"
#include "utils/scattering.h"
#include "utils/spectra.h"

namespace Yutrel
{
class Renderer;
class Interaction;

enum class ScatterFlags : uint
{
    Reflection   = 1u,
    Transmission = 2u,
    All          = 3u,
};

[[nodiscard]] constexpr bool has_scatter_flag(ScatterFlags flags, ScatterFlags flag) noexcept
{
    return (static_cast<uint>(flags) & static_cast<uint>(flag)) != 0u;
}

class Surface
{
public:
    static constexpr auto event_reflect  = 0x00u;
    static constexpr auto event_enter    = 0x01u;
    static constexpr auto event_exit     = 0x02u;
    static constexpr auto event_transmit = event_enter | event_exit;
    static constexpr auto event_through  = 0x04u;

    static constexpr auto lobe_reflection   = 1u << 0u;
    static constexpr auto lobe_transmission = 1u << 1u;
    static constexpr auto lobe_diffuse      = 1u << 2u;
    static constexpr auto lobe_glossy       = 1u << 3u;
    static constexpr auto lobe_delta        = 1u << 4u;

    static constexpr auto property_reflective   = 1u << 0u;
    static constexpr auto property_transmissive = 1u << 1u;

    struct Evaluation
    {
        SampledSpectrum f;
        Float pdf;
        SampledSpectrum f_diffuse;
        Float pdf_diffuse;
        [[nodiscard]] static auto zero(uint dimension) noexcept
        {
            return Evaluation{
                .f           = SampledSpectrum{dimension},
                .pdf         = 0.f,
                .f_diffuse   = SampledSpectrum{dimension},
                .pdf_diffuse = 0.f};
        }
    };

    struct Sample
    {
        Evaluation eval;
        Float3 wi;
        UInt event;
        // eval.pdf is the sampled random-walk path PDF used for throughput.
        // pdf_mis is the directional PDF used when the sampled ray hits a light.
        Float pdf_mis;
        Bool delta;
        // Relative IOR along the sampled ray (eta_t / eta_i), matching PBRT's
        // BSDFSample::eta convention. Reflection samples keep this at one.
        Float eta;

        [[nodiscard]] static auto zero(uint dimension) noexcept
        {
            return Sample{
                .eval    = Evaluation::zero(dimension),
                .wi      = make_float3(0.f, 0.f, 1.f),
                .event   = Surface::event_reflect,
                .pdf_mis = 0.0f,
                .delta   = false,
                .eta     = 1.0f};
        }
    };

public:
    class Instance;
    class Closure;

public:
    explicit Surface(bool two_sided) noexcept;
    virtual ~Surface() noexcept = default;

    Surface()                          = delete;
    Surface(const Surface&)            = delete;
    Surface& operator=(const Surface&) = delete;
    Surface(Surface&&)                 = delete;
    Surface& operator=(Surface&&)      = delete;

public:
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] virtual bool maybe_non_opaque() const noexcept { return false; }
    [[nodiscard]] virtual uint properties() const noexcept = 0;
    [[nodiscard]] bool is_reflective() const noexcept { return (properties() & property_reflective) != 0u; }
    [[nodiscard]] bool is_transmissive() const noexcept { return (properties() & property_transmissive) != 0u; }
    [[nodiscard]] bool two_sided() const noexcept { return m_two_sided; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept = 0;

private:
    bool m_two_sided;
};

class Surface::Instance
{
private:
    const Renderer& m_renderer;
    const Surface* m_surface;

public:
    explicit Instance(const Renderer& renderer, const Surface* surface) noexcept
        : m_renderer{renderer}, m_surface{surface} {}
    virtual ~Instance() noexcept = default;

    Instance()                           = delete;
    Instance(const Instance&)            = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&)                 = delete;
    Instance& operator=(Instance&&)      = delete;

public:
    template <typename T = Surface>
        requires std::is_base_of_v<Surface, T>
    [[nodiscard]] auto base() const noexcept
    {
        return static_cast<const T*>(m_surface);
    }
    [[nodiscard]] auto& renderer() const noexcept { return m_renderer; }
    [[nodiscard]] virtual bool maybe_non_opaque() const noexcept { return false; }
    [[nodiscard]] virtual luisa::optional<Float> evaluate_opacity(
        const Interaction&, Expr<float>) const noexcept { return luisa::nullopt; }
    void closure(PolymorphicCall<Closure>& call, const Interaction& it, Expr<float3> wo,
                 SampledWavelengths& swl, Expr<float> time, Expr<float> eta_i) const noexcept;

    [[nodiscard]] virtual luisa::string closure_identifier() const noexcept                                                   = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Closure> create_closure(SampledWavelengths& swl, Expr<float> time) const noexcept = 0;
    virtual void populate_closure(Closure* closure, const Interaction& it,
                                  Expr<float3> wo, Expr<float> eta_i) const noexcept = 0;
};

class Surface::Closure : public PolymorphicClosure
{
private:
    const Renderer& m_renderer;
    const SampledWavelengths& m_swl;
    Float m_time;

public:
    explicit Closure(const Renderer& renderer, const SampledWavelengths& swl, Float time) noexcept
        : m_renderer{renderer}, m_swl{swl}, m_time{time} {}
    virtual ~Closure() noexcept override = default;

    Closure()                          = delete;
    Closure(const Closure&)            = delete;
    Closure& operator=(const Closure&) = delete;
    Closure(Closure&&)                 = delete;
    Closure& operator=(Closure&&)      = delete;

public:
    [[nodiscard]] auto& renderer() const noexcept { return m_renderer; }
    [[nodiscard]] auto& swl() const noexcept { return m_swl; }
    [[nodiscard]] auto time() const noexcept { return m_time; }
    [[nodiscard]] virtual const Interaction& it() const noexcept = 0;
    [[nodiscard]] virtual UInt lobe_flags() const noexcept = 0;
    [[nodiscard]] virtual luisa::optional<Float> eta() const noexcept { return luisa::nullopt; }

    [[nodiscard]] Surface::Sample sample(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u,
                                         TransportMode mode = TransportMode::RADIANCE,
                                         ScatterFlags flags = ScatterFlags::All) const noexcept;
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode = TransportMode::RADIANCE,
                                               ScatterFlags flags = ScatterFlags::All) const noexcept;

private:
    [[nodiscard]] virtual Surface::Sample sample_impl(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u,
                                                      TransportMode mode, ScatterFlags flags) const noexcept = 0;
    [[nodiscard]] virtual Surface::Evaluation evaluate_impl(Expr<float3> wo, Expr<float3> wi,
                                                            TransportMode mode, ScatterFlags flags) const noexcept = 0;
};
} // namespace Yutrel
