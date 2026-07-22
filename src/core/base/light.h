#pragma once

#include <luisa/dsl/syntax.h>

#include "base/texture.h"
#include "utils/command_buffer.h"
#include "utils/spectra.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;

class Light
{
public:
    struct Handle
    {
        uint instance_id;
        uint light_tag;
    };

    struct Evaluation
    {
        SampledSpectrum L;
        Float pdf;
        Float3 p;
        Float3 ng;
        [[nodiscard]] static auto zero(uint dimension) noexcept
        {
            return Evaluation{
                .L   = SampledSpectrum{dimension},
                .pdf = 0.0f,
                .p   = make_float3(0.0f),
                .ng  = make_float3(0.0f),
            };
        }
    };

    struct Sample
    {
        Evaluation eval;
        Float3 p;
        [[nodiscard]] static auto zero(uint dimension) noexcept
        {
            return Sample{.eval = Evaluation::zero(dimension), .p = make_float3()};
        }
    };

public:
    class Instance;
    class Closure;

public:
    Light() noexcept          = default;
    virtual ~Light() noexcept = default;

    Light(const Light&)            = delete;
    Light& operator=(const Light&) = delete;
    Light(Light&&)                 = delete;
    Light& operator=(Light&&)      = delete;

public:
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept = 0;
};

class Light::Instance
{
private:
    const Renderer& m_renderer;
    const Light* m_light;

public:
    explicit Instance(const Renderer& renderer, const Light* light) noexcept
        : m_renderer{renderer}, m_light{light} {}
    virtual ~Instance() noexcept = default;

    Instance()                           = delete;
    Instance(const Instance&)            = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&)                 = delete;
    Instance& operator=(Instance&&)      = delete;

public:
    template <typename T = Light>
        requires std::is_base_of_v<Light, T>
    [[nodiscard]] auto base() const noexcept
    {
        return static_cast<const T*>(m_light);
    }

    [[nodiscard]] auto& renderer() const noexcept { return m_renderer; }
    [[nodiscard]] virtual luisa::unique_ptr<Closure> closure(const SampledWavelengths& swl, Expr<float> time) const noexcept = 0;
};

class Light::Closure
{
private:
    const Instance* m_instance;

private:
    const SampledWavelengths& m_swl;
    Float m_time;

public:
    explicit Closure(const Instance* instance, const SampledWavelengths& swl, Expr<float> time) noexcept
        : m_instance{instance}, m_swl{swl}, m_time{time} {}
    virtual ~Closure() noexcept = default;

    Closure()                          = delete;
    Closure(const Closure&)            = delete;
    Closure& operator=(const Closure&) = delete;
    Closure(Closure&&)                 = delete;
    Closure& operator=(Closure&&)      = delete;

public:
    template <typename T = Instance>
        requires std::is_base_of_v<Instance, T>
    [[nodiscard]] auto instance() const noexcept
    {
        return static_cast<const T*>(m_instance);
    }

    [[nodiscard]] auto& swl() const noexcept { return m_swl; }
    [[nodiscard]] auto time() const noexcept { return m_time; }

    [[nodiscard]] virtual Evaluation evaluate(const Interaction& it_light, const Interaction& it_from) const noexcept = 0;
};

} // namespace Yutrel

LUISA_STRUCT(Yutrel::Light::Handle, instance_id, light_tag){};
