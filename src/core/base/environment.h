#pragma once

#include "base/light.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;
struct ExternalDirectionalLightState;

class Environment
{
public:
    using Evaluation = Light::Evaluation;

    struct Sample
    {
        Evaluation eval;
        Float3 wi;
        Bool delta;

        [[nodiscard]] static auto zero(uint dimension) noexcept
        {
            return Sample{
                .eval = Evaluation::zero(dimension),
                .wi = make_float3(0.0f),
                .delta = false,
            };
        }
    };

    class Instance
    {
    private:
        const Renderer& _renderer;
        const Environment* _environment;

    public:
        Instance(const Renderer& renderer, const Environment* environment) noexcept
            : _renderer{renderer}, _environment{environment} {}
        virtual ~Instance() noexcept = default;

        Instance()                              = delete;
        Instance(const Instance&)               = delete;
        Instance& operator=(const Instance&)    = delete;
        Instance(Instance&&)                    = delete;
        Instance& operator=(Instance&&)         = delete;

        template<typename T = Environment>
            requires std::is_base_of_v<Environment, T>
        [[nodiscard]] auto base() const noexcept
        {
            return static_cast<const T*>(_environment);
        }

        [[nodiscard]] auto& renderer() const noexcept { return _renderer; }

        [[nodiscard]] virtual Evaluation evaluate(
            Expr<float3> wi,
            const SampledWavelengths& swl,
            Expr<float> time,
            bool allow_incomplete_pdf) const noexcept = 0;

        [[nodiscard]] virtual Sample sample(
            const SampledWavelengths& swl,
            Expr<float> time,
            Expr<float2> u,
            bool allow_incomplete_pdf) const noexcept = 0;

        [[nodiscard]] virtual bool supports_external_directional_light() const noexcept
        {
            return false;
        }

        virtual void update_external_directional_light(
            CommandBuffer&,
            const ExternalDirectionalLightState&) noexcept {}
    };

public:
    Environment() noexcept = default;
    virtual ~Environment() noexcept = default;

    Environment(const Environment&)            = delete;
    Environment& operator=(const Environment&) = delete;
    Environment(Environment&&)                 = delete;
    Environment& operator=(Environment&&)      = delete;

    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept = 0;
};

} // namespace Yutrel
