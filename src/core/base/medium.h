#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/dsl/syntax.h>

#include "utils/command_buffer.h"
#include "utils/spectra.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Interaction;
class Renderer;

class Medium
{
public:
    static constexpr uint vacuum_tag = 0u;

    struct Properties
    {
        SampledSpectrum sigma_a;
        SampledSpectrum sigma_s;
        SampledSpectrum sigma_t;
        SampledSpectrum sigma_maj;
        Float g;

        [[nodiscard]] static Properties vacuum(uint dimension) noexcept
        {
            return {
                .sigma_a   = SampledSpectrum{dimension},
                .sigma_s   = SampledSpectrum{dimension},
                .sigma_t   = SampledSpectrum{dimension},
                .sigma_maj = SampledSpectrum{dimension},
                .g         = 0.0f,
            };
        }
    };

public:
    class Instance
    {
    private:
        const Renderer& _renderer;
        const Medium* _medium;

    public:
        Instance(const Renderer& renderer, const Medium* medium) noexcept
            : _renderer{renderer}, _medium{medium} {}
        virtual ~Instance() noexcept = default;

        template <typename T = Medium>
            requires std::is_base_of_v<Medium, T>
        [[nodiscard]] const T* base() const noexcept
        {
            return static_cast<const T*>(_medium);
        }

        [[nodiscard]] const Renderer& renderer() const noexcept { return _renderer; }
        [[nodiscard]] virtual Properties properties(const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept = 0;
    };

public:
    virtual ~Medium() noexcept                                                                                                = default;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept = 0;
};

} // namespace Yutrel
