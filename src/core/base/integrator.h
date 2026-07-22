#pragma once

#include <cmath>

#include <luisa/core/stl/memory.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/stream.h>

#include "base/camera.h"
#include "base/sampler.h"
#include "scene/spec_base.h"
#include "utils/command_buffer.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;
class LightSampler;

class Integrator
{
public:
    class Instance
    {
    private:
        Renderer& _renderer;
        const Integrator* _integrator;
        luisa::unique_ptr<Sampler::Instance> _sampler;
        luisa::unique_ptr<LightSampler> _light_sampler;

    public:
        Instance(Renderer& renderer, CommandBuffer& command_buffer, const Integrator* integrator, const Sampler* sampler) noexcept;
        virtual ~Instance() noexcept;

        [[nodiscard]] const Renderer& renderer() const noexcept { return _renderer; }
        [[nodiscard]] Renderer& renderer() noexcept { return _renderer; }
        [[nodiscard]] Sampler::Instance* sampler() const noexcept { return _sampler.get(); }
        [[nodiscard]] LightSampler* light_sampler() const noexcept { return _light_sampler.get(); }

        template <typename T = Integrator>
            requires std::is_base_of_v<Integrator, T>
        [[nodiscard]] const T* base() const noexcept
        {
            return static_cast<const T*>(_integrator);
        }

        virtual void render(Stream& stream, bool enable_display) = 0;
        virtual void render_interactive(Stream& stream)          = 0;
    };

public:
    virtual ~Integrator() noexcept = default;

    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Renderer& renderer, CommandBuffer& command_buffer, const Sampler* sampler) const noexcept = 0;
};

class ProgressiveIntegrator : public Integrator
{
public:
    class Instance : public Integrator::Instance
    {
    public:
        Instance(Renderer& renderer, CommandBuffer& command_buffer, const ProgressiveIntegrator* integrator, const Sampler* sampler) noexcept;

        void render(Stream& stream, bool enable_display) override;
        void render_interactive(Stream& stream) override;

    protected:
        [[nodiscard]] uint max_depth() const noexcept { return base<ProgressiveIntegrator>()->max_depth(); }
        [[nodiscard]] virtual Float3 Li(const Camera::Instance* camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept = 0;

    private:
        void render_one_camera(CommandBuffer& command_buffer, Camera::Instance* camera);
    };

private:
    uint _max_depth;

public:
    explicit ProgressiveIntegrator(uint max_depth) noexcept;

    [[nodiscard]] uint max_depth() const noexcept { return _max_depth; }
};

} // namespace Yutrel
