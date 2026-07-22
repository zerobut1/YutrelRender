#pragma once

#include <utility>

#include <luisa/core/stl/optional.h>

#include "base/interaction.h"
#include "base/surface.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class Texture;

struct LayeredSurfaceParams
{
    SurfaceRef top;
    SurfaceRef bottom;
    luisa::optional<TextureRef> thickness;
    luisa::optional<TextureRef> albedo;
    luisa::optional<TextureRef> g;
    uint max_depth{10u};
    uint samples{1u};
    bool two_sided{false};
};

class Layered final : public Surface
{
public:
    class Instance;
    class Closure;

private:
    const Surface* m_top;
    const Surface* m_bottom;
    const Texture* m_thickness;
    const Texture* m_albedo;
    const Texture* m_g;
    uint m_max_depth;
    uint m_samples;

public:
    Layered(const Surface* top, const Surface* bottom,
            const Texture* thickness, const Texture* albedo, const Texture* g,
            uint max_depth, uint samples, bool two_sided) noexcept;

    [[nodiscard]] uint max_depth() const noexcept { return m_max_depth; }
    [[nodiscard]] uint samples() const noexcept { return m_samples; }
    [[nodiscard]] uint properties() const noexcept override;
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class Layered::Instance final : public Surface::Instance
{
private:
    luisa::unique_ptr<Surface::Instance> m_top;
    luisa::unique_ptr<Surface::Instance> m_bottom;
    const Texture::Instance* m_thickness;
    const Texture::Instance* m_albedo;
    const Texture::Instance* m_g;

public:
    Instance(const Renderer& renderer, const Layered* surface,
             luisa::unique_ptr<Surface::Instance> top,
             luisa::unique_ptr<Surface::Instance> bottom,
             const Texture::Instance* thickness,
             const Texture::Instance* albedo,
             const Texture::Instance* g) noexcept;

    [[nodiscard]] luisa::string closure_identifier() const noexcept override;
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(
        SampledWavelengths& swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure* closure, const Interaction& it,
                          Expr<float3> wo, Expr<float> eta_i) const noexcept override;
};

class Layered::Closure final : public Surface::Closure
{
public:
    struct Context
    {
        Interaction it;
        Float thickness;
        SampledSpectrum albedo;
        Float g;
        UInt max_depth;
        UInt samples;
    };

private:
    class Impl;
    luisa::unique_ptr<Surface::Closure> m_top;
    luisa::unique_ptr<Surface::Closure> m_bottom;
    luisa::unique_ptr<Impl> m_impl;

public:
    Closure(const Renderer& renderer, const SampledWavelengths& swl, Expr<float> time,
            luisa::unique_ptr<Surface::Closure> top,
            luisa::unique_ptr<Surface::Closure> bottom) noexcept;
    ~Closure() noexcept override;

    [[nodiscard]] Surface::Closure* top() const noexcept { return m_top.get(); }
    [[nodiscard]] Surface::Closure* bottom() const noexcept { return m_bottom.get(); }
    [[nodiscard]] const Interaction& it() const noexcept override { return context<Context>().it; }
    [[nodiscard]] UInt lobe_flags() const noexcept override;
    void pre_eval() noexcept override;
    void post_eval() noexcept override;

private:
    [[nodiscard]] Surface::Sample sample_impl(Expr<float3> wo, Expr<float> u_lobe,
                                              Expr<float2> u, TransportMode mode,
                                              ScatterFlags flags) const noexcept override;
    [[nodiscard]] Surface::Evaluation evaluate_impl(Expr<float3> wo, Expr<float3> wi,
                                                    TransportMode mode,
                                                    ScatterFlags flags) const noexcept override;
};

class LayeredSurfaceSpec final : public SurfaceSpec
{
private:
    LayeredSurfaceParams m_params;

public:
    explicit LayeredSurfaceSpec(LayeredSurfaceParams params) noexcept
        : m_params{std::move(params)} {}

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override;
    [[nodiscard]] const Surface* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel

