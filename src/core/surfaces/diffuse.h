#pragma once

#include "base/interaction.h"
#include "base/surface.h"
#include "scene/spec_base.h"
#include "utils/scattering.h"

namespace Yutrel
{
class Texture;

class Diffuse : public Surface
{
public:
    class Instance;
    class Closure;

private:
    const Texture* m_reflectance;

public:
    Diffuse(const Texture* reflectance, bool two_sided) noexcept;

public:
    [[nodiscard]] uint properties() const noexcept override { return property_reflective; }
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class Diffuse::Instance : public Surface::Instance
{
private:
    const Texture::Instance* m_reflectance;

public:
    explicit Instance(const Renderer& renderer, const Diffuse* surface, const Texture::Instance* reflectance) noexcept
        : Surface::Instance(renderer, surface), m_reflectance(reflectance) {}

public:
    [[nodiscard]] luisa::string closure_identifier() const noexcept override { return "Diffuse"; }
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(SampledWavelengths& swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure* closure, const Interaction& it,
                          Expr<float3> wo, Expr<float> eta_i) const noexcept override;
};

class Diffuse::Closure : public Surface::Closure
{
private:
    luisa::unique_ptr<BxDF> m_bxdf;

public:
    struct Context
    {
        Interaction it;
        SampledSpectrum reflectance;
    };

public:
    using Surface::Closure::Closure;

    [[nodiscard]] virtual const Interaction& it() const noexcept override { return context<Context>().it; };
    [[nodiscard]] UInt lobe_flags() const noexcept override
    {
        return Surface::lobe_reflection | Surface::lobe_diffuse;
    }

    void pre_eval() noexcept override;
    void post_eval() noexcept override;

    [[nodiscard]] Surface::Sample sample_impl(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u,
                                              TransportMode mode, ScatterFlags flags) const noexcept override;
    [[nodiscard]] Surface::Evaluation evaluate_impl(Expr<float3> wo, Expr<float3> wi,
                                                    TransportMode mode, ScatterFlags flags) const noexcept override;
};

class DiffuseSurfaceSpec final : public SurfaceSpec
{
private:
    TextureRef _reflectance;
    bool _two_sided;

public:
    DiffuseSurfaceSpec(TextureRef reflectance, bool two_sided) noexcept
        : _reflectance{reflectance}, _two_sided{two_sided} {}

    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override
    {
        visitor.visit(_reflectance);
    }

    [[nodiscard]] const Surface* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
