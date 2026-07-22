#pragma once

#include <utility>

#include <luisa/core/stl/optional.h>

#include "base/interaction.h"
#include "base/surface.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class Texture;

struct DielectricSurfaceParams
{
    luisa::optional<TextureRef> roughness;
    luisa::optional<TextureRef> u_roughness;
    luisa::optional<TextureRef> v_roughness;
    luisa::optional<TextureRef> eta;
    bool remap_roughness{true};
    bool two_sided{false};
};

class Dielectric final : public Surface
{
public:
    class Instance;
    class Closure;

private:
    const Texture* m_roughness;
    const Texture* m_u_roughness;
    const Texture* m_v_roughness;
    const Texture* m_eta;
    bool m_remap_roughness;

public:
    Dielectric(const Texture* roughness, const Texture* u_roughness,
               const Texture* v_roughness, const Texture* eta,
               bool remap_roughness, bool two_sided) noexcept;

    [[nodiscard]] bool remap_roughness() const noexcept { return m_remap_roughness; }
    [[nodiscard]] uint properties() const noexcept override
    {
        return property_reflective | property_transmissive;
    }
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class Dielectric::Instance final : public Surface::Instance
{
private:
    const Texture::Instance* m_roughness;
    const Texture::Instance* m_u_roughness;
    const Texture::Instance* m_v_roughness;
    const Texture::Instance* m_eta;

public:
    Instance(const Renderer& renderer, const Dielectric* surface,
             const Texture::Instance* roughness,
             const Texture::Instance* u_roughness,
             const Texture::Instance* v_roughness,
             const Texture::Instance* eta) noexcept;

    [[nodiscard]] luisa::string closure_identifier() const noexcept override { return "Dielectric"; }
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(
        SampledWavelengths& swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure* closure, const Interaction& it,
                          Expr<float3> wo, Expr<float> eta_i) const noexcept override;
};

class Dielectric::Closure final : public Surface::Closure
{
public:
    struct Context
    {
        Interaction it;
        Float2 alpha;
        Float eta_i;
        Float eta_t;
    };

private:
    class Impl;
    luisa::unique_ptr<Impl> m_impl;

public:
    Closure(const Renderer& renderer, const SampledWavelengths& swl, Expr<float> time) noexcept;
    ~Closure() noexcept override;

    [[nodiscard]] const Interaction& it() const noexcept override { return context<Context>().it; }
    [[nodiscard]] UInt lobe_flags() const noexcept override;
    [[nodiscard]] luisa::optional<Float> eta() const noexcept override { return context<Context>().eta_t; }
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

class DielectricSurfaceSpec final : public SurfaceSpec
{
private:
    DielectricSurfaceParams m_params;

public:
    explicit DielectricSurfaceSpec(DielectricSurfaceParams params) noexcept
        : m_params{std::move(params)} {}

    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override;
    [[nodiscard]] const Surface* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel

