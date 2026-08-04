#pragma once

#include <utility>

#include <luisa/core/stl/optional.h>

#include "base/interaction.h"
#include "base/surface.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class Texture;

struct OpenPBRSurfaceParams
{
    luisa::optional<TextureRef> base_weight;
    luisa::optional<TextureRef> base_color;
    luisa::optional<TextureRef> base_metalness;
    luisa::optional<TextureRef> base_diffuse_roughness;
    luisa::optional<TextureRef> specular_weight;
    luisa::optional<TextureRef> specular_color;
    luisa::optional<TextureRef> specular_roughness;
    luisa::optional<TextureRef> specular_roughness_anisotropy;
    luisa::optional<TextureRef> specular_ior;
    bool two_sided{true};
};

class OpenPBRSurface final : public Surface
{
public:
    class Instance;
    class Closure;

private:
    const Texture* m_base_weight;
    const Texture* m_base_color;
    const Texture* m_base_metalness;
    const Texture* m_base_diffuse_roughness;
    const Texture* m_specular_weight;
    const Texture* m_specular_color;
    const Texture* m_specular_roughness;
    const Texture* m_specular_roughness_anisotropy;
    const Texture* m_specular_ior;

public:
    OpenPBRSurface(const Texture* base_weight,
                   const Texture* base_color,
                   const Texture* base_metalness,
                   const Texture* base_diffuse_roughness,
                   const Texture* specular_weight,
                   const Texture* specular_color,
                   const Texture* specular_roughness,
                   const Texture* specular_roughness_anisotropy,
                   const Texture* specular_ior,
                   bool two_sided) noexcept;

    [[nodiscard]] uint properties() const noexcept override { return property_reflective; }
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class OpenPBRSurface::Instance final : public Surface::Instance
{
private:
    const Texture::Instance* m_base_weight;
    const Texture::Instance* m_base_color;
    const Texture::Instance* m_base_metalness;
    const Texture::Instance* m_base_diffuse_roughness;
    const Texture::Instance* m_specular_weight;
    const Texture::Instance* m_specular_color;
    const Texture::Instance* m_specular_roughness;
    const Texture::Instance* m_specular_roughness_anisotropy;
    const Texture::Instance* m_specular_ior;
    uint m_opaque_dielectric_energy_lut;
    uint m_opaque_dielectric_average_lut;
    uint m_ideal_metal_energy_lut;
    uint m_ideal_metal_average_lut;

public:
    Instance(const Renderer& renderer,
             const OpenPBRSurface* surface,
             const Texture::Instance* base_weight,
             const Texture::Instance* base_color,
             const Texture::Instance* base_metalness,
             const Texture::Instance* base_diffuse_roughness,
             const Texture::Instance* specular_weight,
             const Texture::Instance* specular_color,
             const Texture::Instance* specular_roughness,
             const Texture::Instance* specular_roughness_anisotropy,
             const Texture::Instance* specular_ior,
             uint opaque_dielectric_energy_lut,
             uint opaque_dielectric_average_lut,
             uint ideal_metal_energy_lut,
             uint ideal_metal_average_lut) noexcept;

    [[nodiscard]] luisa::string closure_identifier() const noexcept override { return "OpenPBRSurface"; }
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(
        SampledWavelengths& swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure* closure, const Interaction& it,
                          Expr<float3> wo, Expr<float> eta_i) const noexcept override;
};

class OpenPBRSurface::Closure final : public Surface::Closure
{
public:
    struct Context
    {
        Interaction it;
        SampledSpectrum weighted_base_color;
        SampledSpectrum specular_color;
        SampledSpectrum diffuse_albedo;
        SampledSpectrum metal_mms_scale;
        Float metalness;
        Float specular_weight;
        Float diffuse_roughness;
        Float alpha;
        Float2 anisotropic_alpha;
        Float weighted_ior;
        Float dielectric_view_compensation;
        Float metal_view_energy_complement;
        Float specular_lobe_weight;
        Float metal_mms_lobe_weight;
        Float diffuse_lobe_weight;
        Float total_lobe_weight;
    };

private:
    uint m_opaque_dielectric_energy_lut;
    uint m_opaque_dielectric_average_lut;
    uint m_ideal_metal_energy_lut;
    uint m_ideal_metal_average_lut;

public:
    Closure(const Renderer& renderer, const SampledWavelengths& swl,
            Expr<float> time, uint opaque_dielectric_energy_lut,
            uint opaque_dielectric_average_lut,
            uint ideal_metal_energy_lut,
            uint ideal_metal_average_lut) noexcept;

    [[nodiscard]] const Interaction& it() const noexcept override { return context<Context>().it; }
    [[nodiscard]] UInt lobe_flags() const noexcept override;

private:
    [[nodiscard]] Surface::Sample sample_impl(Expr<float3> wo, Expr<float> u_lobe,
                                              Expr<float2> u, TransportMode mode,
                                              ScatterFlags flags) const noexcept override;
    [[nodiscard]] Surface::Evaluation evaluate_impl(Expr<float3> wo, Expr<float3> wi,
                                                    TransportMode mode,
                                                    ScatterFlags flags) const noexcept override;
};

class OpenPBRSurfaceSpec final : public SurfaceSpec
{
private:
    OpenPBRSurfaceParams m_params;

public:
    explicit OpenPBRSurfaceSpec(OpenPBRSurfaceParams params) noexcept
        : m_params{std::move(params)} {}

    [[nodiscard]] const OpenPBRSurfaceParams& params() const noexcept { return m_params; }
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override;
    [[nodiscard]] const Surface* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
