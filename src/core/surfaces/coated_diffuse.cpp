#include "coated_diffuse.h"

#include "scene/scene_builder.h"
#include "surfaces/dielectric.h"
#include "surfaces/diffuse.h"
#include "surfaces/layered.h"
#include "textures/constant.h"

namespace Yutrel
{
luisa::optional<luisa::string> CoatedDiffuseSurfaceSpec::validate() const noexcept
{
    if (m_params.max_depth == 0u)
    {
        return spec_validation_error("CoatedDiffuse max_depth must be positive.");
    }
    if (m_params.samples == 0u)
    {
        return spec_validation_error("CoatedDiffuse samples must be positive.");
    }
    return luisa::nullopt;
}

void CoatedDiffuseSurfaceSpec::visit_dependencies(SpecDependencyVisitor& visitor) const noexcept
{
    auto visit = [&visitor](const luisa::optional<TextureRef>& ref) noexcept
    {
        if (ref) { visitor.visit(*ref); }
    };
    visit(m_params.reflectance);
    visit(m_params.roughness);
    visit(m_params.u_roughness);
    visit(m_params.v_roughness);
    visit(m_params.thickness);
    visit(m_params.albedo);
    visit(m_params.g);
    visit(m_params.eta);
}

const Surface* CoatedDiffuseSurfaceSpec::build(SceneBuilder& builder) const noexcept
{
    auto resolve = [&builder](const luisa::optional<TextureRef>& ref) noexcept -> const Texture*
    {
        return ref ? builder.resolve(*ref) : nullptr;
    };

    auto reflectance = resolve(m_params.reflectance);
    if (reflectance == nullptr)
    {
        reflectance = builder.emplace<Texture, ConstantTexture>(
            make_float4(0.5f, 0.5f, 0.5f, 1.0f));
    }
    auto coat = builder.emplace<Surface, Dielectric>(
        resolve(m_params.roughness), resolve(m_params.u_roughness),
        resolve(m_params.v_roughness), resolve(m_params.eta),
        luisa::nullopt,
        m_params.remap_roughness, false);
    auto substrate = builder.emplace<Surface, Diffuse>(reflectance, false);
    return builder.emplace<Surface, Layered>(
        coat, substrate, resolve(m_params.thickness), resolve(m_params.albedo),
        resolve(m_params.g), m_params.max_depth, m_params.samples, true);
}
} // namespace Yutrel
