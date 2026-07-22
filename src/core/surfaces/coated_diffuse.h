#pragma once

#include <cstdint>
#include <utility>

#include <luisa/core/stl/optional.h>

#include "scene/spec_base.h"

namespace Yutrel
{
struct CoatedDiffuseSurfaceParams
{
    luisa::optional<TextureRef> reflectance;
    luisa::optional<TextureRef> roughness;
    luisa::optional<TextureRef> u_roughness;
    luisa::optional<TextureRef> v_roughness;
    luisa::optional<TextureRef> thickness;
    luisa::optional<TextureRef> albedo;
    luisa::optional<TextureRef> g;
    luisa::optional<TextureRef> eta;
    bool remap_roughness{true};
    uint32_t max_depth{10u};
    uint32_t samples{1u};
};

class CoatedDiffuseSurfaceSpec final : public SurfaceSpec
{
private:
    CoatedDiffuseSurfaceParams m_params;

public:
    explicit CoatedDiffuseSurfaceSpec(CoatedDiffuseSurfaceParams params) noexcept
        : m_params{std::move(params)} {}

    [[nodiscard]] const CoatedDiffuseSurfaceParams& params() const noexcept { return m_params; }
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override;
    [[nodiscard]] const Surface* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
