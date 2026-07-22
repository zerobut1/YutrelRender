#include "homogeneous.h"

#include <cmath>

#include "base/interaction.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"

namespace Yutrel
{

HomogeneousMedium::HomogeneousMedium(const Texture* sigma_a, const Texture* sigma_s, float scale, float g) noexcept
    : _sigma_a{sigma_a}, _sigma_s{sigma_s}, _scale{scale}, _g{g}
{
}

HomogeneousMedium::Instance::Instance(const Renderer& renderer, const HomogeneousMedium* medium,
                                      const Texture::Instance* sigma_a, const Texture::Instance* sigma_s) noexcept
    : Medium::Instance{renderer, medium}, _sigma_a{sigma_a}, _sigma_s{sigma_s}
{
}

Medium::Properties HomogeneousMedium::Instance::properties(const Interaction& it, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    auto scale           = base<HomogeneousMedium>()->scale();
    auto sigma_a_decoded = _sigma_a->evaluate_unbounded_spectrum(it, swl, time);
    auto sigma_s_decoded = _sigma_s->evaluate_unbounded_spectrum(it, swl, time);
    auto sigma_a         = sigma_a_decoded.value * sigma_a_decoded.strength * scale;
    auto sigma_s         = sigma_s_decoded.value * sigma_s_decoded.strength * scale;
    auto sigma_t         = sigma_a + sigma_s;
    return {
        .sigma_a   = sigma_a,
        .sigma_s   = sigma_s,
        .sigma_t   = sigma_t,
        .sigma_maj = sigma_t,
        .g         = base<HomogeneousMedium>()->g(),
    };
}

luisa::unique_ptr<Medium::Instance> HomogeneousMedium::build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    return luisa::make_unique<Instance>(
        renderer,
        this,
        renderer.build_texture(command_buffer, _sigma_a),
        renderer.build_texture(command_buffer, _sigma_s));
}

luisa::optional<luisa::string> HomogeneousMediumSpec::validate() const noexcept
{
    if (!std::isfinite(_params.scale) || _params.scale < 0.0f)
    {
        return spec_validation_error("Homogeneous medium scale must be finite and non-negative.");
    }
    if (!std::isfinite(_params.g) || std::abs(_params.g) >= 1.0f)
    {
        return spec_validation_error("Homogeneous medium g must be finite and satisfy abs(g) < 1.");
    }
    return luisa::nullopt;
}

void HomogeneousMediumSpec::visit_dependencies(SpecDependencyVisitor& visitor) const noexcept
{
    visitor.visit(_params.sigma_a);
    visitor.visit(_params.sigma_s);
}

const Medium* HomogeneousMediumSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Medium, HomogeneousMedium>(
        builder.resolve(_params.sigma_a),
        builder.resolve(_params.sigma_s),
        _params.scale,
        _params.g);
}

} // namespace Yutrel
