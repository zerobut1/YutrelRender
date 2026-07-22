#include "triangle.h"

#include "scene/scene_builder.h"

namespace Yutrel
{
const Filter* TriangleFilterSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Filter, TriangleFilter>(_radius);
}

luisa::unique_ptr<Filter::Instance> TriangleFilter::build(const Renderer& renderer) const noexcept
{
    return luisa::make_unique<TriangleFilter::Instance>(renderer, this);
}

Filter::Sample TriangleFilter::Instance::sample(Expr<float2> u) const noexcept
{
    // Analytic inverse CDF, equivalent to PBRT-v4's SampleTent.
    auto a      = 2.0f * u;
    auto offset = ite(u < 0.5f, sqrt(a) - 1.0f, 1.0f - sqrt(2.0f - a));
    return {offset * base<TriangleFilter>()->radius(), 1.0f};
}
} // namespace Yutrel
