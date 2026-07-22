#pragma once

#include <algorithm>
#include <cmath>

#include "base/filter.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class TriangleFilter final : public Filter
{
public:
    class Instance final : public Filter::Instance
    {
    public:
        using Filter::Instance::Instance;
        [[nodiscard]] Sample sample(Expr<float2> u) const noexcept override;
    };

public:
    explicit TriangleFilter(float radius) noexcept
        : Filter{radius} {}

    [[nodiscard]] float evaluate(float x) const noexcept override
    {
        return std::max(0.0f, radius() - std::abs(x));
    }

    [[nodiscard]] luisa::unique_ptr<Filter::Instance> build(const Renderer& renderer) const noexcept override;
};

class TriangleFilterSpec final : public FilterSpec
{
private:
    float _radius;

public:
    explicit TriangleFilterSpec(float radius) noexcept : _radius{radius} {}
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override { return std::isfinite(_radius) && _radius > 0.0f ? luisa::nullopt : spec_validation_error("Filter radius must be finite and positive."); }
    [[nodiscard]] const Filter* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
