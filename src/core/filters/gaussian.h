#pragma once

#include <algorithm>

#include "base/filter.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
class GaussianFilter : public Filter
{
private:
    float m_sigma;

public:
    explicit GaussianFilter(float radius, float sigma = 0.5f) noexcept
        : Filter{radius}, m_sigma{sigma} {}

    [[nodiscard]] float evaluate(float x) const noexcept override
    {
        auto gaussian = [s = 2.0f * m_sigma * m_sigma](float value) noexcept
        {
            return std::exp(-value * value / s) / std::sqrt(pi * s);
        };
        return std::max(0.0f, gaussian(x) - gaussian(radius()));
    }
};

class GaussianFilterSpec final : public FilterSpec
{
private:
    float _radius;
    float _sigma;

public:
    explicit GaussianFilterSpec(float radius, float sigma = 0.5f) noexcept
        : _radius{radius}, _sigma{sigma} {}
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        if (!std::isfinite(_radius) || _radius <= 0.0f)
        {
            return spec_validation_error("Filter radius must be finite and positive.");
        }
        if (!std::isfinite(_sigma) || _sigma <= 0.0f)
        {
            return spec_validation_error("Gaussian filter sigma must be finite and positive.");
        }
        return luisa::nullopt;
    }
    [[nodiscard]] const Filter* build(SceneBuilder& builder) const noexcept override
    {
        return builder.emplace<Filter, GaussianFilter>(_radius, _sigma);
    }
};
} // namespace Yutrel
