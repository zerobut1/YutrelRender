#pragma once

#include "base/filter.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
class MitchellFilter : public Filter
{
private:
    float m_b{1.0f / 3.0f};
    float m_c{1.0f / 3.0f};

public:
    explicit MitchellFilter(float radius) noexcept
        : Filter{radius} {}

    [[nodiscard]] float evaluate(float x) const noexcept override
    {
        x = 2.f * std::abs(x / radius());
        if (x <= 1.0f)
        {
            return ((12.0f - 9.0f * m_b - 6.0f * m_c) * x * x * x +
                    (-18.0f + 12.0f * m_b + 6.0f * m_c) * x * x +
                    (6.0f - 2.0f * m_b)) *
                   (1.f / 6.f);
        }
        if (x <= 2.0f)
        {
            return ((-m_b - 6.0f * m_c) * x * x * x +
                    (6.0f * m_b + 30.0f * m_c) * x * x +
                    (-12.0f * m_b - 48.0f * m_c) * x +
                    (8.0f * m_b + 24.0f * m_c)) *
                   (1.0f / 6.0f);
        }
        return 0.0f;
    }
};

class MitchellFilterSpec final : public FilterSpec
{
private:
    float _radius;

public:
    explicit MitchellFilterSpec(float radius) noexcept : _radius{radius} {}
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override { return std::isfinite(_radius) && _radius > 0.0f ? luisa::nullopt : spec_validation_error("Filter radius must be finite and positive."); }
    [[nodiscard]] const Filter* build(SceneBuilder& builder) const noexcept override
    {
        return builder.emplace<Filter, MitchellFilter>(_radius);
    }
};
} // namespace Yutrel
