#pragma once

#include "base/filter.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
class LanczosSincFilter final : public Filter
{
private:
    float m_tau{3.0f};

public:
    explicit LanczosSincFilter(float radius) noexcept
        : Filter{radius} {}

    [[nodiscard]] float evaluate(float x) const noexcept override
    {
        x = x / radius();

        static constexpr auto sin_x_over_x = [](auto x) noexcept
        {
            return 1.0f + x * x == 1.0f ? 1.0f : std::sin(x) / x;
        };
        static constexpr auto sinc = [](auto x) noexcept
        {
            return sin_x_over_x(pi * x);
        };
        if (std::abs(x) > 1.0f) [[unlikely]]
        {
            return 0.0f;
        }
        return sinc(x) * sinc(x / m_tau);
    }
};

class LanczosSincFilterSpec final : public FilterSpec
{
private:
    float _radius;

public:
    explicit LanczosSincFilterSpec(float radius) noexcept : _radius{radius} {}
    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override { return std::isfinite(_radius) && _radius > 0.0f ? luisa::nullopt : spec_validation_error("Filter radius must be finite and positive."); }
    [[nodiscard]] const Filter* build(SceneBuilder& builder) const noexcept override
    {
        return builder.emplace<Filter, LanczosSincFilter>(_radius);
    }
};

} // namespace Yutrel
