#pragma once

#include "base/light.h"
#include "scene/scene_builder.h"

namespace Yutrel
{
class NullLight : public Light
{
public:
    NullLight() noexcept = default;

    [[nodiscard]] bool is_null() const noexcept override { return true; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Renderer& renderer, CommandBuffer& command_buffer) const noexcept override
    {
        return nullptr;
    }
};

class NullLightSpec final : public LightSpec
{
public:
    [[nodiscard]] const Light* build(SceneBuilder& builder) const noexcept override
    {
        return builder.emplace<Light, NullLight>();
    }
};

} // namespace Yutrel
