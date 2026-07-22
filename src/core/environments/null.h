#pragma once

#include "base/environment.h"
#include "scene/scene_builder.h"

namespace Yutrel
{

class NullEnvironment final : public Environment
{
public:
    [[nodiscard]] bool is_black() const noexcept override { return true; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Renderer&, CommandBuffer&) const noexcept override
    {
        return nullptr;
    }
};

class NullEnvironmentSpec final : public EnvironmentSpec
{
public:
    [[nodiscard]] const Environment* build(SceneBuilder& builder) const noexcept override
    {
        return builder.emplace<Environment, NullEnvironment>();
    }
};

} // namespace Yutrel
