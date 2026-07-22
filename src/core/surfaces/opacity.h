#pragma once

#include "base/surface.h"
#include "scene/spec_base.h"

namespace Yutrel
{
class OpacitySurface final : public Surface
{
public:
    class Instance final : public Surface::Instance
    {
    private:
        const Surface::Instance* _base;
        const Texture::Instance* _alpha;

    public:
        Instance(const Renderer& renderer, const OpacitySurface* surface,
                 const Surface::Instance* base, const Texture::Instance* alpha) noexcept
            : Surface::Instance{renderer, surface}, _base{base}, _alpha{alpha} {}

        [[nodiscard]] bool maybe_non_opaque() const noexcept override;
        [[nodiscard]] luisa::optional<Float> evaluate_opacity(
            const Interaction& it, Expr<float> time) const noexcept override;
        [[nodiscard]] luisa::string closure_identifier() const noexcept override;
        [[nodiscard]] luisa::unique_ptr<Closure> create_closure(
            SampledWavelengths& swl, Expr<float> time) const noexcept override;
        void populate_closure(Closure* closure, const Interaction& it,
                              Expr<float3> wo, Expr<float> eta_i) const noexcept override;
    };

private:
    const Surface* _base;
    const Texture* _alpha;

public:
    OpacitySurface(const Surface* base, const Texture* alpha) noexcept
        : Surface{base->two_sided()}, _base{base}, _alpha{alpha} {}

    [[nodiscard]] bool is_null() const noexcept override { return _base->is_null(); }
    [[nodiscard]] bool maybe_non_opaque() const noexcept override;
    [[nodiscard]] uint properties() const noexcept override { return _base->properties(); }
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> build(
        Renderer& renderer, CommandBuffer& command_buffer) const noexcept override;
};

class OpacitySurfaceSpec final : public SurfaceSpec
{
private:
    SurfaceRef _base;
    TextureRef _alpha;

public:
    OpacitySurfaceSpec(SurfaceRef base, TextureRef alpha) noexcept
        : _base{base}, _alpha{alpha} {}

    [[nodiscard]] auto base() const noexcept { return _base; }
    [[nodiscard]] auto alpha() const noexcept { return _alpha; }

    void visit_dependencies(SpecDependencyVisitor& visitor) const noexcept override
    {
        visitor.visit(_base);
        visitor.visit(_alpha);
    }
    [[nodiscard]] const Surface* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
