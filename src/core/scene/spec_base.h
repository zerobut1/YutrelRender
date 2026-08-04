#pragma once

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

#include "scene/scene_ref.h"

namespace Yutrel
{

[[nodiscard]] inline luisa::optional<luisa::string> spec_validation_error(luisa::string_view message) noexcept
{
    return luisa::optional<luisa::string>{luisa::string{message}};
}

class SceneBuilder;

class Texture;
class Surface;
class Medium;
class Light;
class Environment;
class Shape;
class Spectrum;
class Camera;
class Film;
class Filter;
class Sampler;
class Integrator;

class SpecDependencyVisitor
{
public:
    virtual ~SpecDependencyVisitor() noexcept = default;

    virtual void visit(TextureRef ref) noexcept     = 0;
    virtual void visit(SurfaceRef ref) noexcept     = 0;
    virtual void visit(MediumRef ref) noexcept      = 0;
    virtual void visit(LightRef ref) noexcept       = 0;
    virtual void visit(EnvironmentRef ref) noexcept = 0;
    virtual void visit(ShapeRef ref) noexcept       = 0;
    virtual void visit(SpectrumRef ref) noexcept    = 0;
    virtual void visit(CameraRef ref) noexcept      = 0;
    virtual void visit(FilmRef ref) noexcept        = 0;
    virtual void visit(FilterRef ref) noexcept      = 0;
    virtual void visit(SamplerRef ref) noexcept     = 0;
    virtual void visit(IntegratorRef ref) noexcept  = 0;
};

class TextureSpec
{
public:
    virtual ~TextureSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Texture* build(SceneBuilder& builder) const noexcept = 0;
};

class SurfaceSpec
{
public:
    virtual ~SurfaceSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Surface* build(SceneBuilder& builder) const noexcept = 0;
};

class MediumSpec
{
public:
    virtual ~MediumSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Medium* build(SceneBuilder& builder) const noexcept = 0;
};

class LightSpec
{
public:
    virtual ~LightSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Light* build(SceneBuilder& builder) const noexcept = 0;
};

class EnvironmentSpec
{
public:
    virtual ~EnvironmentSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Environment* build(SceneBuilder& builder) const noexcept = 0;
};

class ShapeSpec
{
public:
    virtual ~ShapeSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Shape* build(SceneBuilder& builder) const noexcept = 0;
};

class SpectrumSpec
{
public:
    virtual ~SpectrumSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Spectrum* build(SceneBuilder& builder) const noexcept = 0;
};

class CameraSpec
{
public:
    virtual ~CameraSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Camera* build(SceneBuilder& builder) const noexcept = 0;
};

class FilmSpec
{
public:
    virtual ~FilmSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Film* build(SceneBuilder& builder) const noexcept = 0;
};

class FilterSpec
{
public:
    virtual ~FilterSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Filter* build(SceneBuilder& builder) const noexcept = 0;
};

class SamplerSpec
{
public:
    virtual ~SamplerSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Sampler* build(SceneBuilder& builder) const noexcept = 0;
};

class IntegratorSpec
{
public:
    virtual ~IntegratorSpec() noexcept = default;
    [[nodiscard]] virtual luisa::optional<luisa::string> validate() const noexcept { return luisa::nullopt; }
    virtual void visit_dependencies(SpecDependencyVisitor&) const noexcept {}
    [[nodiscard]] virtual const Integrator* build(SceneBuilder& builder) const noexcept = 0;
};

} // namespace Yutrel
