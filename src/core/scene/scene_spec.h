#pragma once

#include <utility>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/vector.h>

#include "scene/spec_base.h"
#include "scene/spec_table.h"

namespace Yutrel
{

class SceneSpecBuilder;

struct ShapeInstanceSpec
{
    SourceLocation source;
    ShapeRef shape;
    SurfaceRef surface;
    luisa::optional<LightRef> light;
    luisa::optional<MediumRef> inside_medium;
    luisa::optional<MediumRef> outside_medium;
    luisa::float4x4 transform{luisa::make_float4x4(1.0f)};
};

struct RenderSpec
{
    SpectrumRef spectrum;
    EnvironmentRef environment;
    CameraRef camera;
    FilmRef film;
    FilterRef filter;
    SamplerRef sampler;
    IntegratorRef integrator;
};

class SceneSpec
{
private:
    SpecTable<TextureSpec> _textures;
    SpecTable<SurfaceSpec> _surfaces;
    SpecTable<MediumSpec> _media;
    SpecTable<LightSpec> _lights;
    SpecTable<EnvironmentSpec> _environments;
    SpecTable<ShapeSpec> _shapes;
    SpecTable<SpectrumSpec> _spectra;
    SpecTable<CameraSpec> _cameras;
    SpecTable<FilmSpec> _films;
    SpecTable<FilterSpec> _filters;
    SpecTable<SamplerSpec> _samplers;
    SpecTable<IntegratorSpec> _integrators;
    luisa::vector<LightRef> _standalone_lights;
    luisa::vector<ShapeInstanceSpec> _instances;
    RenderSpec _render;

private:
    SceneSpec(
        SpecTable<TextureSpec> textures,
        SpecTable<SurfaceSpec> surfaces,
        SpecTable<MediumSpec> media,
        SpecTable<LightSpec> lights,
        SpecTable<EnvironmentSpec> environments,
        SpecTable<ShapeSpec> shapes,
        SpecTable<SpectrumSpec> spectra,
        SpecTable<CameraSpec> cameras,
        SpecTable<FilmSpec> films,
        SpecTable<FilterSpec> filters,
        SpecTable<SamplerSpec> samplers,
        SpecTable<IntegratorSpec> integrators,
        luisa::vector<LightRef> standalone_lights,
        luisa::vector<ShapeInstanceSpec> instances,
        RenderSpec render) noexcept
        : _textures{std::move(textures)},
          _surfaces{std::move(surfaces)},
          _media{std::move(media)},
          _lights{std::move(lights)},
          _environments{std::move(environments)},
          _shapes{std::move(shapes)},
          _spectra{std::move(spectra)},
          _cameras{std::move(cameras)},
          _films{std::move(films)},
          _filters{std::move(filters)},
          _samplers{std::move(samplers)},
          _integrators{std::move(integrators)},
          _standalone_lights{std::move(standalone_lights)},
          _instances{std::move(instances)},
          _render{render}
    {
    }

    friend class SceneSpecBuilder;

public:
    SceneSpec()                                = delete;
    SceneSpec(SceneSpec&&) noexcept            = default;
    SceneSpec& operator=(SceneSpec&&) noexcept = default;
    SceneSpec(const SceneSpec&)                = delete;
    SceneSpec& operator=(const SceneSpec&)     = delete;
    ~SceneSpec() noexcept                      = default;

    [[nodiscard]] const SpecTable<TextureSpec>& textures() const noexcept { return _textures; }
    [[nodiscard]] const SpecTable<SurfaceSpec>& surfaces() const noexcept { return _surfaces; }
    [[nodiscard]] const SpecTable<MediumSpec>& media() const noexcept { return _media; }
    [[nodiscard]] const SpecTable<LightSpec>& lights() const noexcept { return _lights; }
    [[nodiscard]] const SpecTable<EnvironmentSpec>& environments() const noexcept { return _environments; }
    [[nodiscard]] const SpecTable<ShapeSpec>& shapes() const noexcept { return _shapes; }
    [[nodiscard]] const SpecTable<SpectrumSpec>& spectra() const noexcept { return _spectra; }
    [[nodiscard]] const SpecTable<CameraSpec>& cameras() const noexcept { return _cameras; }
    [[nodiscard]] const SpecTable<FilmSpec>& films() const noexcept { return _films; }
    [[nodiscard]] const SpecTable<FilterSpec>& filters() const noexcept { return _filters; }
    [[nodiscard]] const SpecTable<SamplerSpec>& samplers() const noexcept { return _samplers; }
    [[nodiscard]] const SpecTable<IntegratorSpec>& integrators() const noexcept { return _integrators; }
    [[nodiscard]] luisa::span<const LightRef> standalone_lights() const noexcept { return _standalone_lights; }
    [[nodiscard]] luisa::span<const ShapeInstanceSpec> instances() const noexcept { return _instances; }
    [[nodiscard]] const RenderSpec& render() const noexcept { return _render; }
};

} // namespace Yutrel
