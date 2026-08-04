#pragma once

#include <utility>

#include "scene/scene_spec.h"

namespace Yutrel
{

class SceneSpecBuilder
{
private:
    SpecTable<TextureSpec> _textures{"texture"};
    SpecTable<SurfaceSpec> _surfaces{"surface"};
    SpecTable<MediumSpec> _media{"medium"};
    SpecTable<LightSpec> _lights{"light"};
    SpecTable<EnvironmentSpec> _environments{"environment"};
    SpecTable<ShapeSpec> _shapes{"shape"};
    SpecTable<SpectrumSpec> _spectra{"spectrum"};
    SpecTable<CameraSpec> _cameras{"camera"};
    SpecTable<FilmSpec> _films{"film"};
    SpecTable<FilterSpec> _filters{"filter"};
    SpecTable<SamplerSpec> _samplers{"sampler"};
    SpecTable<IntegratorSpec> _integrators{"integrator"};
    luisa::vector<LightRef> _standalone_lights;
    luisa::vector<ShapeInstanceSpec> _instances;
    luisa::optional<RenderSpec> _render;
    bool _finished{};

public:
    SceneSpecBuilder() noexcept                          = default;
    SceneSpecBuilder(SceneSpecBuilder&&)                 = delete;
    SceneSpecBuilder& operator=(SceneSpecBuilder&&)      = delete;
    SceneSpecBuilder(const SceneSpecBuilder&)            = delete;
    SceneSpecBuilder& operator=(const SceneSpecBuilder&) = delete;

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, TextureSpec>
    [[nodiscard]] TextureRef add_texture(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _textures.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, TextureSpec>
    [[nodiscard]] TextureRef add_anonymous_texture(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _textures.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] TextureRef reference_texture(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, SurfaceSpec>
    [[nodiscard]] SurfaceRef add_surface(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _surfaces.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, SurfaceSpec>
    [[nodiscard]] SurfaceRef add_anonymous_surface(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _surfaces.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] SurfaceRef reference_surface(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, MediumSpec>
    [[nodiscard]] MediumRef add_medium(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _media.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    [[nodiscard]] MediumRef reference_medium(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, LightSpec>
    [[nodiscard]] LightRef add_light(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _lights.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, LightSpec>
    [[nodiscard]] LightRef add_anonymous_light(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _lights.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] LightRef reference_light(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, EnvironmentSpec>
    [[nodiscard]] EnvironmentRef add_environment(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _environments.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, EnvironmentSpec>
    [[nodiscard]] EnvironmentRef add_anonymous_environment(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _environments.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] EnvironmentRef reference_environment(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, ShapeSpec>
    [[nodiscard]] ShapeRef add_shape(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _shapes.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, ShapeSpec>
    [[nodiscard]] ShapeRef add_anonymous_shape(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _shapes.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] ShapeRef reference_shape(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, SpectrumSpec>
    [[nodiscard]] SpectrumRef add_spectrum(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _spectra.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, SpectrumSpec>
    [[nodiscard]] SpectrumRef add_anonymous_spectrum(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _spectra.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] SpectrumRef reference_spectrum(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, CameraSpec>
    [[nodiscard]] CameraRef add_camera(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _cameras.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, CameraSpec>
    [[nodiscard]] CameraRef add_anonymous_camera(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _cameras.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] CameraRef reference_camera(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, FilmSpec>
    [[nodiscard]] FilmRef add_film(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _films.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, FilmSpec>
    [[nodiscard]] FilmRef add_anonymous_film(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _films.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] FilmRef reference_film(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, FilterSpec>
    [[nodiscard]] FilterRef add_filter(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _filters.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, FilterSpec>
    [[nodiscard]] FilterRef add_anonymous_filter(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _filters.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] FilterRef reference_filter(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, SamplerSpec>
    [[nodiscard]] SamplerRef add_sampler(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _samplers.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, SamplerSpec>
    [[nodiscard]] SamplerRef add_anonymous_sampler(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _samplers.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] SamplerRef reference_sampler(luisa::string name, SourceLocation use_site);

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, IntegratorSpec>
    [[nodiscard]] IntegratorRef add_integrator(SpecMeta meta, Args&&... args)
    {
        _ensure_mutable();
        return _integrators.add<Impl>(std::move(meta), std::forward<Args>(args)...);
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, IntegratorSpec>
    [[nodiscard]] IntegratorRef add_anonymous_integrator(SourceLocation source, Args&&... args)
    {
        _ensure_mutable();
        return _integrators.add_anonymous<Impl>(std::move(source), std::forward<Args>(args)...);
    }

    [[nodiscard]] IntegratorRef reference_integrator(luisa::string name, SourceLocation use_site);

    void add_standalone_light(LightRef light);
    void add_instance(ShapeInstanceSpec instance);
    void set_render(RenderSpec render);

    [[nodiscard]] SceneSpec finish();

private:
    void _ensure_mutable() const;
    void _validate() const;
};

} // namespace Yutrel
