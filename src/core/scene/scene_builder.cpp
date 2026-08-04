#include "scene/scene_builder.h"

#include <luisa/core/logging.h>

#include "base/scene.h"

namespace Yutrel
{

SceneBuilder::SceneBuilder(Scene& scene, const SceneSpec& spec) noexcept
    : _scene{scene},
      _spec{spec},
      _textures{spec.textures().size()},
      _surfaces{spec.surfaces().size()},
      _media{spec.media().size()},
      _lights{spec.lights().size()},
      _environments{spec.environments().size()},
      _shapes{spec.shapes().size()},
      _spectra{spec.spectra().size()},
      _cameras{spec.cameras().size()},
      _films{spec.films().size()},
      _filters{spec.filters().size()},
      _samplers{spec.samplers().size()},
      _integrators{spec.integrators().size()}
{
}

void SceneBuilder::_store(luisa::unique_ptr<Texture> texture) noexcept
{
    (void)_scene._store(std::move(texture));
}

void SceneBuilder::_store(luisa::unique_ptr<Surface> surface) noexcept
{
    (void)_scene._store(std::move(surface));
}

void SceneBuilder::_store(luisa::unique_ptr<Medium> medium) noexcept
{
    (void)_scene._store(std::move(medium));
}

void SceneBuilder::_store(luisa::unique_ptr<Light> light) noexcept
{
    (void)_scene._store(std::move(light));
}

void SceneBuilder::_store(luisa::unique_ptr<Environment> environment) noexcept
{
    (void)_scene._store(std::move(environment));
}

void SceneBuilder::_store(luisa::unique_ptr<Shape> object) noexcept { (void)_scene._store(std::move(object)); }
void SceneBuilder::_store(luisa::unique_ptr<Spectrum> object) noexcept { (void)_scene._store(std::move(object)); }
void SceneBuilder::_store(luisa::unique_ptr<Camera> object) noexcept { (void)_scene._store(std::move(object)); }
void SceneBuilder::_store(luisa::unique_ptr<Film> object) noexcept { (void)_scene._store(std::move(object)); }
void SceneBuilder::_store(luisa::unique_ptr<Filter> object) noexcept { (void)_scene._store(std::move(object)); }
void SceneBuilder::_store(luisa::unique_ptr<Sampler> object) noexcept { (void)_scene._store(std::move(object)); }
void SceneBuilder::_store(luisa::unique_ptr<Integrator> object) noexcept { (void)_scene._store(std::move(object)); }

void SceneBuilder::build() noexcept
{
    auto& render     = _spec.render();
    auto spectrum    = resolve(render.spectrum);
    auto environment = resolve(render.environment);
    auto camera      = resolve(render.camera);
    auto film        = resolve(render.film);
    auto filter      = resolve(render.filter);
    auto sampler     = resolve(render.sampler);
    auto integrator  = resolve(render.integrator);
    _scene._set_render_roots(spectrum, environment, camera, film, filter, sampler, integrator);

    for (auto light : _spec.standalone_lights())
    {
        _scene._add_standalone_light(resolve(light));
    }

    for (auto& instance : _spec.instances())
    {
        _scene._add_instance(ShapeInstance{
            .shape          = resolve(instance.shape),
            .surface        = resolve(instance.surface),
            .light          = instance.light ? resolve(*instance.light) : nullptr,
            .inside_medium  = instance.inside_medium ? resolve(*instance.inside_medium) : nullptr,
            .outside_medium = instance.outside_medium ? resolve(*instance.outside_medium) : nullptr,
            .transform      = instance.transform,
        });
    }
}

const Texture* SceneBuilder::resolve(TextureRef ref) noexcept
{
    return _resolve(ref, _spec.textures(), _textures);
}

const Surface* SceneBuilder::resolve(SurfaceRef ref) noexcept
{
    return _resolve(ref, _spec.surfaces(), _surfaces);
}

const Medium* SceneBuilder::resolve(MediumRef ref) noexcept
{
    return _resolve(ref, _spec.media(), _media);
}

const Light* SceneBuilder::resolve(LightRef ref) noexcept
{
    return _resolve(ref, _spec.lights(), _lights);
}

const Environment* SceneBuilder::resolve(EnvironmentRef ref) noexcept
{
    return _resolve(ref, _spec.environments(), _environments);
}

const Shape* SceneBuilder::resolve(ShapeRef ref) noexcept
{
    return _resolve(ref, _spec.shapes(), _shapes);
}

const Spectrum* SceneBuilder::resolve(SpectrumRef ref) noexcept
{
    return _resolve(ref, _spec.spectra(), _spectra);
}

const Camera* SceneBuilder::resolve(CameraRef ref) noexcept
{
    return _resolve(ref, _spec.cameras(), _cameras);
}

const Film* SceneBuilder::resolve(FilmRef ref) noexcept
{
    return _resolve(ref, _spec.films(), _films);
}

const Filter* SceneBuilder::resolve(FilterRef ref) noexcept
{
    return _resolve(ref, _spec.filters(), _filters);
}

const Sampler* SceneBuilder::resolve(SamplerRef ref) noexcept
{
    return _resolve(ref, _spec.samplers(), _samplers);
}

const Integrator* SceneBuilder::resolve(IntegratorRef ref) noexcept
{
    return _resolve(ref, _spec.integrators(), _integrators);
}

template <typename Spec, typename Runtime>
const Runtime* SceneBuilder::_resolve(SceneRef<Spec> ref, const SpecTable<Spec>& table, BuildCache<Runtime>& cache) noexcept
{
    if (!table.contains(ref))
    {
        LUISA_ERROR("{} spec ref index {} is out of bounds (size {}).", table.category(), ref.index(), table.size());
        return nullptr;
    }

    auto& state  = cache.states[ref.index()];
    auto& object = cache.objects[ref.index()];
    switch (state)
    {
    case BuildState::Built:
        return object;
    case BuildState::Visiting:
        LUISA_ERROR(
            "Runtime spec dependency cycle at {} spec '{}' ({}).",
            table.category(),
            table.meta(ref).name,
            format_source_location(table.meta(ref).source));
        return nullptr;
    case BuildState::Failed:
        LUISA_ERROR(
            "{} spec '{}' has already failed to build ({}).",
            table.category(),
            table.meta(ref).name,
            format_source_location(table.meta(ref).source));
        return nullptr;
    case BuildState::Unvisited:
        break;
    }

    state  = BuildState::Visiting;
    object = table.spec(ref).build(*this);
    if (object == nullptr)
    {
        state = BuildState::Failed;
        LUISA_ERROR(
            "{} spec '{}' returned a null runtime object ({}).",
            table.category(),
            table.meta(ref).name,
            format_source_location(table.meta(ref).source));
        return nullptr;
    }
    state = BuildState::Built;
    return object;
}

} // namespace Yutrel
