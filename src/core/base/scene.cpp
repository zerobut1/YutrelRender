#include "scene.h"

#include "scene/scene_builder.h"
#include "scene/scene_spec.h"

namespace Yutrel
{
struct Scene::Config
{
    luisa::vector<luisa::unique_ptr<Texture>> textures;
    luisa::vector<luisa::unique_ptr<Surface>> surfaces;
    luisa::vector<luisa::unique_ptr<Medium>> media;
    luisa::vector<luisa::unique_ptr<Light>> lights;
    luisa::vector<luisa::unique_ptr<Environment>> environments;
    luisa::vector<luisa::unique_ptr<Shape>> shapes;
    luisa::vector<luisa::unique_ptr<Spectrum>> spectra;
    luisa::vector<luisa::unique_ptr<Camera>> cameras;
    luisa::vector<luisa::unique_ptr<Film>> films;
    luisa::vector<luisa::unique_ptr<Filter>> filters;
    luisa::vector<luisa::unique_ptr<Sampler>> samplers;
    luisa::vector<luisa::unique_ptr<Integrator>> integrators;
    luisa::vector<ShapeInstance> instances;

    const Spectrum* spectrum{};
    const Environment* environment{};
    const Camera* camera{};
    const Film* film{};
    const Filter* filter{};
    const Sampler* sampler{};
    const Integrator* integrator{};
};

Scene::Scene() noexcept
    : m_config{luisa::make_unique<Config>()}
{
}

Scene::~Scene() noexcept = default;

luisa::unique_ptr<Scene> Scene::create(const SceneSpec& spec) noexcept
{
    auto scene = luisa::make_unique<Scene>();
    SceneBuilder{*scene, spec}.build();
    return scene;
}

const Texture* Scene::_store(luisa::unique_ptr<Texture> object) noexcept { return m_config->textures.emplace_back(std::move(object)).get(); }
const Surface* Scene::_store(luisa::unique_ptr<Surface> object) noexcept { return m_config->surfaces.emplace_back(std::move(object)).get(); }
const Medium* Scene::_store(luisa::unique_ptr<Medium> object) noexcept { return m_config->media.emplace_back(std::move(object)).get(); }
const Light* Scene::_store(luisa::unique_ptr<Light> object) noexcept { return m_config->lights.emplace_back(std::move(object)).get(); }
const Environment* Scene::_store(luisa::unique_ptr<Environment> object) noexcept { return m_config->environments.emplace_back(std::move(object)).get(); }
const Shape* Scene::_store(luisa::unique_ptr<Shape> object) noexcept { return m_config->shapes.emplace_back(std::move(object)).get(); }
const Spectrum* Scene::_store(luisa::unique_ptr<Spectrum> object) noexcept { return m_config->spectra.emplace_back(std::move(object)).get(); }
const Camera* Scene::_store(luisa::unique_ptr<Camera> object) noexcept { return m_config->cameras.emplace_back(std::move(object)).get(); }
const Film* Scene::_store(luisa::unique_ptr<Film> object) noexcept { return m_config->films.emplace_back(std::move(object)).get(); }
const Filter* Scene::_store(luisa::unique_ptr<Filter> object) noexcept { return m_config->filters.emplace_back(std::move(object)).get(); }
const Sampler* Scene::_store(luisa::unique_ptr<Sampler> object) noexcept { return m_config->samplers.emplace_back(std::move(object)).get(); }
const Integrator* Scene::_store(luisa::unique_ptr<Integrator> object) noexcept { return m_config->integrators.emplace_back(std::move(object)).get(); }

void Scene::_set_render_roots(const Spectrum* spectrum, const Environment* environment, const Camera* camera, const Film* film, const Filter* filter, const Sampler* sampler, const Integrator* integrator) noexcept
{
    m_config->spectrum    = spectrum;
    m_config->environment = environment;
    m_config->camera      = camera;
    m_config->film        = film;
    m_config->filter      = filter;
    m_config->sampler     = sampler;
    m_config->integrator  = integrator;
}

void Scene::_add_instance(ShapeInstance instance) noexcept { m_config->instances.emplace_back(instance); }

const Spectrum* Scene::spectrum() const noexcept { return m_config->spectrum; }
const Environment* Scene::environment() const noexcept { return m_config->environment; }
const Camera* Scene::camera() const noexcept { return m_config->camera; }
const Film* Scene::film() const noexcept { return m_config->film; }
const Filter* Scene::filter() const noexcept { return m_config->filter; }
const Sampler* Scene::sampler() const noexcept { return m_config->sampler; }
const Integrator* Scene::integrator() const noexcept { return m_config->integrator; }
luisa::span<const ShapeInstance> Scene::instances() const noexcept { return m_config->instances; }
} // namespace Yutrel
