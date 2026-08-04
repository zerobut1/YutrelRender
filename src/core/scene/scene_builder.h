#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

#include "scene/scene_spec.h"

namespace Yutrel
{

class Scene;

class SceneBuilder
{
private:
    enum class BuildState : uint8_t
    {
        Unvisited,
        Visiting,
        Built,
        Failed,
    };

    template <typename Runtime>
    struct BuildCache
    {
        luisa::vector<BuildState> states;
        luisa::vector<const Runtime*> objects;

        explicit BuildCache(size_t size)
            : states(size, BuildState::Unvisited),
              objects(size, nullptr)
        {
        }
    };

private:
    Scene& _scene;
    const SceneSpec& _spec;
    BuildCache<Texture> _textures;
    BuildCache<Surface> _surfaces;
    BuildCache<Medium> _media;
    BuildCache<Light> _lights;
    BuildCache<Environment> _environments;
    BuildCache<Shape> _shapes;
    BuildCache<Spectrum> _spectra;
    BuildCache<Camera> _cameras;
    BuildCache<Film> _films;
    BuildCache<Filter> _filters;
    BuildCache<Sampler> _samplers;
    BuildCache<Integrator> _integrators;

public:
    SceneBuilder(Scene& scene, const SceneSpec& spec) noexcept;

    SceneBuilder()                               = delete;
    SceneBuilder(SceneBuilder&&)                 = delete;
    SceneBuilder& operator=(SceneBuilder&&)      = delete;
    SceneBuilder(const SceneBuilder&)            = delete;
    SceneBuilder& operator=(const SceneBuilder&) = delete;

    [[nodiscard]] const Texture* resolve(TextureRef ref) noexcept;
    [[nodiscard]] const Surface* resolve(SurfaceRef ref) noexcept;
    [[nodiscard]] const Medium* resolve(MediumRef ref) noexcept;
    [[nodiscard]] const Light* resolve(LightRef ref) noexcept;
    [[nodiscard]] const Environment* resolve(EnvironmentRef ref) noexcept;
    [[nodiscard]] const Shape* resolve(ShapeRef ref) noexcept;
    [[nodiscard]] const Spectrum* resolve(SpectrumRef ref) noexcept;
    [[nodiscard]] const Camera* resolve(CameraRef ref) noexcept;
    [[nodiscard]] const Film* resolve(FilmRef ref) noexcept;
    [[nodiscard]] const Filter* resolve(FilterRef ref) noexcept;
    [[nodiscard]] const Sampler* resolve(SamplerRef ref) noexcept;
    [[nodiscard]] const Integrator* resolve(IntegratorRef ref) noexcept;

    void build() noexcept;

    template <typename Base, typename Impl, typename... Args>
        requires std::derived_from<Impl, Base>
    [[nodiscard]] const Base* emplace(Args&&... args) noexcept
    {
        auto object = luisa::make_unique<Impl>(std::forward<Args>(args)...);
        auto ptr    = static_cast<const Base*>(object.get());
        _store(luisa::unique_ptr<Base>{std::move(object)});
        return ptr;
    }

private:
    void _store(luisa::unique_ptr<Texture> texture) noexcept;
    void _store(luisa::unique_ptr<Surface> surface) noexcept;
    void _store(luisa::unique_ptr<Medium> medium) noexcept;
    void _store(luisa::unique_ptr<Light> light) noexcept;
    void _store(luisa::unique_ptr<Environment> environment) noexcept;
    void _store(luisa::unique_ptr<Shape> shape) noexcept;
    void _store(luisa::unique_ptr<Spectrum> spectrum) noexcept;
    void _store(luisa::unique_ptr<Camera> camera) noexcept;
    void _store(luisa::unique_ptr<Film> film) noexcept;
    void _store(luisa::unique_ptr<Filter> filter) noexcept;
    void _store(luisa::unique_ptr<Sampler> sampler) noexcept;
    void _store(luisa::unique_ptr<Integrator> integrator) noexcept;

    template <typename Spec, typename Runtime>
    [[nodiscard]] const Runtime* _resolve(SceneRef<Spec> ref, const SpecTable<Spec>& table, BuildCache<Runtime>& cache) noexcept;
};

} // namespace Yutrel
