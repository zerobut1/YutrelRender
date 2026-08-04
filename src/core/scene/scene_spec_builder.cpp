#include "scene/scene_spec_builder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <luisa/core/logging.h>

namespace Yutrel
{
namespace
{

enum class SpecCategory : uint8_t
{
    Texture,
    Surface,
    Medium,
    Light,
    Environment,
    Shape,
    Spectrum,
    Camera,
    Film,
    Filter,
    Sampler,
    Integrator,
    Count,
};

struct SpecNode
{
    SpecCategory category;
    uint64_t table_id;
    uint32_t index;
};

struct SpecNodeData
{
    SpecCategory category;
    SpecMeta meta;
    luisa::vector<SpecNode> dependencies;
};

class DependencyCollector final : public SpecDependencyVisitor
{
private:
    luisa::vector<SpecNode>& _dependencies;

public:
    explicit DependencyCollector(luisa::vector<SpecNode>& dependencies) noexcept
        : _dependencies{dependencies}
    {
    }

    void visit(TextureRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Texture, ref.table_id(), ref.index()}); }
    void visit(SurfaceRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Surface, ref.table_id(), ref.index()}); }
    void visit(MediumRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Medium, ref.table_id(), ref.index()}); }
    void visit(LightRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Light, ref.table_id(), ref.index()}); }
    void visit(EnvironmentRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Environment, ref.table_id(), ref.index()}); }
    void visit(ShapeRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Shape, ref.table_id(), ref.index()}); }
    void visit(SpectrumRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Spectrum, ref.table_id(), ref.index()}); }
    void visit(CameraRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Camera, ref.table_id(), ref.index()}); }
    void visit(FilmRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Film, ref.table_id(), ref.index()}); }
    void visit(FilterRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Filter, ref.table_id(), ref.index()}); }
    void visit(SamplerRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Sampler, ref.table_id(), ref.index()}); }
    void visit(IntegratorRef ref) noexcept override { _dependencies.emplace_back(SpecNode{SpecCategory::Integrator, ref.table_id(), ref.index()}); }
};

[[nodiscard]] constexpr size_t category_index(SpecCategory category) noexcept
{
    return static_cast<size_t>(category);
}

[[nodiscard]] constexpr luisa::string_view category_name(SpecCategory category) noexcept
{
    switch (category)
    {
    case SpecCategory::Texture:
        return "texture";
    case SpecCategory::Surface:
        return "surface";
    case SpecCategory::Medium:
        return "medium";
    case SpecCategory::Light:
        return "light";
    case SpecCategory::Environment:
        return "environment";
    case SpecCategory::Shape:
        return "shape";
    case SpecCategory::Spectrum:
        return "spectrum";
    case SpecCategory::Camera:
        return "camera";
    case SpecCategory::Film:
        return "film";
    case SpecCategory::Filter:
        return "filter";
    case SpecCategory::Sampler:
        return "sampler";
    case SpecCategory::Integrator:
        return "integrator";
    case SpecCategory::Count:
        break;
    }
    return "unknown";
}

[[noreturn]] void throw_validation_error(const SourceLocation& source, luisa::string message)
{
    auto formatted = luisa::format("{}: {}", format_source_location(source), message);
    throw std::runtime_error{formatted.c_str()};
}

} // namespace

TextureRef SceneSpecBuilder::reference_texture(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _textures.reference(std::move(name), std::move(use_site));
}

SurfaceRef SceneSpecBuilder::reference_surface(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _surfaces.reference(std::move(name), std::move(use_site));
}

MediumRef SceneSpecBuilder::reference_medium(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _media.reference(std::move(name), std::move(use_site));
}

LightRef SceneSpecBuilder::reference_light(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _lights.reference(std::move(name), std::move(use_site));
}

EnvironmentRef SceneSpecBuilder::reference_environment(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _environments.reference(std::move(name), std::move(use_site));
}

ShapeRef SceneSpecBuilder::reference_shape(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _shapes.reference(std::move(name), std::move(use_site));
}

SpectrumRef SceneSpecBuilder::reference_spectrum(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _spectra.reference(std::move(name), std::move(use_site));
}

CameraRef SceneSpecBuilder::reference_camera(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _cameras.reference(std::move(name), std::move(use_site));
}

FilmRef SceneSpecBuilder::reference_film(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _films.reference(std::move(name), std::move(use_site));
}

FilterRef SceneSpecBuilder::reference_filter(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _filters.reference(std::move(name), std::move(use_site));
}

SamplerRef SceneSpecBuilder::reference_sampler(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _samplers.reference(std::move(name), std::move(use_site));
}

IntegratorRef SceneSpecBuilder::reference_integrator(luisa::string name, SourceLocation use_site)
{
    _ensure_mutable();
    return _integrators.reference(std::move(name), std::move(use_site));
}

void SceneSpecBuilder::add_instance(ShapeInstanceSpec instance)
{
    _ensure_mutable();
    _instances.emplace_back(std::move(instance));
}

void SceneSpecBuilder::add_standalone_light(LightRef light)
{
    _ensure_mutable();
    _standalone_lights.emplace_back(light);
}

void SceneSpecBuilder::set_render(RenderSpec render)
{
    _ensure_mutable();
    if (_render)
    {
        throw std::runtime_error{"SceneSpec render root has already been set."};
    }
    _render.emplace(render);
}

SceneSpec SceneSpecBuilder::finish()
{
    _ensure_mutable();
    _validate();
    _finished = true;
    return SceneSpec{
        std::move(_textures),
        std::move(_surfaces),
        std::move(_media),
        std::move(_lights),
        std::move(_environments),
        std::move(_shapes),
        std::move(_spectra),
        std::move(_cameras),
        std::move(_films),
        std::move(_filters),
        std::move(_samplers),
        std::move(_integrators),
        std::move(_standalone_lights),
        std::move(_instances),
        *_render,
    };
}

void SceneSpecBuilder::_ensure_mutable() const
{
    if (_finished)
    {
        throw std::runtime_error{"SceneSpecBuilder has already been finished."};
    }
}

void SceneSpecBuilder::_validate() const
{
    _textures.validate_definitions();
    _surfaces.validate_definitions();
    _media.validate_definitions();
    _lights.validate_definitions();
    _environments.validate_definitions();
    _shapes.validate_definitions();
    _spectra.validate_definitions();
    _cameras.validate_definitions();
    _films.validate_definitions();
    _filters.validate_definitions();
    _samplers.validate_definitions();
    _integrators.validate_definitions();

    if (!_render)
    {
        throw std::runtime_error{"SceneSpec render root has not been set."};
    }
    // Empty scenes (no shape instances) are allowed: they render fully black.
    // The Unity external render path uses this to represent scenes without
    // renderable geometry.
    if (_instances.empty())
    {
        LUISA_WARNING_WITH_LOCATION("SceneSpec contains no shape instances; the scene will render black.");
    }
    auto validate_ref = [](const auto& table, auto ref, const SourceLocation& source)
    {
        if (!table.contains(ref))
        {
            if (ref.table_id() != table.table_id())
            {
                throw_validation_error(source, luisa::format("{} spec ref belongs to a different scene spec.", table.category()));
            }
            throw_validation_error(source, luisa::format("{} spec ref index {} is out of bounds (size {}).", table.category(), ref.index(), table.size()));
        }
    };
    SourceLocation root_source{};
    validate_ref(_spectra, _render->spectrum, root_source);
    validate_ref(_environments, _render->environment, root_source);
    validate_ref(_cameras, _render->camera, root_source);
    validate_ref(_films, _render->film, root_source);
    validate_ref(_filters, _render->filter, root_source);
    validate_ref(_samplers, _render->sampler, root_source);
    validate_ref(_integrators, _render->integrator, root_source);
    for (auto light : _standalone_lights)
    {
        validate_ref(_lights, light, root_source);
    }
    for (auto& instance : _instances)
    {
        validate_ref(_shapes, instance.shape, instance.source);
        validate_ref(_surfaces, instance.surface, instance.source);
        if (instance.light)
        {
            validate_ref(_lights, *instance.light, instance.source);
        }
        if (instance.inside_medium)
        {
            validate_ref(_media, *instance.inside_medium, instance.source);
        }
        if (instance.outside_medium)
        {
            validate_ref(_media, *instance.outside_medium, instance.source);
        }
        auto& m = instance.transform;
        for (auto column = 0u; column < 4u; column++)
        {
            for (auto row = 0u; row < 4u; row++)
            {
                if (!std::isfinite(m[column][row]))
                {
                    throw_validation_error(instance.source, "Shape instance transform contains a non-finite value.");
                }
            }
        }
        if (std::abs(m[0].w) > 1e-6f || std::abs(m[1].w) > 1e-6f || std::abs(m[2].w) > 1e-6f || std::abs(m[3].w - 1.0f) > 1e-6f)
        {
            throw_validation_error(instance.source, "Shape instance transform must be affine.");
        }
        auto determinant = m[0].x * (m[1].y * m[2].z - m[1].z * m[2].y) -
                           m[1].x * (m[0].y * m[2].z - m[0].z * m[2].y) +
                           m[2].x * (m[0].y * m[1].z - m[0].z * m[1].y);
        if (std::abs(determinant) < 1e-8f)
        {
            throw_validation_error(instance.source, "Shape instance transform is singular.");
        }
    }

    constexpr auto category_count = category_index(SpecCategory::Count);
    std::array<size_t, category_count> counts{
        _textures.size(),
        _surfaces.size(),
        _media.size(),
        _lights.size(),
        _environments.size(),
        _shapes.size(),
        _spectra.size(),
        _cameras.size(),
        _films.size(),
        _filters.size(),
        _samplers.size(),
        _integrators.size(),
    };
    std::array<size_t, category_count> offsets{};
    for (size_t i = 1u; i < category_count; i++)
    {
        offsets[i] = offsets[i - 1u] + counts[i - 1u];
    }
    auto total_count = offsets.back() + counts.back();

    luisa::vector<SpecNodeData> nodes;
    nodes.reserve(total_count);
    auto append_table = [&]<typename Spec>(SpecCategory category, const SpecTable<Spec>& table)
    {
        table.visit_entries(
            [&](SceneRef<Spec>, const SpecMeta& meta, const Spec* spec)
        {
            if (auto error = spec->validate())
            {
                throw_validation_error(meta.source, *error);
            }
            SpecNodeData node{.category = category, .meta = meta};
            DependencyCollector collector{node.dependencies};
            spec->visit_dependencies(collector);
            nodes.emplace_back(std::move(node));
        });
        if (nodes.size() != offsets[category_index(category)] + counts[category_index(category)])
        {
            throw std::runtime_error{"Internal SceneSpec category layout mismatch."};
        }
    };

    append_table(SpecCategory::Texture, _textures);
    append_table(SpecCategory::Surface, _surfaces);
    append_table(SpecCategory::Medium, _media);
    append_table(SpecCategory::Light, _lights);
    append_table(SpecCategory::Environment, _environments);
    append_table(SpecCategory::Shape, _shapes);
    append_table(SpecCategory::Spectrum, _spectra);
    append_table(SpecCategory::Camera, _cameras);
    append_table(SpecCategory::Film, _films);
    append_table(SpecCategory::Filter, _filters);
    append_table(SpecCategory::Sampler, _samplers);
    append_table(SpecCategory::Integrator, _integrators);

    auto dependency_exists = [&](SpecNode node) noexcept
    {
        switch (node.category)
        {
        case SpecCategory::Texture:
            return _textures.contains(node.table_id, node.index);
        case SpecCategory::Surface:
            return _surfaces.contains(node.table_id, node.index);
        case SpecCategory::Medium:
            return _media.contains(node.table_id, node.index);
        case SpecCategory::Light:
            return _lights.contains(node.table_id, node.index);
        case SpecCategory::Environment:
            return _environments.contains(node.table_id, node.index);
        case SpecCategory::Shape:
            return _shapes.contains(node.table_id, node.index);
        case SpecCategory::Spectrum:
            return _spectra.contains(node.table_id, node.index);
        case SpecCategory::Camera:
            return _cameras.contains(node.table_id, node.index);
        case SpecCategory::Film:
            return _films.contains(node.table_id, node.index);
        case SpecCategory::Filter:
            return _filters.contains(node.table_id, node.index);
        case SpecCategory::Sampler:
            return _samplers.contains(node.table_id, node.index);
        case SpecCategory::Integrator:
            return _integrators.contains(node.table_id, node.index);
        case SpecCategory::Count:
            return false;
        }
        return false;
    };
    auto flat_index = [&](SpecNode node) noexcept
    {
        return offsets[category_index(node.category)] + node.index;
    };

    for (auto& node : nodes)
    {
        for (auto dependency : node.dependencies)
        {
            if (!dependency_exists(dependency))
            {
                throw_validation_error(
                    node.meta.source,
                    luisa::format(
                        "{} spec '{}' references an invalid {} spec ref with index {}.",
                        category_name(node.category),
                        node.meta.name,
                        category_name(dependency.category),
                        dependency.index));
            }
        }
    }

    enum class VisitState : uint8_t
    {
        Unvisited,
        Visiting,
        Visited,
    };
    luisa::vector<VisitState> states(nodes.size(), VisitState::Unvisited);
    luisa::vector<size_t> stack;
    auto visit = [&](auto&& self, size_t node_index) -> void
    {
        states[node_index] = VisitState::Visiting;
        stack.emplace_back(node_index);
        for (auto dependency : nodes[node_index].dependencies)
        {
            auto dependency_index = flat_index(dependency);
            if (states[dependency_index] == VisitState::Unvisited)
            {
                self(self, dependency_index);
            }
            else if (states[dependency_index] == VisitState::Visiting)
            {
                auto cycle_begin = std::find(stack.begin(), stack.end(), dependency_index);
                luisa::string cycle;
                for (auto iter = cycle_begin; iter != stack.end(); ++iter)
                {
                    if (!cycle.empty())
                    {
                        cycle.append(" -> ");
                    }
                    cycle.append(luisa::format("{} '{}'", category_name(nodes[*iter].category), nodes[*iter].meta.name));
                }
                cycle.append(luisa::format(" -> {} '{}'", category_name(nodes[dependency_index].category), nodes[dependency_index].meta.name));
                throw_validation_error(nodes[dependency_index].meta.source, luisa::format("Spec dependency cycle: {}.", cycle));
            }
        }
        stack.pop_back();
        states[node_index] = VisitState::Visited;
    };

    for (size_t i = 0u; i < nodes.size(); i++)
    {
        if (states[i] == VisitState::Unvisited)
        {
            visit(visit, i);
        }
    }
}

} // namespace Yutrel
