#pragma once

#include "base/shape.h"
#include "scene/spec_base.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class MeshLoader
{
private:
    luisa::vector<Vertex> m_vertices;
    luisa::vector<Triangle> m_triangles;
    uint m_properties{};

public:
    [[nodiscard]] auto mesh() const noexcept { return MeshView{m_vertices, m_triangles}; }
    [[nodiscard]] auto properties() const noexcept { return m_properties; }

    [[nodiscard]] static luisa::shared_ptr<MeshLoader> load(std::filesystem::path path,
                                                            uint subdiv_level = 0u,
                                                            bool flip_uv      = false,
                                                            bool drop_normal  = false,
                                                            bool drop_uv      = false) noexcept;
};

class Mesh : public Shape
{
private:
    luisa::shared_ptr<MeshLoader> m_loader;

public:
    explicit Mesh(std::filesystem::path path) noexcept;
    ~Mesh() noexcept override = default;

public:
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return m_loader->mesh(); }
    [[nodiscard]] virtual uint vertex_properties() const noexcept override { return m_loader->properties(); }
};

class MeshShapeSpec final : public ShapeSpec
{
private:
    std::filesystem::path _path;

public:
    explicit MeshShapeSpec(std::filesystem::path path) noexcept
        : _path{std::move(path)} {}

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return _path; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override
    {
        if (_path.empty())
        {
            return spec_validation_error("Mesh path cannot be empty.");
        }
        std::error_code error;
        if (!std::filesystem::is_regular_file(_path, error))
        {
            return spec_validation_error("Mesh path must reference a regular file.");
        }
        return luisa::nullopt;
    }
    [[nodiscard]] const Shape* build(SceneBuilder& builder) const noexcept override;
};
} // namespace Yutrel
