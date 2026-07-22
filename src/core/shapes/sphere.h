#pragma once

#include <luisa/core/stl/memory.h>

#include "base/shape.h"
#include "scene/spec_base.h"

namespace Yutrel
{
using namespace luisa;

class SphereGeometry;

class Sphere final : public Shape
{
public:
    static constexpr uint default_subdivision = 4u;
    static constexpr uint max_subdivision     = 8u;

private:
    luisa::shared_ptr<const SphereGeometry> m_geometry;

public:
    explicit Sphere(float radius, uint subdivision = default_subdivision) noexcept;
    ~Sphere() noexcept override = default;

public:
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override;
    [[nodiscard]] uint vertex_properties() const noexcept override
    {
        return property_flag_has_vertex_normal |
               property_flag_has_vertex_uv;
    }
};

class SphereShapeSpec final : public ShapeSpec
{
private:
    float m_radius;
    uint m_subdivision;

public:
    explicit SphereShapeSpec(float radius, uint subdivision = Sphere::default_subdivision) noexcept
        : m_radius{radius}, m_subdivision{subdivision} {}

    [[nodiscard]] float radius() const noexcept { return m_radius; }
    [[nodiscard]] uint subdivision() const noexcept { return m_subdivision; }

    [[nodiscard]] luisa::optional<luisa::string> validate() const noexcept override;
    [[nodiscard]] const Shape* build(SceneBuilder& builder) const noexcept override;
};

} // namespace Yutrel
