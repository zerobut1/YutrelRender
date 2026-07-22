#pragma once

#include <luisa/core/stl.h>

#include "base/light.h"
#include "base/medium.h"
#include "base/surface.h"
#include "utils/vertex.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

struct MeshView
{
    luisa::span<const Vertex> vertices;
    luisa::span<const Triangle> triangles;
};

class Shape
{
public:
    class Handle;

public:
    static constexpr auto property_flag_has_vertex_normal = 1u << 0u;
    static constexpr auto property_flag_has_vertex_uv     = 1u << 1u;
    static constexpr auto property_flag_has_surface       = 1u << 2u;
    static constexpr auto property_flag_has_light         = 1u << 3u;
    static constexpr auto property_flag_has_medium        = 1u << 4u;
    static constexpr auto property_flag_maybe_non_opaque  = 1u << 5u;

public:
    Shape() noexcept          = default;
    virtual ~Shape() noexcept = default;

    Shape(const Shape&)            = delete;
    Shape& operator=(const Shape&) = delete;
    Shape(Shape&&)                 = delete;
    Shape& operator=(Shape&&)      = delete;

public:
    [[nodiscard]] virtual bool is_mesh() const noexcept { return false; }
    [[nodiscard]] virtual MeshView mesh() const noexcept { return {}; }
    [[nodiscard]] virtual uint vertex_properties() const noexcept { return 0u; }
};

struct ShapeInstance
{
    const Shape* shape;
    const Surface* surface;
    const Light* light;
    const Medium* inside_medium;
    const Medium* outside_medium;
    float4x4 transform{make_float4x4(1.0f)};
};

class Shape::Handle
{
public:
    static constexpr auto property_flag_bits = 10u;
    static constexpr auto property_flag_mask = (1u << property_flag_bits) - 1u;

    static constexpr auto buffer_base_max = (1u << (32u - property_flag_bits)) - 1u;

    static constexpr auto light_tag_bits     = 12u;
    static constexpr auto surface_tag_bits   = 12u;
    static constexpr auto medium_tag_bits    = 32u - light_tag_bits - surface_tag_bits;
    static constexpr auto surface_tag_max    = (1u << surface_tag_bits) - 1u;
    static constexpr auto light_tag_max      = (1u << light_tag_bits) - 1u;
    static constexpr auto medium_tag_max     = (1u << medium_tag_bits) - 1u;
    static constexpr auto light_tag_offset   = 0u;
    static constexpr auto surface_tag_offset = light_tag_offset + light_tag_bits;
    static constexpr auto medium_tag_offset  = surface_tag_offset + surface_tag_bits;

    static constexpr auto vertex_buffer_id_offset      = 0u;
    static constexpr auto triangle_buffer_id_offset    = 1u;
    static constexpr auto alias_table_buffer_id_offset = 2u;
    static constexpr auto pdf_buffer_id_offset         = 3u;

private:
    UInt m_buffer_base;
    UInt m_properties;
    UInt m_surface_tag;
    UInt m_light_tag;
    UInt m_medium_tag;
    UInt m_triangle_count;
    Float m_shadow_terminator;
    Float m_intersection_offset;

private:
    Handle(Expr<uint> buffer_base, Expr<uint> flags,
           Expr<uint> surface_tag, Expr<uint> light_tag, Expr<uint> medium_tag,
           Expr<uint> triangle_count,
           Expr<float> shadow_terminator, Expr<float> intersection_offset) noexcept
        : m_buffer_base{buffer_base},
          m_properties{flags},
          m_surface_tag{surface_tag},
          m_light_tag{light_tag},
          m_medium_tag{medium_tag},
          m_triangle_count{triangle_count},
          m_shadow_terminator{shadow_terminator},
          m_intersection_offset{intersection_offset} {}

public:
    Handle() noexcept = default;
    [[nodiscard]] static uint4 encode(uint buffer_base, uint flags,
                                      uint surface_tag, uint light_tag, uint medium_tag,
                                      uint tri_count, float shadow_terminator, float intersection_offset) noexcept;
    [[nodiscard]] static Shape::Handle decode(Expr<uint4> compressed) noexcept;

public:
    [[nodiscard]] auto geometry_buffer_base() const noexcept { return m_buffer_base; }
    [[nodiscard]] auto property_flags() const noexcept { return m_properties; }
    [[nodiscard]] auto vertex_buffer_id() const noexcept { return geometry_buffer_base() + Yutrel::Shape::Handle::vertex_buffer_id_offset; }
    [[nodiscard]] auto triangle_buffer_id() const noexcept { return geometry_buffer_base() + Yutrel::Shape::Handle::triangle_buffer_id_offset; }
    [[nodiscard]] auto triangle_count() const noexcept { return m_triangle_count; }
    [[nodiscard]] auto alias_table_buffer_id() const noexcept { return geometry_buffer_base() + Yutrel::Shape::Handle::alias_table_buffer_id_offset; }
    [[nodiscard]] auto pdf_buffer_id() const noexcept { return geometry_buffer_base() + Yutrel::Shape::Handle::pdf_buffer_id_offset; }
    [[nodiscard]] auto surface_tag() const noexcept { return m_surface_tag; }
    [[nodiscard]] auto light_tag() const noexcept { return m_light_tag; }
    [[nodiscard]] auto medium_tag() const noexcept { return m_medium_tag; }
    [[nodiscard]] auto test_property_flag(luisa::uint flag) const noexcept { return (property_flags() & flag) != 0u; }
    [[nodiscard]] auto has_vertex_normal() const noexcept { return test_property_flag(Yutrel::Shape::property_flag_has_vertex_normal); }
    [[nodiscard]] auto has_vertex_uv() const noexcept { return test_property_flag(Yutrel::Shape::property_flag_has_vertex_uv); }
    [[nodiscard]] auto has_light() const noexcept { return test_property_flag(Yutrel::Shape::property_flag_has_light); }
    [[nodiscard]] auto has_surface() const noexcept { return test_property_flag(Yutrel::Shape::property_flag_has_surface); }
    [[nodiscard]] auto has_medium() const noexcept { return test_property_flag(Yutrel::Shape::property_flag_has_medium); }
    [[nodiscard]] auto maybe_non_opaque() const noexcept { return test_property_flag(Yutrel::Shape::property_flag_maybe_non_opaque); }
    [[nodiscard]] auto shadow_terminator_factor() const noexcept { return m_shadow_terminator; }
    [[nodiscard]] auto intersection_offset_factor() const noexcept { return m_intersection_offset; }
};

} // namespace Yutrel
