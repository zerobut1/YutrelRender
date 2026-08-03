#pragma once

#include <cstdint>

#include <luisa/core/basic_types.h>

namespace Yutrel
{

class Shape;

enum class ExternalMeshOp : uint32_t
{
    add_or_replace,
    transform,
    remove,
};

struct ExternalMeshUpdate
{
    uint64_t id;
    ExternalMeshOp operation;
    const Shape* shape;
    luisa::float4x4 local_to_world;
};

struct ExternalDirectionalLightState
{
    luisa::float3 color;
    float intensity;
    luisa::float3 direction;
    uint32_t enabled;
};

}// namespace Yutrel
