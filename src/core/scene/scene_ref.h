#pragma once

#include <compare>
#include <cstdint>

namespace Yutrel
{

class TextureSpec;
class SurfaceSpec;
class MediumSpec;
class LightSpec;
class EnvironmentSpec;
class ShapeSpec;
class SpectrumSpec;
class CameraSpec;
class FilmSpec;
class FilterSpec;
class SamplerSpec;
class IntegratorSpec;

template <typename Spec>
class SpecTable;

template <typename Spec>
class SceneRef
{
private:
    uint32_t _index;
    uint64_t _table_id;

private:
    explicit SceneRef(uint32_t index, uint64_t table_id) noexcept
        : _index{index}, _table_id{table_id}
    {
    }

    friend class SpecTable<Spec>;

public:
    SceneRef() = delete;

    [[nodiscard]] uint32_t index() const noexcept { return _index; }
    [[nodiscard]] uint64_t table_id() const noexcept { return _table_id; }
    auto operator<=>(const SceneRef&) const noexcept = default;
};

using TextureRef     = SceneRef<TextureSpec>;
using SurfaceRef     = SceneRef<SurfaceSpec>;
using MediumRef      = SceneRef<MediumSpec>;
using LightRef       = SceneRef<LightSpec>;
using EnvironmentRef = SceneRef<EnvironmentSpec>;
using ShapeRef       = SceneRef<ShapeSpec>;
using SpectrumRef    = SceneRef<SpectrumSpec>;
using CameraRef      = SceneRef<CameraSpec>;
using FilmRef        = SceneRef<FilmSpec>;
using FilterRef      = SceneRef<FilterSpec>;
using SamplerRef     = SceneRef<SamplerSpec>;
using IntegratorRef  = SceneRef<IntegratorSpec>;

} // namespace Yutrel
