#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "scene/scene_ref.h"
#include "scene/source_location.h"

namespace Yutrel
{

template <typename Spec>
class SpecTable
{
private:
    struct Entry
    {
        SpecMeta meta;
        luisa::unique_ptr<Spec> spec;
        luisa::optional<SourceLocation> first_use;
    };

private:
    inline static std::atomic_uint64_t _next_table_id{1u};

private:
    uint64_t _table_id;
    luisa::string _category;
    luisa::vector<Entry> _entries;
    luisa::unordered_map<luisa::string, SceneRef<Spec>> _named_refs;
    uint32_t _anonymous_count{};

public:
    explicit SpecTable(luisa::string category) noexcept
        : _table_id{_next_table_id.fetch_add(1u, std::memory_order_relaxed)},
          _category{std::move(category)}
    {
    }

    SpecTable(SpecTable&&) noexcept            = default;
    SpecTable& operator=(SpecTable&&) noexcept = default;
    SpecTable(const SpecTable&)                = delete;
    SpecTable& operator=(const SpecTable&)     = delete;

    [[nodiscard]] SceneRef<Spec> reference(luisa::string name, SourceLocation use_site)
    {
        _validate_name(name, use_site);
        if (auto iter = _named_refs.find(name); iter != _named_refs.end())
        {
            return iter->second;
        }
        auto ref = _append(Entry{
            .meta      = SpecMeta{.name = std::move(name), .source = use_site},
            .spec      = nullptr,
            .first_use = std::move(use_site),
        });
        _named_refs.emplace(_entries[ref.index()].meta.name, ref);
        return ref;
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, Spec>
    [[nodiscard]] SceneRef<Spec> add(SpecMeta meta, Args&&... args)
    {
        _validate_name(meta.name, meta.source);
        if (auto iter = _named_refs.find(meta.name); iter != _named_refs.end())
        {
            auto ref    = iter->second;
            auto& entry = _entry(ref);
            if (entry.spec)
            {
                _throw_error(meta.source, luisa::format("Duplicate {} spec '{}'; first definition is at {}.", _category, meta.name, format_source_location(entry.meta.source)));
            }
            entry.meta      = std::move(meta);
            entry.spec      = luisa::make_unique<Impl>(std::forward<Args>(args)...);
            entry.first_use = luisa::nullopt;
            return ref;
        }
        auto ref = _append(Entry{
            .meta      = std::move(meta),
            .spec      = luisa::make_unique<Impl>(std::forward<Args>(args)...),
            .first_use = luisa::nullopt,
        });
        _named_refs.emplace(_entries[ref.index()].meta.name, ref);
        return ref;
    }

    template <typename Impl, typename... Args>
        requires std::derived_from<Impl, Spec>
    [[nodiscard]] SceneRef<Spec> add_anonymous(SourceLocation source, Args&&... args)
    {
        luisa::string name;
        do
        {
            name = luisa::format("@{}/{}", _category, _anonymous_count++);
        } while (_named_refs.find(name) != _named_refs.end());
        return add<Impl>(SpecMeta{.name = std::move(name), .source = std::move(source)}, std::forward<Args>(args)...);
    }

    [[nodiscard]] bool contains(SceneRef<Spec> ref) const noexcept
    {
        return ref.table_id() == _table_id && ref.index() < _entries.size();
    }

    [[nodiscard]] bool contains(uint64_t table_id, uint32_t index) const noexcept
    {
        return table_id == _table_id && index < _entries.size();
    }

    [[nodiscard]] size_t size() const noexcept { return _entries.size(); }
    [[nodiscard]] uint64_t table_id() const noexcept { return _table_id; }
    [[nodiscard]] luisa::string_view category() const noexcept { return _category; }

    [[nodiscard]] const SpecMeta& meta(SceneRef<Spec> ref) const
    {
        return _entry(ref).meta;
    }

    [[nodiscard]] const Spec& spec(SceneRef<Spec> ref) const
    {
        auto& entry = _entry(ref);
        if (!entry.spec)
        {
            auto location = entry.first_use.value_or(entry.meta.source);
            _throw_error(location, luisa::format("Undefined {} spec '{}'.", _category, entry.meta.name));
        }
        return *entry.spec;
    }

    void validate_definitions() const
    {
        for (auto& entry : _entries)
        {
            if (!entry.spec)
            {
                auto location = entry.first_use.value_or(entry.meta.source);
                _throw_error(location, luisa::format("Undefined {} spec '{}'.", _category, entry.meta.name));
            }
        }
    }

    template <typename Visitor>
    void visit_entries(Visitor&& visitor) const
    {
        for (uint32_t index = 0u; index < _entries.size(); index++)
        {
            auto& entry = _entries[index];
            visitor(SceneRef<Spec>{index, _table_id}, entry.meta, entry.spec.get());
        }
    }

private:
    [[nodiscard]] SceneRef<Spec> _append(Entry entry)
    {
        if (_entries.size() >= std::numeric_limits<uint32_t>::max())
        {
            _throw_error(entry.meta.source, luisa::format("Too many {} specs.", _category));
        }
        auto ref = SceneRef<Spec>{static_cast<uint32_t>(_entries.size()), _table_id};
        _entries.emplace_back(std::move(entry));
        return ref;
    }

    [[nodiscard]] Entry& _entry(SceneRef<Spec> ref)
    {
        if (!contains(ref))
        {
            _throw_invalid_ref(ref);
        }
        return _entries[ref.index()];
    }

    [[nodiscard]] const Entry& _entry(SceneRef<Spec> ref) const
    {
        if (!contains(ref))
        {
            _throw_invalid_ref(ref);
        }
        return _entries[ref.index()];
    }

    void _validate_name(luisa::string_view name, const SourceLocation& source) const
    {
        if (name.empty())
        {
            _throw_error(source, luisa::format("{} spec name cannot be empty.", _category));
        }
    }

    [[noreturn]] void _throw_invalid_ref(SceneRef<Spec> ref) const
    {
        if (ref.table_id() != _table_id)
        {
            _throw_error({}, luisa::format("{} spec ref belongs to a different scene spec.", _category));
        }
        _throw_error({}, luisa::format("{} spec ref index {} is out of bounds (size {}).", _category, ref.index(), _entries.size()));
    }

    [[noreturn]] static void _throw_error(const SourceLocation& source, luisa::string message)
    {
        auto formatted = luisa::format("{}: {}", format_source_location(source), message);
        throw std::runtime_error{formatted.c_str()};
    }
};

} // namespace Yutrel
