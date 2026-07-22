#pragma once

#include <cmath>

#include <luisa/core/stl.h>
#include <luisa/dsl/syntax.h>

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class Renderer;

class Filter
{
public:
    static constexpr auto look_up_table_size = 64u;

public:
    struct Sample
    {
        Float2 offset;
        Float weight;
    };

public:
    class Instance
    {
    private:
        const Renderer& m_renderer;
        const Filter* m_filter;

        std::array<float, look_up_table_size> m_lut{};
        std::array<float, look_up_table_size - 1u> m_pdf{};
        std::array<float, look_up_table_size - 1u> m_alias_probs{};
        std::array<uint, look_up_table_size - 1u> m_alias_indices{};

    public:
        Instance(const Renderer& renderer, const Filter* filter) noexcept;
        virtual ~Instance() noexcept = default;

        Instance() noexcept                           = delete;
        Instance(const Instance&) noexcept            = delete;
        Instance(Instance&&) noexcept                 = delete;
        Instance& operator=(const Instance&) noexcept = delete;
        Instance& operator=(Instance&&) noexcept      = delete;

    public:
        template <typename T = Filter>
            requires std::is_base_of_v<Filter, T>
        [[nodiscard]] auto base() const noexcept
        {
            return static_cast<const T*>(m_filter);
        }
        [[nodiscard]] auto look_up_table() const noexcept { return luisa::span{m_lut}; }
        [[nodiscard]] auto pdf_table() const noexcept { return luisa::span{m_pdf}; }
        [[nodiscard]] auto alias_table_indices() const noexcept { return luisa::span{m_alias_indices}; }
        [[nodiscard]] auto alias_table_probabilities() const noexcept { return luisa::span{m_alias_probs}; }
        [[nodiscard]] virtual Sample sample(Expr<float2> u) const noexcept;
    };

private:
    float m_radius;

public:
    explicit Filter(float radius) noexcept
        : m_radius{radius} {}
    virtual ~Filter() noexcept = default;

    Filter(const Filter&) noexcept            = delete;
    Filter(Filter&&) noexcept                 = delete;
    Filter& operator=(const Filter&) noexcept = delete;
    Filter& operator=(Filter&&) noexcept      = delete;

public:
    [[nodiscard]] virtual luisa::unique_ptr<Filter::Instance> build(const Renderer& renderer) const noexcept;

    [[nodiscard]] auto radius() const noexcept { return m_radius; }

    [[nodiscard]] virtual float evaluate(float x) const noexcept = 0;
};
} // namespace Yutrel
