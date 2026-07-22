#include "filter.h"

#include "utils/sampling.h"

namespace Yutrel
{
luisa::unique_ptr<Filter::Instance> Filter::build(const Renderer& renderer) const noexcept
{
    return luisa::make_unique<Filter::Instance>(renderer, this);
}

Filter::Instance::Instance(const Renderer& renderer, const Filter* filter) noexcept
    : m_renderer(renderer),
      m_filter(filter)
{
    static constexpr auto n     = Filter::look_up_table_size - 1u;
    static constexpr auto inv_n = 1.0f / static_cast<float>(n);
    std::array<float, n> abs_f{};
    m_lut[0u]     = filter->evaluate(-filter->radius());
    auto integral = 0.0f;
    for (auto i = 0u; i < n; i++)
    {
        auto x        = static_cast<float>(i + 1u) * inv_n * 2.0f - 1.0f;
        m_lut[i + 1u] = filter->evaluate(x * filter->radius());
        auto f_mid    = 0.5f * (m_lut[i] + m_lut[i + 1u]);
        integral += f_mid;
        abs_f[i] = abs(f_mid);
    }

    auto inv_integral = 1.0f / integral;
    for (auto& f : m_lut)
    {
        f *= inv_integral;
    }
    auto [alias_table, pdf] = create_alias_table(abs_f);
    LUISA_ASSERT(alias_table.size() == n && pdf.size() == n);
    for (auto i = 0u; i < n; i++)
    {
        m_pdf[i]           = pdf[i];
        m_alias_probs[i]   = alias_table[i].prob;
        m_alias_indices[i] = alias_table[i].alias;
    }
}

Filter::Sample Filter::Instance::sample(Expr<float2> u) const noexcept
{
    Constant lut           = look_up_table();
    Constant pdfs          = pdf_table();
    Constant alias_indices = alias_table_indices();
    Constant alias_probs   = alias_table_probabilities();

    auto n        = look_up_table_size - 1u;
    auto [ix, ux] = sample_alias_table(alias_probs, alias_indices, n, u.x);
    auto [iy, uy] = sample_alias_table(alias_probs, alias_indices, n, u.y);
    auto pdf      = pdfs[ix] * pdfs[iy];
    auto f        = lerp(lut[ix], lut[ix + 1u], ux) * lerp(lut[iy], lut[iy + 1u], uy);
    auto p        = make_float2(make_uint2(ix, iy)) + make_float2(ux, uy);
    auto inv_size = 1.0f / static_cast<float>(n);
    auto pixel    = (p * inv_size * 2.0f - 1.0f) * m_filter->radius();
    return {pixel, f / pdf};
}

} // namespace Yutrel
