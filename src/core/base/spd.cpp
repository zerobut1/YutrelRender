#include "spd.h"

#include "base/renderer.h"
#include "utils/spectra.h"

namespace Yutrel
{

SPD::SPD(Renderer& renderer, uint buffer_id, float sample_interval) noexcept
    : m_renderer{renderer}, m_buffer_id{buffer_id}, m_sample_interval{sample_interval} {}

static inline auto densely_sampled_spectrum_integral(uint t, const float* spec) noexcept
{
    auto sum = 0.0;
    auto tt  = static_cast<float>(t);
    auto n   = (visible_wavelength_max - visible_wavelength_min) / tt;
    LUISA_ASSERT(n == std::floor(n), "Invalid SPD sample interval.");
    auto nn = static_cast<uint>(n) + 1u;
    for (auto i = 0u; i < nn - 1u; i++)
    {
        sum += 0.5f * (cie_y_samples[i * t] + cie_y_samples[(i + 1u) * t]);
    }
    return static_cast<float>(sum * tt);
}

static inline auto downsample_densely_sampled_spectrum(uint t, const float* spec) noexcept
{
    auto n = (visible_wavelength_max - visible_wavelength_min) / static_cast<float>(t);
    LUISA_ASSERT(n == std::floor(n), "Invalid SPD sample interval.");
    auto nn = static_cast<uint>(n) + 1u;
    luisa::vector<float> samples(nn);
    for (auto x = 0u; x < nn; x++)
    {
        samples[x] = spec[x * t];
    }
    return samples;
}

static constexpr auto spd_lut_interval = 1u;

SPD SPD::create_cie_x(Renderer& renderer, CommandBuffer& cb) noexcept
{
    auto buffer = renderer.register_named_id(
        "__internal_spd_cie_x",
        [&renderer, &cb]
    {
        auto s = downsample_densely_sampled_spectrum(
            spd_lut_interval,
            cie_x_samples.data());
        auto [view, index] = renderer.bindless_arena_buffer<float>(s.size());
        cb << view.copy_from(s.data()) << compute::commit();
        return index;
    });
    return {renderer, buffer, spd_lut_interval};
}

SPD SPD::create_cie_y(Renderer& renderer, CommandBuffer& cb) noexcept
{
    auto buffer = renderer.register_named_id(
        "__internal_spd_cie_y",
        [&renderer, &cb]
    {
        auto s = downsample_densely_sampled_spectrum(
            spd_lut_interval,
            cie_y_samples.data());
        auto [view, index] = renderer.bindless_arena_buffer<float>(s.size());
        cb << view.copy_from(s.data()) << compute::commit();
        return index;
    });
    return {renderer, buffer, spd_lut_interval};
}

SPD SPD::create_cie_z(Renderer& renderer, CommandBuffer& cb) noexcept
{
    auto buffer = renderer.register_named_id(
        "__internal_spd_cie_z",
        [&renderer, &cb]
    {
        auto s = downsample_densely_sampled_spectrum(
            spd_lut_interval,
            cie_z_samples.data());
        auto [view, index] = renderer.bindless_arena_buffer<float>(s.size());
        cb << view.copy_from(s.data()) << compute::commit();
        return index;
    });
    return {renderer, buffer, spd_lut_interval};
}

SPD SPD::create_cie_d65(Renderer& renderer, CommandBuffer& cb) noexcept
{
    auto buffer = renderer.register_named_id(
        "__internal_spd_cie_d65",
        [&renderer, &cb]
    {
        auto s = downsample_densely_sampled_spectrum(
            spd_lut_interval,
            cie_d65_samples.data());
        auto [view, index] = renderer.bindless_arena_buffer<float>(s.size());
        cb << view.copy_from(s.data()) << compute::commit();
        return index;
    });
    return {renderer, buffer, spd_lut_interval};
}

float SPD::cie_y_integral() noexcept
{
    static auto integral = densely_sampled_spectrum_integral(
        spd_lut_interval,
        cie_y_samples.data());
    return integral;
}

Float SPD::sample(Expr<float> lambda) const noexcept
{
    using namespace luisa::compute;
    auto t            = (clamp(lambda, visible_wavelength_min, visible_wavelength_max) - visible_wavelength_min) / m_sample_interval;
    auto sample_count = static_cast<uint>((visible_wavelength_max - visible_wavelength_min) / m_sample_interval) + 1u;
    auto i            = cast<uint>(min(t, static_cast<float>(sample_count - 2u)));
    auto s0           = m_renderer.buffer<float>(m_buffer_id).read(i);
    auto s1           = m_renderer.buffer<float>(m_buffer_id).read(i + 1u);
    return lerp(s0, s1, fract(t));
}

} // namespace Yutrel
