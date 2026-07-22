#include "spectrum.h"

#include "utils/color_space.h"

namespace Yutrel
{
Spectrum::Instance::Instance(Renderer& renderer, CommandBuffer& command_buffer, const Spectrum* spectrum) noexcept
    : m_renderer(renderer), m_spectrum(spectrum),
      m_cie_x(SPD::create_cie_x(renderer, command_buffer)),
      m_cie_y(SPD::create_cie_y(renderer, command_buffer)),
      m_cie_z(SPD::create_cie_z(renderer, command_buffer))
{
}

Float Spectrum::Instance::cie_y(const SampledWavelengths& swl, const SampledSpectrum& sp) const noexcept
{
    auto sum                = def(0.f);
    constexpr auto safe_div = [](auto&& a, auto&& b) noexcept
    {
        return ite(b == 0.0f, 0.0f, a / b);
    };
    for (auto i = 0u; i < swl.dimension(); i++)
    {
        sum += safe_div(m_cie_y.sample(swl.lambda(i)) * sp[i], swl.pdf(i));
    }
    auto denom = static_cast<float>(swl.dimension()) * SPD::cie_y_integral();
    return sum / denom;
}

Float3 Spectrum::Instance::cie_xyz(const SampledWavelengths& swl, const SampledSpectrum& sp) const noexcept
{
    constexpr auto safe_div = [](auto a, auto b) noexcept
    {
        return ite(b == 0.f, 0.f, a / b);
    };
    auto sum = def(make_float3());
    for (auto i = 0u; i < swl.dimension(); i++)
    {
        auto lambda = swl.lambda(i);
        auto pdf    = swl.pdf(i);
        sum += make_float3(safe_div(m_cie_x.sample(lambda) * sp[i], pdf),
                           safe_div(m_cie_y.sample(lambda) * sp[i], pdf),
                           safe_div(m_cie_z.sample(lambda) * sp[i], pdf));
    }
    auto denom = static_cast<float>(swl.dimension()) * SPD::cie_y_integral();
    return sum / denom;
}

Float3 Spectrum::Instance::srgb(const SampledWavelengths& swl, const SampledSpectrum& sp) const noexcept
{
    auto xyz = cie_xyz(swl, sp);
    return cie_xyz_to_linear_srgb(xyz);
}

} // namespace Yutrel
