#pragma once

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>

#include "base/spd.h"
#include "utils/color_space.h"
#include "utils/command_buffer.h"

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

class RGB2SpectrumTable
{

public:
    static constexpr auto s_resolution = 64u;
    using coefficient_table_type       = const float[3][s_resolution][s_resolution][s_resolution][4];

private:
    const coefficient_table_type& m_coefficients;

private:
    [[nodiscard]] inline static auto inverse_smooth_step(auto x) noexcept
    {
        return .5f - sin(asin(1.f - 2.f * x) * (1.f / 3.f));
    }

public:
    explicit constexpr RGB2SpectrumTable(const coefficient_table_type& coefficients) noexcept
        : m_coefficients{coefficients} {}

    constexpr RGB2SpectrumTable(RGB2SpectrumTable&&) noexcept      = default;
    constexpr RGB2SpectrumTable(const RGB2SpectrumTable&) noexcept = default;

    [[nodiscard]] static RGB2SpectrumTable srgb() noexcept;

    [[nodiscard]] Float4 decode_albedo(const BindlessArray& array, Expr<uint> base_index, Expr<float3> rgb_in) const noexcept
    {
        auto rgb               = clamp(rgb_in, 0.0f, 1.0f);
        static Callable decode = [](BindlessVar array, UInt base_index, Float3 rgb) noexcept
        {
            auto c = make_float3(0.0f, 0.0f, (rgb[0] - 0.5f) * rsqrt(rgb[0] * (1.0f - rgb[0])));
            $if(!(rgb[0] == rgb[1] & rgb[1] == rgb[2]))
            {
                // Find maximum component and compute remapped component values
                auto maxc = ite(
                    rgb[0] > rgb[1],
                    ite(rgb[0] > rgb[2], 0u, 2u),
                    ite(rgb[1] > rgb[2], 1u, 2u));
                auto z  = rgb[maxc];
                auto x  = rgb[(maxc + 1u) % 3u] / z;
                auto y  = rgb[(maxc + 2u) % 3u] / z;
                auto zz = inverse_smooth_step(inverse_smooth_step(z));

                // Trilinearly interpolate sigmoid polynomial coefficients _c_
                auto coord = fma(
                    make_float3(x, y, zz),
                    (s_resolution - 1.0f) / s_resolution,
                    0.5f / s_resolution);
                c = array.tex3d(base_index + maxc).sample(coord).xyz();
            };
            return c;
        };
        return make_float4(decode(Expr{array}, base_index, rgb), linear_srgb_to_cie_y(rgb));
    }

    [[nodiscard]] float4 decode_albedo(float3 rgb_in) const noexcept
    {
        auto rgb = clamp(rgb_in, 0.0f, 1.0f);
        if (rgb[0] == rgb[1] && rgb[1] == rgb[2])
        {
            auto s = (rgb[0] - 0.5f) / std::sqrt(rgb[0] * (1.0f - rgb[0]));
            return make_float4(0.0f, 0.0f, s, linear_srgb_to_cie_y(rgb));
        }
        // Find maximum component and compute remapped component values
        auto maxc = (rgb[0] > rgb[1]) ? ((rgb[0] > rgb[2]) ? 0u : 2u) : ((rgb[1] > rgb[2]) ? 1u : 2u);
        auto z    = rgb[maxc];
        auto x    = rgb[(maxc + 1u) % 3u] * (s_resolution - 1u) / z;
        auto y    = rgb[(maxc + 2u) % 3u] * (s_resolution - 1u) / z;

        auto zz = inverse_smooth_step(inverse_smooth_step(z)) * (s_resolution - 1u);

        // Compute integer indices and offsets for coefficient interpolation
        auto xi = std::min(static_cast<uint>(x), s_resolution - 2u);
        auto yi = std::min(static_cast<uint>(y), s_resolution - 2u);
        auto zi = std::min(static_cast<uint>(zz), s_resolution - 2u);
        auto dx = x - static_cast<float>(xi);
        auto dy = y - static_cast<float>(yi);
        auto dz = zz - static_cast<float>(zi);

        // Trilinearly interpolate sigmoid polynomial coefficients _c_
        auto c = make_float3();
        for (auto i = 0u; i < 3u; i++)
        {
            // Define _co_ lambda for looking up sigmoid polynomial coefficients
            auto co = [=, this](int dx, int dy, int dz) noexcept
            {
                return m_coefficients[maxc][zi + dz][yi + dy][xi + dx][i];
            };
            c[i] = luisa::lerp(luisa::lerp(luisa::lerp(co(0, 0, 0), co(1, 0, 0), dx),
                                           luisa::lerp(co(0, 1, 0), co(1, 1, 0), dx),
                                           dy),
                               luisa::lerp(luisa::lerp(co(0, 0, 1), co(1, 0, 1), dx),
                                           luisa::lerp(co(0, 1, 1), co(1, 1, 1), dx),
                                           dy),
                               dz);
        }
        return make_float4(c, linear_srgb_to_cie_y(rgb));
    }

    [[nodiscard]] Float4 decode_unbounded(
        const BindlessArray& array, Expr<uint> base_index, Expr<float3> rgb_in) const noexcept
    {
        auto rgb   = max(rgb_in, 0.0f);
        auto m     = max(max(rgb.x, rgb.y), rgb.z);
        auto scale = 2.0f * m;
        auto c     = decode_albedo(
            array,
            base_index,
            ite(scale == 0.0f, 0.0f, rgb / scale));
        return make_float4(c.xyz(), scale);
    }

    [[nodiscard]] float4 decode_unbounded(float3 rgb) const noexcept
    {
        auto m     = luisa::max(luisa::max(rgb.x, rgb.y), rgb.z);
        auto scale = 2.0f * m;
        auto c     = decode_albedo(scale == 0.f ? make_float3() : rgb / scale);
        return make_float4(c.xyz(), scale);
    }

    void encode(CommandBuffer& command_buffer,
                VolumeView<float> t0, VolumeView<float> t1, VolumeView<float> t2) const noexcept
    {
        command_buffer << t0.copy_from(m_coefficients[0])
                       << t1.copy_from(m_coefficients[1])
                       << t2.copy_from(m_coefficients[2])
                       << luisa::compute::commit();
    }
};

extern RGB2SpectrumTable::coefficient_table_type sRGBToSpectrumTable_Data;

template <typename X, typename C0>
[[nodiscard]] inline auto& polynomial(const X& x, const C0& c0) noexcept { return c0; }

template <typename X, typename C0, typename... C>
[[nodiscard]] inline auto polynomial(const X& x, const C0& c0, const C&... c) noexcept
{
    return x * polynomial(x, c...) + c0;
}

class RGBSigmoidPolynomial
{

private:
    Float3 m_c;

private:
    [[nodiscard]] static auto _fma(Expr<float> a, Expr<float> b, Expr<float> c) noexcept
    {
        return a * b + c;
    }

    // sigmoid
    [[nodiscard]] static auto _s(Expr<float> x) noexcept
    {
        return ite(compute::isinf(x), cast<float>(x > 0.f), .5f + x / (2.f * sqrt(1.f + x * x)));
    }

public:
    RGBSigmoidPolynomial() noexcept = default;
    RGBSigmoidPolynomial(Expr<float> c0, Expr<float> c1, Expr<float> c2) noexcept
        : m_c{make_float3(c0, c1, c2)} {}
    explicit RGBSigmoidPolynomial(Expr<float3> c) noexcept : m_c{c} {}
    [[nodiscard]] Float operator()(Expr<float> lambda) const noexcept
    {
        return _s(polynomial(lambda, m_c.z, m_c.y, m_c.x)); // c0 * x * x + c1 * x + c2
    }
};

class RGBAlbedoSpectrum
{

private:
    RGBSigmoidPolynomial m_rsp;

public:
    explicit RGBAlbedoSpectrum(RGBSigmoidPolynomial rsp) noexcept : m_rsp{std::move(rsp)} {}
    [[nodiscard]] auto sample(Expr<float> lambda) const noexcept { return m_rsp(lambda); }
};

class RGBIlluminantSpectrum
{

private:
    RGBSigmoidPolynomial m_rsp;
    Float m_scale;
    SPD m_illum;

public:
    RGBIlluminantSpectrum(RGBSigmoidPolynomial rsp, Expr<float> scale, SPD illum) noexcept
        : m_rsp{std::move(rsp)}, m_scale{scale}, m_illum{illum} {}
    [[nodiscard]] auto sample(Expr<float> lambda) const noexcept
    {
        return m_rsp(lambda) * m_scale * m_illum.sample(lambda);
    }
    [[nodiscard]] auto scale() const noexcept { return m_scale; }
};

} // namespace Yutrel