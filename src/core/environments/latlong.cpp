#include "latlong.h"

#include <algorithm>
#include <numeric>

#include <luisa/core/logging.h>

#include "base/interaction.h"
#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/color_space.h"
#include "utils/sampling.h"

namespace Yutrel
{
namespace
{
constexpr auto inv_two_pi = 0.5f * inv_pi;
constexpr auto two_pi = 2.0f * pi;
constexpr auto inv_two_pi_squared = 0.5f * inv_pi * inv_pi;
}

Float2 LatLongEnvironment::direction_to_uv(Expr<float3> direction) noexcept
{
    auto w = normalize(direction);
    auto phi = atan2(w.x, w.z);
    auto u = fract(phi * inv_two_pi + 0.5f);
    auto v = acos(clamp(w.y, -1.0f, 1.0f)) * inv_pi;
    return make_float2(u, v);
}

Float3 LatLongEnvironment::uv_to_direction(Expr<float2> uv) noexcept
{
    auto phi = (uv.x - 0.5f) * two_pi;
    auto theta = uv.y * pi;
    auto sin_theta = sin(theta);
    return normalize(make_float3(
        sin(phi) * sin_theta,
        cos(theta),
        cos(phi) * sin_theta));
}

SampledSpectrum LatLongEnvironment::Instance::_evaluate_radiance(
    Expr<float2> uv, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    auto half_texel_v = 0.5f / static_cast<float>(_resolution.y);
    auto sample_uv = make_float2(
        fract(uv.x),
        clamp(uv.y, half_texel_v, 1.0f - half_texel_v));
    auto it = Interaction::from_uv(sample_uv);
    return _emission->evaluate_illuminant_spectrum(it, swl, time).value *
           base<LatLongEnvironment>()->scale();
}

Float LatLongEnvironment::Instance::_directional_pdf(
    Expr<float> pdf_uv, Expr<float> theta) noexcept
{
    auto sin_theta = sin(theta);
    auto inv_sin_theta = ite(sin_theta > 0.0f, 1.0f / sin_theta, 0.0f);
    return pdf_uv * inv_sin_theta * inv_two_pi_squared;
}

Environment::Evaluation LatLongEnvironment::Instance::evaluate(
    Expr<float3> wi, const SampledWavelengths& swl, Expr<float> time,
    bool allow_incomplete_pdf) const noexcept
{
    auto pdf_offset = allow_incomplete_pdf ? _pdf_distribution_stride : 0u;
    auto result = Evaluation::zero(swl.dimension());
    $outline
    {
        Float3x3 transform_to_world = base<LatLongEnvironment>()->transform_to_world();
        auto wi_local = normalize(transpose(transform_to_world) * normalize(wi));
        auto uv = direction_to_uv(wi_local);
        auto size = make_float2(_resolution);
        auto pixel = make_uint2(
            cast<uint>(clamp(uv.x * size.x, 0.0f, size.x - 1.0f)),
            cast<uint>(clamp(uv.y * size.y, 0.0f, size.y - 1.0f)));
        auto pdf_uv = renderer().buffer<float>(_pdf_buffer_id).read(
            pdf_offset + pixel.y * _resolution.x + pixel.x);
        result = {
            .L = _evaluate_radiance(uv, swl, time),
            .pdf = _directional_pdf(pdf_uv, uv.y * pi),
            .p = make_float3(0.0f),
            .ng = make_float3(0.0f),
        };
    };
    return result;
}

Environment::Sample LatLongEnvironment::Instance::sample(
    const SampledWavelengths& swl, Expr<float> time, Expr<float2> u,
    bool allow_incomplete_pdf) const noexcept
{
    auto alias_offset = allow_incomplete_pdf ? _alias_distribution_stride : 0u;
    auto pdf_offset = allow_incomplete_pdf ? _pdf_distribution_stride : 0u;
    auto result = Sample::zero(swl.dimension());
    $outline
    {
        auto aliases = renderer().buffer<AliasEntry>(_alias_buffer_id);
        auto [iy, uy] = sample_alias_table(
            aliases, _resolution.y, u.y, alias_offset);
        auto row_offset = alias_offset + _resolution.y + iy * _resolution.x;
        auto [ix, ux] = sample_alias_table(aliases, _resolution.x, u.x, row_offset);
        auto uv = make_float2(cast<float>(ix) + ux, cast<float>(iy) + uy) /
                  make_float2(_resolution);
        auto pdf_uv = renderer().buffer<float>(_pdf_buffer_id).read(
            pdf_offset + iy * _resolution.x + ix);
        auto wi_local = uv_to_direction(uv);
        Float3x3 transform_to_world = base<LatLongEnvironment>()->transform_to_world();
        result = {
            .eval = {
                .L = _evaluate_radiance(uv, swl, time),
                .pdf = _directional_pdf(pdf_uv, uv.y * pi),
                .p = make_float3(0.0f),
                .ng = make_float3(0.0f),
            },
            .wi = normalize(transform_to_world * wi_local),
            .delta = false,
        };
    };
    return result;
}

luisa::unique_ptr<Environment::Instance> LatLongEnvironment::build(
    Renderer& renderer, CommandBuffer& command_buffer) const noexcept
{
    if (_scale == 0.0f)
    {
        return nullptr;
    }

    auto resolution = _emission->resolution();
    if (resolution.x == 0u || resolution.y == 0u ||
        static_cast<uint64_t>(resolution.x) != 2ull * resolution.y)
    {
        LUISA_ERROR_WITH_LOCATION(
            "Lat-long environment image must have a non-empty 2:1 resolution, got {}x{}.",
            resolution.x, resolution.y);
    }
    if (_emission->channels() < 3u)
    {
        LUISA_ERROR_WITH_LOCATION(
            "Lat-long environment image must have RGB channels, got {} channel(s).",
            _emission->channels());
    }

    auto texture = renderer.build_texture(command_buffer, _emission);
    if (renderer.bindless_array().dirty())
    {
        command_buffer << renderer.bindless_array().update() << commit();
    }

    auto pixel_count = static_cast<size_t>(resolution.x) * resolution.y;
    luisa::vector<float> weights(pixel_count);
    auto device_weights = renderer.device().create_buffer<float>(pixel_count);
    Kernel2D generate_weights_kernel = [&]() noexcept
    {
        auto pixel = dispatch_id().xy();
        auto uv = (make_float2(pixel) + 0.5f) / make_float2(resolution);
        auto rgb = texture->evaluate(Interaction::from_uv(uv), 0.0f).xyz();
        auto finite_rgb = !any(compute::isnan(rgb) || compute::isinf(rgb));
        auto luminance = linear_srgb_to_cie_y(max(rgb, 0.0f));
        auto finite_luminance = !compute::isnan(luminance) & !compute::isinf(luminance);
        auto sin_theta = sin(pi * uv.y);
        auto weight = ite(
            finite_rgb & finite_luminance,
            min(max(luminance, 0.0f) * sin_theta, 1.0e8f),
            0.0f);
        device_weights->write(pixel.y * resolution.x + pixel.x, weight);
    };
    auto generate_weights = renderer.device().compile(generate_weights_kernel);
    command_buffer << generate_weights().dispatch(resolution)
                   << device_weights.copy_to(weights.data())
                   << synchronize();

    auto total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(total_weight > 0.0) || !std::isfinite(total_weight))
    {
        return nullptr;
    }

    auto compensated_weights = weights;
    auto average = static_cast<float>(total_weight / pixel_count);
    for (auto& weight : compensated_weights)
    {
        weight = std::max(weight - average, 0.0f);
    }
    if (std::all_of(
            compensated_weights.begin(), compensated_weights.end(),
            [](auto weight) noexcept { return weight == 0.0f; }))
    {
        std::fill(compensated_weights.begin(), compensated_weights.end(), 1.0f);
    }

    auto complete = create_alias_distribution_2d(weights, resolution);
    auto incomplete = create_alias_distribution_2d(compensated_weights, resolution);
    auto alias_distribution_stride = static_cast<uint>(complete.aliases.size());
    auto pdf_distribution_stride = static_cast<uint>(complete.pdfs.size());
    LUISA_ASSERT(
        incomplete.aliases.size() == alias_distribution_stride &&
            incomplete.pdfs.size() == pdf_distribution_stride,
        "Mismatched complete and incomplete lat-long environment distributions.");

    luisa::vector<AliasEntry> aliases(static_cast<size_t>(alias_distribution_stride) * 2u);
    luisa::vector<float> pdfs(static_cast<size_t>(pdf_distribution_stride) * 2u);
    std::copy(complete.aliases.begin(), complete.aliases.end(), aliases.begin());
    std::copy(incomplete.aliases.begin(), incomplete.aliases.end(),
              aliases.begin() + alias_distribution_stride);
    std::copy(complete.pdfs.begin(), complete.pdfs.end(), pdfs.begin());
    std::copy(incomplete.pdfs.begin(), incomplete.pdfs.end(),
              pdfs.begin() + pdf_distribution_stride);

    auto [alias_buffer, alias_buffer_id] = renderer.bindless_arena_buffer<AliasEntry>(aliases.size());
    auto [pdf_buffer, pdf_buffer_id] = renderer.bindless_arena_buffer<float>(pdfs.size());
    command_buffer << alias_buffer.copy_from(aliases.data())
                   << pdf_buffer.copy_from(pdfs.data())
                   << commit();

    return luisa::make_unique<Instance>(
        renderer, this, texture, resolution, alias_buffer_id, pdf_buffer_id,
        alias_distribution_stride, pdf_distribution_stride);
}

luisa::optional<luisa::string> LatLongEnvironmentSpec::validate() const noexcept
{
    if (!std::isfinite(_scale) || _scale < 0.0f)
    {
        return spec_validation_error("Lat-long environment scale must be finite and non-negative.");
    }
    constexpr auto epsilon = 1e-4f;
    for (auto c = 0u; c < 3u; c++)
    {
        for (auto r = 0u; r < 3u; r++)
        {
            if (!std::isfinite(_transform_to_world[c][r]))
            {
                return spec_validation_error("Lat-long environment transform contains a non-finite value.");
            }
        }
        auto length_squared = dot(_transform_to_world[c], _transform_to_world[c]);
        if (std::abs(length_squared - 1.0f) > epsilon)
        {
            return spec_validation_error("Lat-long environment transform must be orthogonal (scale is not supported).");
        }
    }
    if (std::abs(dot(_transform_to_world[0], _transform_to_world[1])) > epsilon ||
        std::abs(dot(_transform_to_world[0], _transform_to_world[2])) > epsilon ||
        std::abs(dot(_transform_to_world[1], _transform_to_world[2])) > epsilon)
    {
        return spec_validation_error("Lat-long environment transform must be orthogonal (shear is not supported).");
    }
    auto determinant = dot(
        _transform_to_world[0], cross(_transform_to_world[1], _transform_to_world[2]));
    if (std::abs(std::abs(determinant) - 1.0f) > 4.0f * epsilon)
    {
        return spec_validation_error("Lat-long environment transform determinant must be +1 or -1.");
    }
    return luisa::nullopt;
}

const Environment* LatLongEnvironmentSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Environment, LatLongEnvironment>(
        builder.resolve(_emission), _scale, _transform_to_world);
}

} // namespace Yutrel
