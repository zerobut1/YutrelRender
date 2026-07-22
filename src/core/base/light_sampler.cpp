#include "light_sampler.h"

#include "base/geometry.h"
#include "base/interaction.h"
#include "base/light.h"
#include "base/renderer.h"
#include "utils/sampling.h"

namespace Yutrel
{
luisa::unique_ptr<LightSampler> LightSampler::create(Renderer& renderer, CommandBuffer& command_buffer) noexcept
{
    return luisa::make_unique<LightSampler>(renderer, command_buffer);
}

LightSampler::LightSampler(Renderer& renderer, CommandBuffer& command_buffer) noexcept
    : m_renderer(renderer)
{
    auto light_instances = renderer.geometry()->light_instances();
    if (!light_instances.empty())
    {
        auto [view, buffer_id] = renderer.bindless_arena_buffer<Light::Handle>(light_instances.size());
        m_light_buffer_id      = buffer_id;
        command_buffer
            << view.copy_from(light_instances.data())
            << commit();
    }
}

LightSampler::Evaluation LightSampler::evaluate_hit(const Interaction& it, const Interaction& it_from, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    auto eval = Light::Evaluation::zero(swl.dimension());

    auto light_instances = renderer().geometry()->light_instances();
    if (light_instances.empty()) [[unlikely]]
    {
        LUISA_WARNING_WITH_LOCATION("No lights in scene.");
        return eval;
    };
    renderer().lights().dispatch(it.shape.light_tag(), [&](auto light) noexcept
    {
        auto closure = light->closure(swl, time);
        eval         = closure->evaluate(it, it_from);
    });
    auto n                = static_cast<float>(light_instances.size());
    auto area_probability = renderer().environment() == nullptr ? 1.0f : 0.5f;
    eval.pdf *= area_probability / n;
    return eval;
}

LightSampler::Evaluation LightSampler::evaluate_miss(
    Expr<float3> wi, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    if (renderer().environment() == nullptr)
    {
        return Evaluation::zero(swl.dimension());
    }
    constexpr auto allow_incomplete_pdf = true;
    auto eval                           = renderer().environment()->evaluate(
        wi,
        swl,
        time,
        allow_incomplete_pdf);
    auto environment_probability = renderer().lights().empty() ? 1.0f : 0.5f;
    eval.pdf *= environment_probability;
    return eval;
}

LightSampler::Sample LightSampler::sample(const Interaction& it_from, Expr<float> u_select, Expr<float2> u_light, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    if (!renderer().has_lighting())
    {
        return Sample::zero(swl.dimension());
    }

    Selection sel = select(u_select, time);
    return sample_selection(it_from, sel, u_light, swl, time);
}

LightSampler::Selection LightSampler::select(Expr<float> u, Expr<float> time) const noexcept
{
    LUISA_ASSERT(renderer().has_lighting(), "No lighting in scene.");
    if (renderer().environment() != nullptr)
    {
        auto light_instances = renderer().geometry()->light_instances();
        if (light_instances.empty())
        {
            return {.tag = selection_environment, .prob = 1.0f};
        }
        auto n                  = static_cast<float>(light_instances.size());
        auto select_environment = u < 0.5f;
        auto light_u            = clamp((u - 0.5f) * 2.0f, 0.0f, 1.0f);
        auto light_tag          = cast<uint>(clamp(light_u * n, 0.0f, n - 1.0f));
        return {
            .tag  = ite(select_environment, selection_environment, light_tag),
            .prob = ite(select_environment, 0.5f, 0.5f / n),
        };
    }
    auto n = static_cast<float>(renderer().geometry()->light_instances().size());

    return {.tag = cast<uint>(clamp(u * n, 0.0f, n - 1.0f)), .prob = 1.0f / n};
}

LightSampler::Sample LightSampler::sample_selection(const Interaction& it_from, const Selection& sel, Expr<float2> u, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    auto sample = Sample::zero(swl.dimension());
    if (renderer().environment() != nullptr)
    {
        if (renderer().geometry()->light_instances().empty())
        {
            return sample_environment(it_from, sel, u, swl, time);
        }
        $if(sel.tag == selection_environment)
        {
            sample = sample_environment(it_from, sel, u, swl, time);
        }
        $else
        {
            sample = sample_light(it_from, sel, u, swl, time);
        };
    }
    else if (!renderer().geometry()->light_instances().empty())
    {
        sample = sample_light(it_from, sel, u, swl, time);
    }
    return sample;
}

auto LightSampler::sample_area(Expr<float3> p_from, Expr<uint> tag, Expr<float2> u_in) const noexcept
{
    auto handle                = renderer().buffer<Light::Handle>(m_light_buffer_id).read(tag);
    auto light_inst            = renderer().geometry()->instance(handle.instance_id);
    auto light_to_world        = renderer().geometry()->instance_to_world(handle.instance_id);
    auto alias_table_buffer_id = light_inst.alias_table_buffer_id();
    auto [triangle_id, ux]     = sample_alias_table(renderer().buffer<AliasEntry>(alias_table_buffer_id), light_inst.triangle_count(), u_in.x);
    auto triangle              = renderer().geometry()->triangle(light_inst, triangle_id);
    auto uv                    = sample_uniform_triangle(make_float2(ux, u_in.y)).xy();
    auto attrib                = renderer().geometry()->shading_point(light_inst, triangle, uv, light_to_world);

    return luisa::make_shared<Interaction>(Interaction::from_surface(
        std::move(light_inst), attrib.pg, attrib.ng, attrib.uv, attrib.pg,
        Frame::make(attrib.ns, attrib.dpdu), handle.instance_id, triangle_id, attrib.area,
        // Match hit-case convention: front_face is true when the outgoing direction
        // (from light point to shading point) lies in the +ng hemisphere.
        dot(attrib.ng, p_from - attrib.pg) > 0.0f));
}

LightSampler::Sample LightSampler::sample_light(const Interaction& it_from, const Selection& sel, Expr<float2> u, const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    LUISA_ASSERT(!renderer().geometry()->light_instances().empty(), "No lights in the scene.");
    auto it   = sample_area(it_from.p_g, sel.tag, u);
    auto eval = Light::Evaluation::zero(swl.dimension());
    renderer().lights().dispatch(it->shape.light_tag(), [&](auto light) noexcept
    {
        auto closure = light->closure(swl, time);
        eval         = closure->evaluate(*it, it_from);
    });
    Light::Sample light_sample = {.eval = std::move(eval), .p = it->p_g};
    light_sample.eval.pdf *= sel.prob;

    return Sample::from_light(light_sample, it_from);
}

LightSampler::Sample LightSampler::sample_environment(
    const Interaction& it_from, const Selection& sel, Expr<float2> u,
    const SampledWavelengths& swl, Expr<float> time) const noexcept
{
    LUISA_ASSERT(renderer().environment() != nullptr, "No environment in the scene.");
    constexpr auto allow_incomplete_pdf = true;
    auto environment_sample             = renderer().environment()->sample(
        swl,
        time,
        u,
        allow_incomplete_pdf);
    environment_sample.eval.pdf *= sel.prob;
    return Sample::from_environment(environment_sample, it_from);
}

LightSampler::Sample LightSampler::Sample::zero(uint dimension) noexcept
{
    return Sample{.eval = Evaluation::zero(dimension), .shadow_ray = {}, .delta = false};
}

LightSampler::Sample LightSampler::Sample::from_light(const Light::Sample& s, const Interaction& it_from) noexcept
{
    return Sample{.eval = s.eval, .shadow_ray = it_from.spawn_ray_to(s.p), .delta = false};
}

LightSampler::Sample LightSampler::Sample::from_environment(
    const Environment::Sample& s, const Interaction& it_from) noexcept
{
    return Sample{.eval = s.eval, .shadow_ray = it_from.spawn_ray(s.wi), .delta = s.delta};
}

} // namespace Yutrel
