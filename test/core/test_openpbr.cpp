#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <luisa/luisa-compute.h>

#include "base/renderer.h"
#include "surfaces/openpbr.h"
#include "utils/rng.h"

#include "../data/openpbr_adobe_8a20d6f9.h"

using namespace luisa;
using namespace luisa::compute;
using namespace Yutrel;

namespace
{
using Inputs = OpenPBRReference::Inputs;

class TestTexture final : public Texture
{
public:
    class Instance final : public Texture::Instance
    {
    public:
        Instance(const Renderer& renderer, const TestTexture* texture) noexcept
            : Texture::Instance{renderer, texture} {}

        [[nodiscard]] Float4 evaluate(const Interaction&, Expr<float>) const noexcept override
        {
            return base<TestTexture>()->m_value;
        }

        [[nodiscard]] Spectrum::Decode evaluate_albedo_spectrum(
            const Interaction&, const SampledWavelengths& swl, Expr<float>) const noexcept override
        {
            Float4 encoded = base<TestTexture>()->m_value;
            auto value = clamp(encoded.xyz(), 0.0f, 1.0f);
            SampledSpectrum result{swl.dimension()};
            for (auto i = 0u; i < result.dimension(); i++)
            {
                result[i] = value[std::min(i, 2u)];
            }
            return {.value = result, .strength = 1.0f};
        }
    };

private:
    float4 m_value;

public:
    explicit TestTexture(float4 value) noexcept : m_value{value} {}

    [[nodiscard]] luisa::unique_ptr<Texture::Instance> build(
        Renderer& renderer, CommandBuffer&) const noexcept override
    {
        return luisa::make_unique<Instance>(renderer, this);
    }
};

struct TestMaterial
{
    std::array<luisa::unique_ptr<TestTexture>, 9u> textures;
    luisa::unique_ptr<OpenPBRSurface> surface;
    luisa::unique_ptr<Surface::Instance> instance;
};

[[nodiscard]] luisa::unique_ptr<TestTexture> scalar_texture(float v)
{
    return luisa::make_unique<TestTexture>(make_float4(v));
}

[[nodiscard]] luisa::unique_ptr<TestTexture> color_texture(float3 v)
{
    return luisa::make_unique<TestTexture>(make_float4(v, 1.0f));
}

[[nodiscard]] TestMaterial make_material(Renderer& renderer,
                                         CommandBuffer& command_buffer,
                                         const Inputs& inputs)
{
    TestMaterial material;
    material.textures[0u] = scalar_texture(inputs.base_weight);
    material.textures[1u] = color_texture(inputs.base_color);
    material.textures[2u] = scalar_texture(inputs.base_metalness);
    material.textures[3u] = scalar_texture(inputs.base_diffuse_roughness);
    material.textures[4u] = scalar_texture(inputs.specular_weight);
    material.textures[5u] = color_texture(inputs.specular_color);
    material.textures[6u] = scalar_texture(inputs.specular_roughness);
    material.textures[7u] = scalar_texture(inputs.specular_roughness_anisotropy);
    material.textures[8u] = scalar_texture(inputs.specular_ior);
    material.surface = luisa::make_unique<OpenPBRSurface>(
        material.textures[0u].get(), material.textures[1u].get(),
        material.textures[2u].get(), material.textures[3u].get(),
        material.textures[4u].get(), material.textures[5u].get(),
        material.textures[6u].get(), material.textures[7u].get(),
        material.textures[8u].get(), true);
    material.instance = material.surface->build(renderer, command_buffer);
    return material;
}

[[nodiscard]] Interaction make_interaction() noexcept
{
    return Interaction::from_surface(
        Shape::Handle::decode(make_uint4(0u)), make_float3(0.0f),
        make_float3(0.0f, 0.0f, 1.0f), make_float2(0.0f),
        make_float3(0.0f), Frame{}, 0u, 0u, 1.0f, true);
}

[[nodiscard]] SampledWavelengths make_wavelengths(uint dimension) noexcept
{
    SampledWavelengths swl{dimension};
    constexpr std::array wavelengths{602.785f, 539.285f, 445.772f, 500.0f};
    for (auto i = 0u; i < dimension; i++)
    {
        swl.set_lambda(i, wavelengths[i]);
        swl.set_pdf(i, 1.0f);
    }
    return swl;
}

[[nodiscard]] bool close_enough(float actual, float reference) noexcept
{
    return std::abs(actual - reference) <= 5.0e-6f + 5.0e-4f * std::abs(reference);
}

[[nodiscard]] bool compare_float(const char* label, size_t case_index,
                                 float actual, float reference) noexcept
{
    if (std::isfinite(actual) && actual >= 0.0f && close_enough(actual, reference))
    {
        return true;
    }
    std::fprintf(stderr, "golden[%zu] %s: actual=%g reference=%g\n",
                 case_index, label, actual, reference);
    return false;
}

[[nodiscard]] bool compare_vector(const char* label, size_t case_index,
                                  float3 actual, float3 reference,
                                  bool require_non_negative = true) noexcept
{
    auto valid = true;
    for (auto i = 0u; i < 3u; i++)
    {
        if (!std::isfinite(actual[i]) || (require_non_negative && actual[i] < 0.0f) ||
            !close_enough(actual[i], reference[i]))
        {
            valid = false;
        }
    }
    if (!valid)
    {
        std::fprintf(stderr,
                     "golden[%zu] %s: actual=(%g,%g,%g) reference=(%g,%g,%g)\n",
                     case_index, label, actual.x, actual.y, actual.z,
                     reference.x, reference.y, reference.z);
    }
    return valid;
}

[[nodiscard]] bool run_adobe_golden(Device& device, Stream& stream,
                                    Renderer& renderer,
                                    CommandBuffer& command_buffer)
{
    std::vector<TestMaterial> materials;
    materials.reserve(OpenPBRReference::cases.size());
    for (auto&& reference : OpenPBRReference::cases)
    {
        materials.emplace_back(make_material(renderer, command_buffer, reference.inputs));
    }
    if (renderer.bindless_array().dirty())
    {
        command_buffer << renderer.bindless_array().update();
    }
    command_buffer << synchronize();

    auto valid = true;
    for (auto case_index = 0u; case_index < materials.size(); case_index++)
    {
        auto reference = OpenPBRReference::cases[case_index];
        auto instance = materials[case_index].instance.get();
        Kernel1D kernel = [instance, reference](BufferFloat4 output) noexcept
        {
            auto it = make_interaction();
            auto swl = make_wavelengths(3u);
            auto closure = instance->create_closure(swl, 0.0f);
            auto wo = def(reference.wo);
            instance->populate_closure(closure.get(), it, wo, 1.0f);
            closure->pre_eval();
            auto eval = closure->evaluate(wo, def(reference.wi));
            auto sample = closure->sample(
                wo, reference.random.x,
                make_float2(reference.random.y, reference.random.z));
            output.write(0u, make_float4(eval.f[0u], eval.f[1u], eval.f[2u], eval.pdf));
            output.write(1u, make_float4(sample.wi, sample.eval.pdf));
            output.write(2u, make_float4(sample.eval.f[0u], sample.eval.f[1u],
                                         sample.eval.f[2u], sample.pdf_mis));
            output.write(3u, make_float4(sample.eta, cast<float>(sample.event),
                                         ite(sample.delta, 1.0f, 0.0f), 0.0f));
            closure->post_eval();
        };
        auto shader = device.compile(kernel);
        auto output = device.create_buffer<float4>(4u);
        std::array<float4, 4u> result{};
        stream << shader(output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        auto reference_index = static_cast<size_t>(case_index);
        valid &= compare_vector("evaluate.f", reference_index, result[0u].xyz(),
                                reference.eval_f_cos);
        valid &= compare_float("evaluate.pdf", reference_index, result[0u].w,
                               reference.eval_pdf);
        valid &= compare_vector("sample.wi", reference_index, result[1u].xyz(),
                                reference.sampled_wi, false);
        valid &= compare_vector("sample.eval.f", reference_index, result[2u].xyz(),
                                reference.sample_f_cos);
        valid &= compare_float("sample.eval.pdf", reference_index, result[1u].w,
                               reference.sample_pdf);
        valid &= compare_float("sample.pdf_mis", reference_index, result[2u].w,
                               reference.sample_pdf);
        if (!close_enough(result[3u].x, 1.0f) || result[3u].y != Surface::event_reflect ||
            result[3u].z != 0.0f)
        {
            std::fprintf(stderr, "golden[%zu] invalid sample metadata=(%g,%g,%g)\n",
                         reference_index, result[3u].x, result[3u].y, result[3u].z);
            valid = false;
        }
    }
    return valid;
}

struct FurnaceCase
{
    std::string name;
    Inputs inputs;
    float no_v;
};

[[nodiscard]] std::vector<FurnaceCase> make_furnace_cases()
{
    Inputs base{
        .base_weight = 1.0f,
        .base_color = make_float3(1.0f),
        .base_metalness = 0.0f,
        .base_diffuse_roughness = 0.0f,
        .specular_weight = 1.0f,
        .specular_color = make_float3(1.0f),
        .specular_roughness = 0.2f,
        .specular_roughness_anisotropy = 0.0f,
        .specular_ior = 1.5f,
    };
    std::vector<FurnaceCase> cases;
    auto add = [&](std::string name, Inputs inputs, float no_v = 1.0f)
    {
        cases.emplace_back(FurnaceCase{std::move(name), inputs, no_v});
    };
    for (auto value : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f})
    {
        auto inputs = base;
        inputs.base_metalness = value;
        add("metalness=" + std::to_string(value), inputs);
    }
    for (auto value : {0.0f, 0.05f, 0.2f, 0.5f, 1.0f})
    {
        auto inputs = base;
        inputs.specular_roughness = value;
        add("roughness=" + std::to_string(value), inputs);
    }
    for (auto value : {0.0f, 0.5f, 0.9f})
    {
        auto inputs = base;
        inputs.specular_roughness_anisotropy = value;
        add("anisotropy=" + std::to_string(value), inputs);
    }
    for (auto value : {1.0f, 1.5f, 2.5f})
    {
        auto inputs = base;
        inputs.specular_ior = value;
        add("ior=" + std::to_string(value), inputs);
    }
    for (auto value : {0.0f, 0.5f, 1.0f})
    {
        auto inputs = base;
        inputs.base_diffuse_roughness = value;
        add("diffuse_roughness=" + std::to_string(value), inputs);
    }
    for (auto value : {1.0f, 0.5f, 0.1f})
    {
        add("NoV=" + std::to_string(value), base, value);
    }
    return cases;
}

[[nodiscard]] bool furnace_channel_ok(float value) noexcept
{
    return std::isfinite(value) && value >= 0.98f && value <= 1.02f;
}

[[nodiscard]] bool run_white_furnace_case(Device& device, Stream& stream,
                                          const FurnaceCase& test_case)
{
    constexpr uint sample_count = 32768u;
    Renderer renderer{device};
    CommandBuffer command_buffer{stream};
    auto material = make_material(renderer, command_buffer, test_case.inputs);
    if (renderer.bindless_array().dirty())
    {
        command_buffer << renderer.bindless_array().update();
    }
    command_buffer << synchronize();

    auto instance = material.instance.get();
    auto no_v = test_case.no_v;
    Kernel1D kernel = [instance, no_v](BufferFloat4 output) noexcept
    {
        auto index = dispatch_x();
        auto random_float = [](Expr<uint> bits) noexcept
        {
            return min(cast<float>(bits) * 2.3283064365386963e-10f,
                       0.99999994f);
        };
        auto u_lobe = random_float(xxhash32(make_uint2(index, 0x7249f30du)));
        auto u = make_float2(
            random_float(xxhash32(make_uint2(index, 0xa341316cu))),
            random_float(xxhash32(make_uint2(index, 0xc8013ea4u))));
        auto z = (cast<float>(index) + 0.5f) / static_cast<float>(sample_count);
        auto phi = 2.0f * pi * fract(cast<float>(index) * 0.61803398875f);
        auto radius = sqrt(max(0.0f, 1.0f - sqr(z)));
        auto wi_uniform = make_float3(radius * cos(phi), radius * sin(phi), z);
        auto wo = make_float3(sqrt(max(0.0f, 1.0f - no_v * no_v)), 0.0f, no_v);
        auto it = make_interaction();

        auto rgb_swl = make_wavelengths(3u);
        auto rgb_closure = instance->create_closure(rgb_swl, 0.0f);
        instance->populate_closure(rgb_closure.get(), it, wo, 1.0f);
        rgb_closure->pre_eval();
        auto rgb_sample = rgb_closure->sample(wo, u_lobe, u);
        auto rgb_weight = ite(rgb_sample.eval.pdf > 0.0f,
                              rgb_sample.eval.f / rgb_sample.eval.pdf,
                              SampledSpectrum{3u});
        auto rgb_uniform = rgb_closure->evaluate(wo, wi_uniform).f * (2.0f * pi);
        output.write(index * 4u,
                     make_float4(rgb_weight[0u], rgb_weight[1u], rgb_weight[2u], 0.0f));
        output.write(index * 4u + 2u,
                     make_float4(rgb_uniform[0u], rgb_uniform[1u], rgb_uniform[2u], 0.0f));
        rgb_closure->post_eval();

        auto hero_swl = make_wavelengths(4u);
        auto hero_closure = instance->create_closure(hero_swl, 0.0f);
        instance->populate_closure(hero_closure.get(), it, wo, 1.0f);
        hero_closure->pre_eval();
        auto hero_sample = hero_closure->sample(wo, u_lobe, u);
        auto hero_weight = ite(hero_sample.eval.pdf > 0.0f,
                               hero_sample.eval.f / hero_sample.eval.pdf,
                               SampledSpectrum{4u});
        auto hero_uniform = hero_closure->evaluate(wo, wi_uniform).f * (2.0f * pi);
        output.write(index * 4u + 1u,
                     make_float4(hero_weight[0u], hero_weight[1u],
                                 hero_weight[2u], hero_weight[3u]));
        output.write(index * 4u + 3u,
                     make_float4(hero_uniform[0u], hero_uniform[1u],
                                 hero_uniform[2u], hero_uniform[3u]));
        hero_closure->post_eval();
    };

    auto shader = device.compile(kernel);
    auto output = device.create_buffer<float4>(sample_count * 4u);
    std::vector<float4> samples(sample_count * 4u);
    stream << shader(output).dispatch(sample_count)
           << output.copy_to(luisa::span{samples})
           << synchronize();

    std::array<double, 3u> rgb{};
    std::array<double, 4u> hero{};
    std::array<double, 3u> rgb_uniform{};
    std::array<double, 4u> hero_uniform{};
    for (auto i = 0u; i < sample_count; i++)
    {
        for (auto channel = 0u; channel < 3u; channel++)
        {
            rgb[channel] += samples[i * 4u][channel];
            rgb_uniform[channel] += samples[i * 4u + 2u][channel];
        }
        for (auto channel = 0u; channel < 4u; channel++)
        {
            hero[channel] += samples[i * 4u + 1u][channel];
            hero_uniform[channel] += samples[i * 4u + 3u][channel];
        }
    }
    auto valid = true;
    for (auto channel = 0u; channel < 3u; channel++)
    {
        auto value = static_cast<float>(rgb[channel] / sample_count);
        if (!furnace_channel_ok(value))
        {
            std::fprintf(stderr, "furnace %s RGB[%u]=%g\n",
                         test_case.name.c_str(), channel, value);
            valid = false;
        }
    }
    for (auto channel = 0u; channel < 4u; channel++)
    {
        auto value = static_cast<float>(hero[channel] / sample_count);
        if (!furnace_channel_ok(value))
        {
            std::fprintf(stderr, "furnace %s Hero[%u]=%g\n",
                         test_case.name.c_str(), channel, value);
            valid = false;
        }
    }
    if (test_case.inputs.specular_roughness >= 0.2f)
    {
        for (auto channel = 0u; channel < 3u; channel++)
        {
            auto sampled = static_cast<float>(rgb[channel] / sample_count);
            auto integrated = static_cast<float>(rgb_uniform[channel] / sample_count);
            if (!std::isfinite(integrated) || std::abs(sampled - integrated) > 0.04f)
            {
                std::fprintf(stderr, "furnace %s RGB[%u] sample=%g uniform=%g\n",
                             test_case.name.c_str(), channel, sampled, integrated);
                valid = false;
            }
        }
        for (auto channel = 0u; channel < 4u; channel++)
        {
            auto sampled = static_cast<float>(hero[channel] / sample_count);
            auto integrated = static_cast<float>(hero_uniform[channel] / sample_count);
            if (!std::isfinite(integrated) || std::abs(sampled - integrated) > 0.04f)
            {
                std::fprintf(stderr, "furnace %s Hero[%u] sample=%g uniform=%g\n",
                             test_case.name.c_str(), channel, sampled, integrated);
                valid = false;
            }
        }
    }
    return valid;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) { return 0; }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    Renderer golden_renderer{device};
    CommandBuffer golden_commands{stream};
    auto valid = run_adobe_golden(device, stream, golden_renderer, golden_commands);

    for (auto&& test_case : make_furnace_cases())
    {
        valid = run_white_furnace_case(device, stream, test_case) && valid;
    }
    return valid ? 0 : 1;
}
