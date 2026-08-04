#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <glm/glm.hpp>

#include "openpbr.h"

namespace
{
struct CaseInput
{
    float base_weight;
    glm::vec3 base_color;
    float base_metalness;
    float base_diffuse_roughness;
    float specular_weight;
    glm::vec3 specular_color;
    float specular_roughness;
    float specular_roughness_anisotropy;
    float specular_ior;
};

struct Case
{
    CaseInput inputs;
    glm::vec3 wo;
    glm::vec3 wi;
    glm::vec3 random;
};

[[nodiscard]] OpenPBR_ResolvedInputs resolve(const CaseInput& in)
{
    auto r = openpbr_make_default_resolved_inputs();
    r.base_weight = in.base_weight;
    r.base_color = in.base_color;
    r.base_metalness = in.base_metalness;
    r.base_diffuse_roughness = in.base_diffuse_roughness;
    r.specular_weight = in.specular_weight;
    r.specular_color = in.specular_color;
    r.specular_roughness = in.specular_roughness;
    r.specular_roughness_anisotropy = in.specular_roughness_anisotropy;
    r.specular_ior = in.specular_ior;
    return r;
}

[[nodiscard]] std::string float_literal(float v)
{
    std::ostringstream out;
    out << std::setprecision(9) << std::defaultfloat << v;
    auto text = out.str();
    if (text.find('.') == std::string::npos && text.find('e') == std::string::npos &&
        text.find('E') == std::string::npos) {
        text += ".0";
    }
    return text + "f";
}

[[nodiscard]] std::string vec3_literal(glm::vec3 v)
{
    return "luisa::make_float3(" + float_literal(v.x) + ", " +
           float_literal(v.y) + ", " + float_literal(v.z) + ")";
}

[[nodiscard]] std::string git_head(const std::filesystem::path& dir)
{
    std::ifstream head_file{dir / ".git" / "HEAD"};
    if (!head_file) { return {}; }
    std::string head;
    std::getline(head_file, head);
    if (head.starts_with("ref: ")) {
        std::ifstream ref_file{dir / ".git" / head.substr(5)};
        if (!ref_file) { return {}; }
        std::getline(ref_file, head);
    }
    // Detached HEAD repositories store the object id directly in HEAD.
    if (head.size() != 40u) { return {}; }
    auto is_hex = [](char c) noexcept {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
    };
    for (auto c : head) {
        if (!is_hex(c)) { return {}; }
    }
    return head;
}
} // namespace

int main()
{
    const std::filesystem::path reference_dir{YUTREL_OPENPBR_REFERENCE_DIR};
    constexpr auto expected_head = "8a20d6f9b695b65ec2ca6defef642a9969992244";
    auto actual_head = git_head(reference_dir);
    if (actual_head != expected_head)
    {
        throw std::runtime_error("openpbr-bsdf HEAD must be 8a20d6f9b695b65ec2ca6defef642a9969992244");
    }

    const std::array cases{
        Case{CaseInput{1.0f, glm::vec3{0.8f}, 0.0f, 0.0f, 1.0f, glm::vec3{1.0f}, 0.3f, 0.0f, 1.5f},
             glm::normalize(glm::vec3{0.0f, 0.0f, 1.0f}), glm::normalize(glm::vec3{0.3f, 0.1f, 0.9486833f}), glm::vec3{0.13f, 0.71f, 0.37f}},
        Case{CaseInput{1.0f, glm::vec3{0.8f, 0.3f, 0.1f}, 0.25f, 0.5f, 0.8f, glm::vec3{0.9f, 0.95f, 1.0f}, 0.05f, 0.5f, 2.5f},
             glm::normalize(glm::vec3{0.6f, -0.2f, 0.77f}), glm::normalize(glm::vec3{-0.4f, 0.5f, 0.76f}), glm::vec3{0.47f, 0.23f, 0.81f}},
        Case{CaseInput{0.7f, glm::vec3{0.2f, 0.7f, 0.9f}, 1.0f, 0.0f, 1.0f, glm::vec3{0.8f, 0.7f, 0.6f}, 0.2f, 0.9f, 1.5f},
             glm::normalize(glm::vec3{0.1f, 0.7f, 0.7f}), glm::normalize(glm::vec3{-0.2f, -0.4f, 0.89f}), glm::vec3{0.89f, 0.42f, 0.17f}},
        Case{CaseInput{1.0f, glm::vec3{1.0f}, 0.0f, 1.0f, 1.0f, glm::vec3{1.0f}, 1.0f, 0.0f, 1.5f},
             glm::normalize(glm::vec3{0.9949874f, 0.0f, 0.1f}), glm::normalize(glm::vec3{0.2f, -0.4f, 0.8944272f}), glm::vec3{0.31f, 0.67f, 0.53f}},
        Case{CaseInput{1.0f, glm::vec3{0.9f, 0.6f, 0.2f}, 0.0f, 0.2f, 1.0f, glm::vec3{0.7f, 0.8f, 1.0f}, 0.5f, 0.0f, 1.0f},
             glm::normalize(glm::vec3{0.4f, 0.2f, 0.8944272f}), glm::normalize(glm::vec3{-0.6f, 0.1f, 0.7937254f}), glm::vec3{0.73f, 0.11f, 0.59f}},
        Case{CaseInput{0.9f, glm::vec3{0.3f, 0.8f, 0.4f}, 0.5f, 0.75f, 0.2f, glm::vec3{1.0f, 0.8f, 0.6f}, 0.35f, 0.5f, 1.8f},
             glm::normalize(glm::vec3{-0.5f, 0.3f, 0.8124038f}), glm::normalize(glm::vec3{0.45f, 0.2f, 0.870344f}), glm::vec3{0.21f, 0.93f, 0.41f}},
        Case{CaseInput{0.4f, glm::vec3{0.95f, 0.4f, 0.05f}, 0.75f, 0.1f, 1.4f, glm::vec3{0.6f, 0.9f, 0.7f}, 0.05f, 0.0f, 2.5f},
             glm::normalize(glm::vec3{0.25f, -0.6f, 0.7599342f}), glm::normalize(glm::vec3{-0.3f, 0.2f, 0.9327379f}), glm::vec3{0.58f, 0.36f, 0.77f}},
    };

    std::filesystem::create_directories("test/data");
    std::ofstream out{"test/data/openpbr_adobe_8a20d6f9.h", std::ios::trunc};
    if (!out) { throw std::runtime_error("failed to open golden output"); }
    out << "#pragma once\n#include <array>\n#include <luisa/core/basic_types.h>\n\nnamespace Yutrel::OpenPBRReference {\n";
    out << "struct Inputs { float base_weight; luisa::float3 base_color; float base_metalness; float base_diffuse_roughness; float specular_weight; luisa::float3 specular_color; float specular_roughness; float specular_roughness_anisotropy; float specular_ior; };\n";
    out << "struct ReferenceCase { Inputs inputs; luisa::float3 wo; luisa::float3 wi; luisa::float3 random; luisa::float3 eval_f_cos; float eval_pdf; luisa::float3 sampled_wi; luisa::float3 sample_f_cos; float sample_pdf; };\n";
    out << "inline constexpr std::array cases{\n";
    for (auto&& c : cases)
    {
        auto prepared = openpbr_prepare(resolve(c.inputs), glm::vec3{1.0f},
                                        OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, c.wo);
        auto eval = openpbr_eval(prepared, c.wi);
        auto eval_f_cos = openpbr_get_sum_of_diffuse_specular(eval);
        auto pdf = openpbr_pdf(prepared, c.wi);
        glm::vec3 sampled_wi{}, weight{};
        float sample_pdf = 0.0f;
        OpenPBR_DiffuseSpecular sample_weight{};
        OpenPBR_BsdfLobeType sampled_type{};
        openpbr_sample(prepared, c.random, sampled_wi, sample_weight, sample_pdf, sampled_type);
        weight = openpbr_get_sum_of_diffuse_specular(sample_weight);
        auto sample_f_cos = weight * sample_pdf;
        out << "    ReferenceCase{Inputs{" << float_literal(c.inputs.base_weight) << ", " << vec3_literal(c.inputs.base_color)
            << ", " << float_literal(c.inputs.base_metalness) << ", " << float_literal(c.inputs.base_diffuse_roughness) << ", "
            << float_literal(c.inputs.specular_weight) << ", " << vec3_literal(c.inputs.specular_color) << ", "
            << float_literal(c.inputs.specular_roughness) << ", " << float_literal(c.inputs.specular_roughness_anisotropy) << ", "
            << float_literal(c.inputs.specular_ior) << "}, " << vec3_literal(c.wo) << ", " << vec3_literal(c.wi) << ", "
            << vec3_literal(c.random) << ", " << vec3_literal(eval_f_cos) << ", " << float_literal(pdf) << ", "
            << vec3_literal(sampled_wi) << ", " << vec3_literal(sample_f_cos) << ", " << float_literal(sample_pdf) << "},\n";
    }
    out << "};\n}\n";
}
