#include "pbrt_parser.h"

#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

namespace Yutrel
{
namespace
{

enum class TokenKind
{
    Word,
    String,
    LBracket,
    RBracket,
    End,
};

struct Token
{
    TokenKind kind{TokenKind::End};
    luisa::string text;
    SourceLocation loc;
};

[[noreturn]] void fail(const SourceLocation& loc, luisa::string_view message)
{
    auto s = luisa::format("{}: {}", format_source_location(loc), message);
    throw std::runtime_error{s.c_str()};
}

[[noreturn]] void fail(const Token& token, luisa::string_view message)
{
    fail(token.loc, message);
}

[[nodiscard]] bool is_space(char c) noexcept
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

class Tokenizer
{
private:
    std::filesystem::path m_path;
    luisa::string m_source;
    size_t m_pos{};
    uint m_line{1u};
    uint m_column{1u};

public:
    Tokenizer(std::filesystem::path path, luisa::string source) noexcept
        : m_path{std::move(path)}, m_source{std::move(source)} {}

    [[nodiscard]] luisa::vector<Token> tokenize()
    {
        luisa::vector<Token> tokens;
        while (true)
        {
            auto token = next();
            tokens.emplace_back(token);
            if (token.kind == TokenKind::End)
            {
                break;
            }
        }
        return tokens;
    }

private:
    [[nodiscard]] SourceLocation loc() const
    {
        return SourceLocation{m_path, m_line, m_column};
    }

    [[nodiscard]] char peek() const noexcept
    {
        return m_pos < m_source.size() ? m_source[m_pos] : '\0';
    }

    [[nodiscard]] char get() noexcept
    {
        auto c = peek();
        if (c == '\0')
        {
            return c;
        }
        m_pos++;
        if (c == '\n')
        {
            m_line++;
            m_column = 1u;
        }
        else
        {
            m_column++;
        }
        return c;
    }

    void skip_trivia() noexcept
    {
        while (true)
        {
            while (is_space(peek()))
            {
                (void)get();
            }
            if (peek() != '#')
            {
                break;
            }
            while (peek() != '\0' && peek() != '\n')
            {
                (void)get();
            }
        }
    }

    [[nodiscard]] Token next()
    {
        skip_trivia();
        auto start = loc();
        auto c     = peek();
        if (c == '\0')
        {
            return Token{TokenKind::End, {}, start};
        }
        if (c == '[')
        {
            (void)get();
            return Token{TokenKind::LBracket, "[", start};
        }
        if (c == ']')
        {
            (void)get();
            return Token{TokenKind::RBracket, "]", start};
        }
        if (c == '"')
        {
            (void)get();
            luisa::string text;
            while (true)
            {
                c = get();
                if (c == '\0' || c == '\n')
                {
                    fail(start, "unterminated string literal");
                }
                if (c == '"')
                {
                    break;
                }
                if (c == '\\')
                {
                    auto escaped = get();
                    switch (escaped)
                    {
                    case '\\':
                    case '"':
                        text.push_back(escaped);
                        break;
                    case 'n':
                        text.push_back('\n');
                        break;
                    case 't':
                        text.push_back('\t');
                        break;
                    default:
                        fail(start, luisa::format("unsupported string escape '\\{}'", escaped));
                    }
                }
                else
                {
                    text.push_back(c);
                }
            }
            return Token{TokenKind::String, std::move(text), start};
        }

        luisa::string text;
        while (true)
        {
            c = peek();
            if (c == '\0' || is_space(c) || c == '[' || c == ']' || c == '"' || c == '#')
            {
                break;
            }
            text.push_back(get());
        }
        return Token{TokenKind::Word, std::move(text), start};
    }
};

[[nodiscard]] float& matrix_at(Matrix4& m, uint32_t row, uint32_t column) noexcept { return m[row * 4u + column]; }
[[nodiscard]] float matrix_at(const Matrix4& m, uint32_t row, uint32_t column) noexcept { return m[row * 4u + column]; }

[[nodiscard]] Matrix4 multiply(const Matrix4& lhs, const Matrix4& rhs) noexcept
{
    Matrix4 result{};
    for (auto row = 0u; row < 4u; row++)
    {
        for (auto column = 0u; column < 4u; column++)
        {
            for (auto i = 0u; i < 4u; i++)
            {
                matrix_at(result, row, column) += matrix_at(lhs, row, i) * matrix_at(rhs, i, column);
            }
        }
    }
    return result;
}

[[nodiscard]] float dot_host(float3 a, float3 b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
[[nodiscard]] float3 cross_host(float3 a, float3 b) noexcept { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

[[nodiscard]] float3 normalize_host(float3 v, const Token& command, luisa::string_view what)
{
    auto length_squared = dot_host(v, v);
    if (length_squared < 1e-16f)
    {
        fail(command, luisa::format("{} must be non-zero", what));
    }
    auto inv_length = 1.0f / std::sqrt(length_squared);
    return v * inv_length;
}

class Parser
{
private:
    PbrtScene m_desc;
    luisa::vector<Token> m_tokens;
    size_t m_cursor{};

    enum class Block
    {
        Options,
        World,
    };

    struct AttributeState
    {
        MaterialBinding material;
        luisa::optional<AreaLightDesc> area_light;
        MediumInterfaceDesc medium_interface;
        Matrix4 transform;
    };

    Block m_block{Block::Options};
    Matrix4 m_current_transform{identity_matrix4};
    MaterialBinding m_current_material;
    luisa::optional<AreaLightDesc> m_current_area_light;
    MediumInterfaceDesc m_current_medium_interface;
    luisa::vector<AttributeState> m_attribute_stack;

public:
    Parser(PbrtScene desc, luisa::vector<Token> tokens) noexcept
        : m_desc{std::move(desc)}, m_tokens{std::move(tokens)} {}

    [[nodiscard]] PbrtScene parse()
    {
        while (!peek(TokenKind::End))
        {
            auto command = expect(TokenKind::Word, "expected PBRT command");
            parse_command(command);
        }
        if (m_block != Block::World)
        {
            fail(m_tokens[m_cursor], "End of file before WorldBegin");
        }
        if (!m_attribute_stack.empty())
        {
            fail(m_tokens[m_cursor], "missing AttributeEnd");
        }
        return std::move(m_desc);
    }

private:
    [[nodiscard]] const Token& current() const noexcept
    {
        return m_tokens[m_cursor];
    }

    [[nodiscard]] bool peek(TokenKind kind) const noexcept
    {
        return current().kind == kind;
    }

    [[nodiscard]] Token advance() noexcept
    {
        return m_tokens[m_cursor++];
    }

    [[nodiscard]] Token expect(TokenKind kind, luisa::string_view message)
    {
        if (!peek(kind))
        {
            fail(current(), message);
        }
        return advance();
    }

    [[nodiscard]] luisa::string expect_string(luisa::string_view context)
    {
        return expect(TokenKind::String, luisa::format("expected quoted string for {}", context)).text;
    }

    void expect_options(const Token& command)
    {
        if (m_block != Block::Options)
        {
            fail(command, luisa::format("'{}' is only supported before WorldBegin", command.text));
        }
    }

    void expect_world(const Token& command)
    {
        if (m_block != Block::World)
        {
            fail(command, luisa::format("'{}' is only supported after WorldBegin", command.text));
        }
    }

    [[nodiscard]] luisa::vector<RawParameter> parse_parameters()
    {
        luisa::vector<RawParameter> params;
        while (peek(TokenKind::String))
        {
            auto decl  = advance();
            auto split = decl.text.find_first_of(" \t");
            if (split == luisa::string::npos)
            {
                fail(decl, luisa::format("invalid parameter declaration '{}'", decl.text));
            }
            auto type     = decl.text.substr(0u, split);
            auto name_pos = decl.text.find_first_not_of(" \t", split);
            if (name_pos == luisa::string::npos)
            {
                fail(decl, luisa::format("missing parameter name in '{}'", decl.text));
            }
            auto name = decl.text.substr(name_pos);
            luisa::vector<RawValue> values;
            auto bracketed = peek(TokenKind::LBracket);
            if (bracketed)
            {
                (void)advance();
                while (!peek(TokenKind::RBracket))
                {
                    if (peek(TokenKind::End))
                    {
                        fail(current(), luisa::format("unterminated value list for parameter '{}'", decl.text));
                    }
                    if (peek(TokenKind::LBracket))
                    {
                        fail(current(), luisa::format("unexpected '[' inside parameter '{}'", decl.text));
                    }
                    auto value = advance();
                    values.emplace_back(RawValue{.source = value.loc, .text = std::move(value.text), .quoted = value.kind == TokenKind::String});
                }
                (void)expect(TokenKind::RBracket, "expected ']'");
            }
            else
            {
                if (!peek(TokenKind::Word) && !peek(TokenKind::String))
                {
                    fail(current(), luisa::format("expected value for parameter '{}'", decl.text));
                }
                auto value = advance();
                values.emplace_back(RawValue{.source = value.loc, .text = std::move(value.text), .quoted = value.kind == TokenKind::String});
            }
            params.emplace_back(RawParameter{
                .source    = decl.loc,
                .type      = std::move(type),
                .name      = std::move(name),
                .values    = std::move(values),
                .bracketed = bracketed,
            });
        }
        return params;
    }

    [[nodiscard]] const RawParameter* find_param(luisa::span<const RawParameter> params, luisa::string_view type, luisa::string_view name) const noexcept
    {
        for (auto&& p : params)
        {
            if (p.type == type && p.name == name)
            {
                return &p;
            }
        }
        return nullptr;
    }

    [[nodiscard]] const RawParameter& require_param(luisa::span<const RawParameter> params, luisa::string_view type, luisa::string_view name, const Token& command) const
    {
        if (auto p = find_param(params, type, name))
        {
            return *p;
        }
        fail(command, luisa::format("missing parameter '\"{} {}\"'", type, name));
    }

    [[nodiscard]] float parse_float_token(const RawValue& token) const
    {
        if (token.quoted)
        {
            fail(token.source, luisa::format("expected float, got string '{}'", token.text));
        }
        try
        {
            size_t parsed_chars = 0u;
            auto v              = std::stof(std::string{token.text}, &parsed_chars);
            if (parsed_chars != token.text.size())
            {
                fail(token.source, luisa::format("invalid float '{}'", token.text));
            }
            return v;
        }
        catch (const std::exception&)
        {
            fail(token.source, luisa::format("invalid float '{}'", token.text));
        }
    }

    [[nodiscard]] float parse_float_token(const Token& token) const
    {
        return parse_float_token(RawValue{.source = token.loc, .text = token.text, .quoted = token.kind == TokenKind::String});
    }

    [[nodiscard]] float next_float(const Token& command, luisa::string_view context)
    {
        if (!peek(TokenKind::Word))
        {
            fail(command, luisa::format("{} expects numeric arguments", context));
        }
        return parse_float_token(advance());
    }

    [[nodiscard]] float3 next_float3(const Token& command, luisa::string_view context)
    {
        auto x = next_float(command, context);
        auto y = next_float(command, context);
        auto z = next_float(command, context);
        return make_float3(x, y, z);
    }

    void concat_transform(const Matrix4& transform) noexcept
    {
        m_current_transform = multiply(m_current_transform, transform);
    }

    [[nodiscard]] int parse_int_token(const RawValue& token) const
    {
        if (token.quoted)
        {
            fail(token.source, luisa::format("expected integer, got string '{}'", token.text));
        }
        int value   = 0;
        auto begin  = token.text.data();
        auto end    = begin + token.text.size();
        auto result = std::from_chars(begin, end, value);
        if (result.ec != std::errc{} || result.ptr != end)
        {
            fail(token.source, luisa::format("invalid integer '{}'", token.text));
        }
        return value;
    }

    [[nodiscard]] luisa::string parse_string_token(const RawValue& token) const
    {
        if (!token.quoted)
        {
            fail(token.source, luisa::format("expected string, got '{}'", token.text));
        }
        return token.text;
    }

    [[nodiscard]] uint one_uint(luisa::span<const RawParameter> params, luisa::string_view name, const Token& command, uint default_value) const
    {
        auto p = find_param(params, "integer", name);
        if (p == nullptr)
        {
            return default_value;
        }
        if (p->values.size() != 1u)
        {
            fail(p->source, luisa::format("'integer {}' expects exactly one value", name));
        }
        auto v = parse_int_token(p->values.front());
        if (v < 0)
        {
            fail(p->values.front().source, luisa::format("'integer {}' must be non-negative", name));
        }
        return static_cast<uint>(v);
    }

    [[nodiscard]] float one_float(luisa::span<const RawParameter> params, luisa::string_view name, const Token& command, float default_value) const
    {
        auto p = find_param(params, "float", name);
        if (p == nullptr)
        {
            return default_value;
        }
        if (p->values.size() != 1u)
        {
            fail(p->source, luisa::format("'float {}' expects exactly one value", name));
        }
        return parse_float_token(p->values.front());
    }

    [[nodiscard]] bool one_bool(luisa::span<const RawParameter> params, luisa::string_view name,
                                const Token& command, bool default_value) const
    {
        auto p = find_param(params, "bool", name);
        if (p == nullptr)
        {
            return default_value;
        }
        if (p->values.size() != 1u)
        {
            fail(p->source, luisa::format("'bool {}' expects exactly one value", name));
        }
        auto&& value = p->values.front();
        if (value.text == "true")
        {
            return true;
        }
        if (value.text == "false")
        {
            return false;
        }
        fail(value.source, luisa::format("'bool {}' expects 'true' or 'false'", name));
    }

    [[nodiscard]] float3 one_float3(
        luisa::span<const RawParameter> params, luisa::string_view type,
        luisa::string_view name, float3 default_value) const
    {
        auto p = find_param(params, type, name);
        if (p == nullptr)
        {
            return default_value;
        }
        if (p->values.size() != 3u)
        {
            fail(p->source, luisa::format("'{} {}' expects exactly three values", type, name));
        }
        return make_float3(
            parse_float_token(p->values[0u]),
            parse_float_token(p->values[1u]),
            parse_float_token(p->values[2u]));
    }

    [[nodiscard]] luisa::string one_string(luisa::span<const RawParameter> params, luisa::string_view name, const Token& command, luisa::string default_value) const
    {
        auto p = find_param(params, "string", name);
        if (p == nullptr)
        {
            return default_value;
        }
        if (p->values.size() != 1u)
        {
            fail(p->source, luisa::format("'string {}' expects exactly one value", name));
        }
        return parse_string_token(p->values.front());
    }

    [[nodiscard]] luisa::optional<luisa::string> optional_texture(
        luisa::span<const RawParameter> params, luisa::string_view name) const
    {
        auto p = find_param(params, "texture", name);
        if (p == nullptr)
        {
            return luisa::nullopt;
        }
        if (p->values.size() != 1u)
        {
            fail(p->source, luisa::format("'texture {}' expects exactly one value", name));
        }
        return parse_string_token(p->values.front());
    }

    [[nodiscard]] float3 rgb(luisa::span<const RawParameter> params, luisa::string_view name, const Token& command) const
    {
        auto&& p = require_param(params, "rgb", name, command);
        if (p.values.size() != 3u)
        {
            fail(p.source, luisa::format("'rgb {}' expects exactly three values", name));
        }
        return make_float3(parse_float_token(p.values[0u]),
                           parse_float_token(p.values[1u]),
                           parse_float_token(p.values[2u]));
    }

    [[nodiscard]] luisa::vector<float3> float3_array(luisa::span<const RawParameter> params, luisa::string_view type, luisa::string_view name, const Token& command) const
    {
        auto&& p = require_param(params, type, name, command);
        if (p.values.size() % 3u != 0u)
        {
            fail(p.source, luisa::format("'{} {}' value count must be a multiple of 3", type, name));
        }
        luisa::vector<float3> values;
        values.reserve(p.values.size() / 3u);
        for (auto i = 0u; i < p.values.size(); i += 3u)
        {
            values.emplace_back(make_float3(parse_float_token(p.values[i]),
                                            parse_float_token(p.values[i + 1u]),
                                            parse_float_token(p.values[i + 2u])));
        }
        return values;
    }

    [[nodiscard]] luisa::vector<float3> optional_float3_array(luisa::span<const RawParameter> params, luisa::string_view type, luisa::string_view name) const
    {
        auto p = find_param(params, type, name);
        if (p == nullptr)
        {
            return {};
        }
        if (p->values.size() % 3u != 0u)
        {
            fail(p->source, luisa::format("'{} {}' value count must be a multiple of 3", type, name));
        }
        luisa::vector<float3> values;
        values.reserve(p->values.size() / 3u);
        for (auto i = 0u; i < p->values.size(); i += 3u)
        {
            values.emplace_back(make_float3(parse_float_token(p->values[i]),
                                            parse_float_token(p->values[i + 1u]),
                                            parse_float_token(p->values[i + 2u])));
        }
        return values;
    }

    [[nodiscard]] luisa::vector<float2> optional_float2_array(luisa::span<const RawParameter> params, luisa::string_view type, luisa::string_view name) const
    {
        auto p = find_param(params, type, name);
        if (p == nullptr)
        {
            return {};
        }
        if (p->values.size() % 2u != 0u)
        {
            fail(p->source, luisa::format("'{} {}' value count must be a multiple of 2", type, name));
        }
        luisa::vector<float2> values;
        values.reserve(p->values.size() / 2u);
        for (auto i = 0u; i < p->values.size(); i += 2u)
        {
            values.emplace_back(make_float2(parse_float_token(p->values[i]),
                                            parse_float_token(p->values[i + 1u])));
        }
        return values;
    }

    [[nodiscard]] luisa::vector<uint3> triangle_indices(luisa::span<const RawParameter> params, const Token& command, size_t vertex_count) const
    {
        auto&& p = require_param(params, "integer", "indices", command);
        if (p.values.size() % 3u != 0u)
        {
            fail(p.source, "'integer indices' value count must be a multiple of 3");
        }
        luisa::vector<uint3> values;
        values.reserve(p.values.size() / 3u);
        for (auto i = 0u; i < p.values.size(); i += 3u)
        {
            auto i0 = parse_int_token(p.values[i]);
            auto i1 = parse_int_token(p.values[i + 1u]);
            auto i2 = parse_int_token(p.values[i + 2u]);
            if (i0 < 0 || i1 < 0 || i2 < 0 ||
                static_cast<size_t>(i0) >= vertex_count ||
                static_cast<size_t>(i1) >= vertex_count ||
                static_cast<size_t>(i2) >= vertex_count)
            {
                fail(p.values[i].source, "triangle index out of bounds");
            }
            values.emplace_back(make_uint3(static_cast<uint>(i0),
                                           static_cast<uint>(i1),
                                           static_cast<uint>(i2)));
        }
        return values;
    }

    void parse_command(const Token& command)
    {
        if (command.text == "Integrator")
        {
            parse_integrator(command);
        }
        else if (command.text == "Transform")
        {
            parse_transform(command);
        }
        else if (command.text == "Scale")
        {
            parse_scale(command);
        }
        else if (command.text == "Translate")
        {
            parse_translate(command);
        }
        else if (command.text == "Rotate")
        {
            parse_rotate(command);
        }
        else if (command.text == "LookAt")
        {
            parse_look_at(command);
        }
        else if (command.text == "Sampler")
        {
            parse_sampler(command);
        }
        else if (command.text == "PixelFilter")
        {
            parse_filter(command);
        }
        else if (command.text == "Film")
        {
            parse_film(command);
        }
        else if (command.text == "Camera")
        {
            parse_camera(command);
        }
        else if (command.text == "WorldBegin")
        {
            parse_world_begin(command);
        }
        else if (command.text == "MakeNamedMaterial")
        {
            parse_make_named_material(command);
        }
        else if (command.text == "MakeNamedMedium")
        {
            parse_make_named_medium(command);
        }
        else if (command.text == "MediumInterface")
        {
            parse_medium_interface(command);
        }
        else if (command.text == "Material")
        {
            parse_material(command);
        }
        else if (command.text == "Texture")
        {
            parse_texture(command);
        }
        else if (command.text == "NamedMaterial")
        {
            parse_named_material(command);
        }
        else if (command.text == "Shape")
        {
            parse_shape(command);
        }
        else if (command.text == "AttributeBegin")
        {
            parse_attribute_begin(command);
        }
        else if (command.text == "AttributeEnd")
        {
            parse_attribute_end(command);
        }
        else if (command.text == "AreaLightSource")
        {
            parse_area_light_source(command);
        }
        else if (command.text == "LightSource")
        {
            parse_light_source(command);
        }
        else
        {
            fail(command, luisa::format("unsupported PBRT command '{}'", command.text));
        }
    }

    void parse_integrator(const Token& command)
    {
        expect_options(command);
        auto type = expect_string("Integrator type");
        if (type == "path")
        {
            m_desc.integrator.type = IntegratorDesc::Type::Path;
        }
        else if (type == "volpath")
        {
            m_desc.integrator.type = IntegratorDesc::Type::VolPath;
        }
        else
        {
            fail(command, luisa::format("unsupported Integrator '{}'", type));
        }
        auto params                  = parse_parameters();
        m_desc.integrator.source     = command.loc;
        m_desc.integrator.max_depth  = one_uint(params, "maxdepth", command, 5u);
        m_desc.integrator.parameters = std::move(params);
    }

    void parse_transform(const Token& command)
    {
        Matrix4 transform{};
        (void)expect(TokenKind::LBracket, "expected '[' after Transform");
        for (auto i = 0u; i < 16u; i++)
        {
            if (peek(TokenKind::RBracket) || peek(TokenKind::End))
            {
                fail(command, "Transform expects exactly 16 floats");
            }
            auto row                          = i % 4u;
            auto column                       = i / 4u;
            matrix_at(transform, row, column) = parse_float_token(advance());
        }
        (void)expect(TokenKind::RBracket, "Transform expects exactly 16 floats");
        m_current_transform = transform;
    }

    void parse_scale(const Token& command)
    {
        Matrix4 transform{identity_matrix4};
        matrix_at(transform, 0u, 0u) = next_float(command, "Scale");
        matrix_at(transform, 1u, 1u) = next_float(command, "Scale");
        matrix_at(transform, 2u, 2u) = next_float(command, "Scale");
        concat_transform(transform);
    }

    void parse_translate(const Token& command)
    {
        Matrix4 transform{identity_matrix4};
        matrix_at(transform, 0u, 3u) = next_float(command, "Translate");
        matrix_at(transform, 1u, 3u) = next_float(command, "Translate");
        matrix_at(transform, 2u, 3u) = next_float(command, "Translate");
        concat_transform(transform);
    }

    void parse_rotate(const Token& command)
    {
        constexpr auto pi  = 3.14159265358979323846f;
        auto angle         = next_float(command, "Rotate") * (pi / 180.0f);
        auto axis          = normalize_host(next_float3(command, "Rotate"), command, "Rotate axis");
        auto sin_angle     = std::sin(angle);
        auto cos_angle     = std::cos(angle);
        auto one_minus_cos = 1.0f - cos_angle;
        Matrix4 transform{identity_matrix4};
        matrix_at(transform, 0u, 0u) = axis.x * axis.x * one_minus_cos + cos_angle;
        matrix_at(transform, 0u, 1u) = axis.x * axis.y * one_minus_cos - axis.z * sin_angle;
        matrix_at(transform, 0u, 2u) = axis.x * axis.z * one_minus_cos + axis.y * sin_angle;
        matrix_at(transform, 1u, 0u) = axis.y * axis.x * one_minus_cos + axis.z * sin_angle;
        matrix_at(transform, 1u, 1u) = axis.y * axis.y * one_minus_cos + cos_angle;
        matrix_at(transform, 1u, 2u) = axis.y * axis.z * one_minus_cos - axis.x * sin_angle;
        matrix_at(transform, 2u, 0u) = axis.z * axis.x * one_minus_cos - axis.y * sin_angle;
        matrix_at(transform, 2u, 1u) = axis.z * axis.y * one_minus_cos + axis.x * sin_angle;
        matrix_at(transform, 2u, 2u) = axis.z * axis.z * one_minus_cos + cos_angle;
        concat_transform(transform);
    }

    void parse_look_at(const Token& command)
    {
        auto eye               = next_float3(command, "LookAt");
        auto target            = next_float3(command, "LookAt");
        auto up                = normalize_host(next_float3(command, "LookAt"), command, "LookAt up vector");
        m_desc.camera.world_up = up;
        auto direction         = normalize_host(target - eye, command, "LookAt direction");
        auto right             = normalize_host(cross_host(up, direction), command, "LookAt right vector");
        auto new_up            = cross_host(direction, right);
        Matrix4 camera_from_world{identity_matrix4};
        matrix_at(camera_from_world, 0u, 0u) = right.x;
        matrix_at(camera_from_world, 0u, 1u) = right.y;
        matrix_at(camera_from_world, 0u, 2u) = right.z;
        matrix_at(camera_from_world, 0u, 3u) = -dot_host(right, eye);
        matrix_at(camera_from_world, 1u, 0u) = new_up.x;
        matrix_at(camera_from_world, 1u, 1u) = new_up.y;
        matrix_at(camera_from_world, 1u, 2u) = new_up.z;
        matrix_at(camera_from_world, 1u, 3u) = -dot_host(new_up, eye);
        matrix_at(camera_from_world, 2u, 0u) = direction.x;
        matrix_at(camera_from_world, 2u, 1u) = direction.y;
        matrix_at(camera_from_world, 2u, 2u) = direction.z;
        matrix_at(camera_from_world, 2u, 3u) = -dot_host(direction, eye);
        concat_transform(camera_from_world);
    }

    void parse_sampler(const Token& command)
    {
        expect_options(command);
        auto type = expect_string("Sampler type");
        if (type == "independent")
        {
            m_desc.sampler.type = SamplerDesc::Type::Independent;
        }
        else if (type == "halton")
        {
            m_desc.sampler.type = SamplerDesc::Type::Halton;
        }
        else if (type == "sobol")
        {
            m_desc.sampler.type = SamplerDesc::Type::Sobol;
        }
        else if (type == "zsobol")
        {
            m_desc.sampler.type = SamplerDesc::Type::ZSobol;
        }
        else
        {
            fail(command, luisa::format("unknown Sampler '{}'", type));
        }
        auto params                  = parse_parameters();
        m_desc.sampler.source        = command.loc;
        auto default_spp             = m_desc.sampler.type == SamplerDesc::Type::Independent ? 4u : 16u;
        m_desc.sampler.pixel_samples = one_uint(params, "pixelsamples", command, default_spp);
        if ((m_desc.sampler.type == SamplerDesc::Type::Sobol ||
             m_desc.sampler.type == SamplerDesc::Type::ZSobol) &&
            m_desc.sampler.pixel_samples == 0u)
        {
            auto parameter = find_param(params, "integer", "pixelsamples");
            fail(parameter == nullptr ? command.loc : parameter->source,
                 "'integer pixelsamples' must be greater than zero");
        }
        if (m_desc.sampler.type == SamplerDesc::Type::ZSobol &&
            !std::has_single_bit(m_desc.sampler.pixel_samples))
        {
            auto parameter = find_param(params, "integer", "pixelsamples");
            fail(parameter == nullptr ? command.loc : parameter->source,
                 "ZSobol 'integer pixelsamples' must be a power of two");
        }
        if (m_desc.sampler.type == SamplerDesc::Type::Independent ||
            m_desc.sampler.type == SamplerDesc::Type::Sobol ||
            m_desc.sampler.type == SamplerDesc::Type::ZSobol)
        {
            m_desc.sampler.seed = one_uint(params, "seed", command, 20120712u);
        }
        if (m_desc.sampler.type == SamplerDesc::Type::Sobol ||
            m_desc.sampler.type == SamplerDesc::Type::ZSobol)
        {
            auto randomization = one_string(params, "randomization", command, "fastowen");
            if (randomization != "fastowen")
            {
                auto parameter = find_param(params, "string", "randomization");
                fail(parameter == nullptr ? command.loc : parameter->source,
                     luisa::format("unsupported Sobol randomization '{}'; expected 'fastowen'", randomization));
            }
        }
        m_desc.sampler.parameters = std::move(params);
    }

    void parse_filter(const Token& command)
    {
        expect_options(command);
        m_desc.filter.source = command.loc;
        auto type            = expect_string("PixelFilter type");
        if (type == "triangle")
        {
            m_desc.filter.type = FilterDesc::Type::Triangle;
        }
        else if (type == "gaussian")
        {
            m_desc.filter.type = FilterDesc::Type::Gaussian;
        }
        else
        {
            fail(command, luisa::format("unsupported PixelFilter '{}'", type));
        }
        auto params          = parse_parameters();
        auto default_radius  = m_desc.filter.type == FilterDesc::Type::Triangle ? 2.0f : 1.5f;
        m_desc.filter.radius = make_float2(one_float(params, "xradius", command, default_radius),
                                           one_float(params, "yradius", command, default_radius));
        if (m_desc.filter.type == FilterDesc::Type::Gaussian)
        {
            m_desc.filter.sigma = one_float(params, "sigma", command, 0.5f);
        }
        m_desc.filter.parameters = std::move(params);
    }

    void parse_film(const Token& command)
    {
        expect_options(command);
        auto type = expect_string("Film type");
        if (type != "rgb")
        {
            fail(command, luisa::format("unsupported Film '{}'", type));
        }
        auto params            = parse_parameters();
        m_desc.film.source     = command.loc;
        m_desc.film.type       = FilmDesc::Type::RGB;
        m_desc.film.resolution = make_uint2(one_uint(params, "xresolution", command, 1280u),
                                            one_uint(params, "yresolution", command, 720u));
        m_desc.film.iso        = one_float(params, "iso", command, 100.0f);
        auto filename          = one_string(params, "filename", command, "pbrt.exr");
        if (!filename.empty())
        {
            m_desc.film.filename = std::filesystem::path{filename};
        }
        m_desc.film.parameters = std::move(params);
    }

    void parse_camera(const Token& command)
    {
        expect_options(command);
        auto type = expect_string("Camera type");
        if (type != "perspective")
        {
            fail(command, luisa::format("unsupported Camera '{}'", type));
        }
        auto params                  = parse_parameters();
        m_desc.camera.source         = command.loc;
        m_desc.camera.type           = CameraDesc::Type::Perspective;
        m_desc.camera.fov            = one_float(params, "fov", command, 90.0f);
        m_desc.camera.shutter_open   = one_float(params, "shutteropen", command, 0.0f);
        m_desc.camera.shutter_close  = one_float(params, "shutterclose", command, 1.0f);
        m_desc.camera.pbrt_transform = m_current_transform;
        m_desc.camera.parameters     = std::move(params);
    }

    void parse_world_begin(const Token& command)
    {
        expect_options(command);
        if (!parse_parameters().empty())
        {
            fail(command, "WorldBegin does not take parameters");
        }
        m_block             = Block::World;
        m_current_transform = identity_matrix4;
    }

    [[nodiscard]] MaterialDesc parse_material_desc(
        const Token& command, MaterialDesc::Type material_type,
        luisa::vector<RawParameter> params)
    {
        auto reflectance         = make_float3(0.5f);
        auto reflectance_rgb     = find_param(params, "rgb", "reflectance");
        auto reflectance_texture = optional_texture(params, "reflectance");
        if (reflectance_rgb != nullptr && reflectance_texture)
        {
            fail(reflectance_rgb->source, "material reflectance cannot specify both rgb and texture values");
        }
        if (reflectance_rgb != nullptr)
        {
            reflectance = rgb(params, "reflectance", command);
        }
        auto parse_float_texture = [&](luisa::string_view name, float default_value)
        {
            auto value   = one_float(params, name, command, default_value);
            auto texture = optional_texture(params, name);
            if (find_param(params, "float", name) != nullptr && texture)
            {
                fail(command, luisa::format("material '{}' cannot specify both float and texture values", name));
            }
            return std::pair{value, std::move(texture)};
        };
        auto parse_rgb_texture = [&](luisa::string_view name, float3 default_value)
        {
            auto value   = default_value;
            auto p       = find_param(params, "rgb", name);
            auto texture = optional_texture(params, name);
            if (p != nullptr && texture)
            {
                fail(p->source, luisa::format("material '{}' cannot specify both rgb and texture values", name));
            }
            if (p != nullptr)
            {
                value = rgb(params, name, command);
            }
            return std::pair{value, std::move(texture)};
        };

        auto [roughness, roughness_texture]     = parse_float_texture("roughness", 0.0f);
        auto [u_roughness, u_roughness_texture] = parse_float_texture("uroughness", roughness);
        auto [v_roughness, v_roughness_texture] = parse_float_texture("vroughness", roughness);
        if (find_param(params, "float", "uroughness") == nullptr && !u_roughness_texture)
        {
            u_roughness_texture = roughness_texture;
        }
        if (find_param(params, "float", "vroughness") == nullptr && !v_roughness_texture)
        {
            v_roughness_texture = roughness_texture;
        }
        auto [thickness, thickness_texture] = parse_float_texture("thickness", 0.01f);
        auto [albedo, albedo_texture]       = parse_rgb_texture("albedo", make_float3(0.0f));
        auto [g, g_texture]                 = parse_float_texture("g", 0.0f);
        auto [eta, eta_texture]             = parse_float_texture("eta", 1.5f);
        return MaterialDesc{
            .source              = command.loc,
            .type                = material_type,
            .reflectance         = reflectance,
            .reflectance_texture = std::move(reflectance_texture),
            .roughness           = roughness,
            .roughness_texture   = std::move(roughness_texture),
            .u_roughness         = u_roughness,
            .u_roughness_texture = std::move(u_roughness_texture),
            .v_roughness         = v_roughness,
            .v_roughness_texture = std::move(v_roughness_texture),
            .thickness           = thickness,
            .thickness_texture   = std::move(thickness_texture),
            .albedo              = albedo,
            .albedo_texture      = std::move(albedo_texture),
            .g                   = g,
            .g_texture           = std::move(g_texture),
            .eta                 = eta,
            .eta_texture         = std::move(eta_texture),
            .remap_roughness     = one_bool(params, "remaproughness", command, true),
            .max_depth           = one_uint(params, "maxdepth", command, 10u),
            .samples             = one_uint(params, "nsamples", command, 1u),
            .parameters          = std::move(params),
        };
    }

    void parse_make_named_material(const Token& command)
    {
        expect_world(command);
        auto name   = expect_string("named material name");
        auto params = parse_parameters();
        if (m_desc.named_materials.find(name) != m_desc.named_materials.end())
        {
            fail(command, luisa::format("named material '{}' is redefined", name));
        }
        auto type = one_string(params, "type", command, {});
        MaterialDesc::Type material_type;
        if (type == "diffuse")
        {
            material_type = MaterialDesc::Type::Diffuse;
        }
        else if (type == "coateddiffuse")
        {
            material_type = MaterialDesc::Type::CoatedDiffuse;
        }
        else if (type == "dielectric")
        {
            material_type = MaterialDesc::Type::Dielectric;
        }
        else if (type == "interface")
        {
            material_type = MaterialDesc::Type::Interface;
        }
        else
        {
            fail(command, luisa::format("unknown named material type '{}'", type));
        }
        m_desc.named_materials.emplace(
            std::move(name),
            parse_material_desc(command, material_type, std::move(params)));
    }

    void parse_material(const Token& command)
    {
        expect_world(command);
        auto type   = expect_string("Material type");
        auto params = parse_parameters();
        MaterialDesc::Type material_type;
        if (type == "diffuse")
        {
            material_type = MaterialDesc::Type::Diffuse;
        }
        else if (type == "coateddiffuse")
        {
            material_type = MaterialDesc::Type::CoatedDiffuse;
        }
        else if (type == "dielectric")
        {
            material_type = MaterialDesc::Type::Dielectric;
        }
        else if (type == "interface")
        {
            material_type = MaterialDesc::Type::Interface;
        }
        else
        {
            fail(command, luisa::format("unknown Material '{}'", type));
        }
        auto index = static_cast<uint>(m_desc.materials.size());
        m_desc.materials.emplace_back(
            parse_material_desc(command, material_type, std::move(params)));
        m_current_material = MaterialBinding{.inline_index = index};
    }

    void parse_make_named_medium(const Token& command)
    {
        expect_world(command);
        auto name   = expect_string("named medium name");
        auto params = parse_parameters();
        if (name.empty())
        {
            fail(command, "MakeNamedMedium name must not be empty");
        }
        if (m_desc.named_media.find(name) != m_desc.named_media.end())
        {
            fail(command, luisa::format("named medium '{}' is redefined", name));
        }
        auto type = one_string(params, "type", command, {});
        if (type != "homogeneous")
        {
            fail(command, luisa::format("unknown named medium type '{}'", type));
        }
        auto sigma_a             = one_float3(params, "rgb", "sigma_a", make_float3(0.0f));
        auto sigma_s             = one_float3(params, "rgb", "sigma_s", make_float3(0.0f));
        auto scale               = one_float(params, "scale", command, 1.0f);
        auto g                   = one_float(params, "g", command, 0.0f);
        auto finite_non_negative = [](float3 value) noexcept
        {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
                   value.x >= 0.0f && value.y >= 0.0f && value.z >= 0.0f;
        };
        if (!finite_non_negative(sigma_a))
        {
            auto p = find_param(params, "rgb", "sigma_a");
            fail(p == nullptr ? command.loc : p->source, "homogeneous sigma_a must be finite and non-negative");
        }
        if (!finite_non_negative(sigma_s))
        {
            auto p = find_param(params, "rgb", "sigma_s");
            fail(p == nullptr ? command.loc : p->source, "homogeneous sigma_s must be finite and non-negative");
        }
        if (!std::isfinite(scale) || scale < 0.0f)
        {
            auto p = find_param(params, "float", "scale");
            fail(p == nullptr ? command.loc : p->source, "homogeneous scale must be finite and non-negative");
        }
        if (!std::isfinite(g) || std::abs(g) >= 1.0f)
        {
            auto p = find_param(params, "float", "g");
            fail(p == nullptr ? command.loc : p->source, "homogeneous g must satisfy abs(g) < 1");
        }
        m_desc.named_media.emplace(std::move(name), MediumDesc{
                                                        .source     = command.loc,
                                                        .type       = MediumDesc::Type::Homogeneous,
                                                        .sigma_a    = sigma_a,
                                                        .sigma_s    = sigma_s,
                                                        .scale      = scale,
                                                        .g          = g,
                                                        .parameters = std::move(params),
                                                    });
    }

    void parse_medium_interface(const Token& command)
    {
        expect_world(command);
        auto inside  = expect_string("MediumInterface inside medium");
        auto outside = expect_string("MediumInterface outside medium");
        if (!parse_parameters().empty())
        {
            fail(command, "MediumInterface does not take parameters");
        }
        m_current_medium_interface = {.inside = std::move(inside), .outside = std::move(outside)};
    }

    void parse_texture(const Token& command)
    {
        expect_world(command);
        auto name       = expect_string("Texture name");
        auto value_type = expect_string("Texture value type");
        auto type       = expect_string("Texture implementation");
        auto params     = parse_parameters();
        TextureDesc desc{.source = command.loc, .name = std::move(name), .parameters = std::move(params)};
        if (value_type == "float")
        {
            desc.value_type = TextureDesc::ValueType::Float;
        }
        else if (value_type == "spectrum")
        {
            desc.value_type = TextureDesc::ValueType::Spectrum;
        }
        else
        {
            fail(command, luisa::format("unknown Texture value type '{}'", value_type));
        }
        if (type == "imagemap")
        {
            desc.type = TextureDesc::Type::ImageMap;
        }
        else if (type == "constant")
        {
            desc.type = TextureDesc::Type::Constant;
        }
        else if (type == "scale")
        {
            desc.type = TextureDesc::Type::Scale;
        }
        else if (type == "checkerboard")
        {
            desc.type = TextureDesc::Type::Checkerboard;
        }
        else
        {
            fail(command, luisa::format("unknown Texture '{}'", type));
        }
        if (desc.type == TextureDesc::Type::ImageMap)
        {
            auto filename = one_string(desc.parameters, "filename", command, {});
            if (filename.empty())
            {
                fail(command, "imagemap texture requires a non-empty 'string filename' parameter");
            }
            desc.filename = std::filesystem::path{filename};
            auto filter   = one_string(desc.parameters, "filter", command, "bilinear");
            if (filter == "point")
            {
                desc.filter = TextureDesc::Filter::Point;
            }
            else if (filter == "bilinear")
            {
                desc.filter = TextureDesc::Filter::Bilinear;
            }
            else
            {
                auto filter_param = find_param(desc.parameters, "string", "filter");
                fail(filter_param == nullptr ? command.loc : filter_param->source,
                     luisa::format("unsupported imagemap filter '{}'; supported filters are 'point' and 'bilinear'", filter));
            }
            if (auto encoding_param = find_param(desc.parameters, "string", "encoding"); encoding_param != nullptr)
            {
                auto encoding = one_string(desc.parameters, "encoding", command, {});
                if (encoding == "linear")
                {
                    desc.encoding = TextureDesc::Encoding::Linear;
                }
                else if (encoding == "sRGB")
                {
                    desc.encoding = TextureDesc::Encoding::SRGB;
                }
                else
                {
                    fail(encoding_param->source,
                         luisa::format("unsupported imagemap encoding '{}'; supported encodings are 'linear' and 'sRGB'", encoding));
                }
            }
            desc.uv_scale = make_float2(
                one_float(desc.parameters, "uscale", command, 1.0f),
                one_float(desc.parameters, "vscale", command, 1.0f));
            desc.image_scale = one_float(desc.parameters, "scale", command, 1.0f);
        }
        else if (desc.type == TextureDesc::Type::Constant)
        {
            if (desc.value_type != TextureDesc::ValueType::Float)
            {
                fail(command, "only float constant textures are currently supported");
            }
            (void)require_param(desc.parameters, "float", "value", command);
            desc.constant_value = one_float(desc.parameters, "value", command, 0.0f);
        }
        else if (desc.type == TextureDesc::Type::Scale)
        {
            if (desc.value_type != TextureDesc::ValueType::Float)
            {
                fail(command, "only float scale textures are currently supported");
            }
            auto tex   = optional_texture(desc.parameters, "tex");
            auto scale = optional_texture(desc.parameters, "scale");
            if (!tex || !scale)
            {
                fail(command, "scale texture requires 'texture tex' and 'texture scale' parameters");
            }
            desc.tex   = std::move(*tex);
            desc.scale = std::move(*scale);
        }
        else if (desc.type == TextureDesc::Type::Checkerboard)
        {
            auto dimension = one_uint(desc.parameters, "dimension", command, 2u);
            if (dimension != 2u)
            {
                auto p = find_param(desc.parameters, "integer", "dimension");
                fail(p == nullptr ? command.loc : p->source,
                     luisa::format("unsupported checkerboard dimension {}; only 2D is supported", dimension));
            }
            auto mapping = one_string(desc.parameters, "mapping", command, "uv");
            if (mapping != "uv")
            {
                auto p = find_param(desc.parameters, "string", "mapping");
                fail(p == nullptr ? command.loc : p->source,
                     luisa::format("unsupported checkerboard mapping '{}'; only 'uv' is supported", mapping));
            }
            desc.uv_scale = make_float2(
                one_float(desc.parameters, "uscale", command, 1.0f),
                one_float(desc.parameters, "vscale", command, 1.0f));

            auto parse_input = [&](luisa::string_view input_name, float default_value)
            {
                TextureInputDesc input;
                auto texture       = optional_texture(desc.parameters, input_name);
                auto constant_type = desc.value_type == TextureDesc::ValueType::Float ? "float" : "rgb";
                auto constant      = find_param(desc.parameters, constant_type, input_name);
                if (texture && constant != nullptr)
                {
                    fail(constant->source,
                         luisa::format("checkerboard '{}' cannot be specified as both texture and {}",
                                       input_name,
                                       constant_type));
                }
                if (texture)
                {
                    input.texture = std::move(*texture);
                }
                else if (constant != nullptr)
                {
                    if (desc.value_type == TextureDesc::ValueType::Float)
                    {
                        input.constant = make_float4(one_float(desc.parameters, input_name, command, default_value));
                    }
                    else
                    {
                        input.constant = make_float4(rgb(desc.parameters, input_name, command), 1.0f);
                    }
                }
                else if (desc.value_type == TextureDesc::ValueType::Float)
                {
                    input.constant = make_float4(default_value);
                }
                else
                {
                    input.constant = make_float4(make_float3(default_value), 1.0f);
                }
                return input;
            };
            desc.tex1 = parse_input("tex1", 1.0f);
            desc.tex2 = parse_input("tex2", 0.0f);
        }

        for (auto i = 0u; i < desc.parameters.size(); i++)
        {
            auto&& p       = desc.parameters[i];
            auto supported = false;
            if (desc.type == TextureDesc::Type::ImageMap)
            {
                supported = (p.type == "string" && (p.name == "filename" || p.name == "filter" || p.name == "encoding")) ||
                            (p.type == "float" && (p.name == "uscale" || p.name == "vscale" || p.name == "scale"));
            }
            else if (desc.type == TextureDesc::Type::Constant)
            {
                supported = p.type == "float" && p.name == "value";
            }
            else if (desc.type == TextureDesc::Type::Scale)
            {
                supported = p.type == "texture" && (p.name == "tex" || p.name == "scale");
            }
            else if (desc.type == TextureDesc::Type::Checkerboard)
            {
                auto input      = p.name == "tex1" || p.name == "tex2";
                auto input_type = p.type == "texture" ||
                                  (desc.value_type == TextureDesc::ValueType::Float ? p.type == "float" : p.type == "rgb");
                supported       = (input && input_type) ||
                                  (p.type == "float" && (p.name == "uscale" || p.name == "vscale")) ||
                                  (p.type == "integer" && p.name == "dimension") ||
                                  (p.type == "string" && p.name == "mapping");
            }
            if (!supported)
            {
                fail(p.source, luisa::format("unsupported parameter '\"{} {}\"' for this texture", p.type, p.name));
            }
            for (auto j = 0u; j < i; j++)
            {
                if (desc.parameters[j].type == p.type && desc.parameters[j].name == p.name)
                {
                    fail(p.source, luisa::format("duplicate texture parameter '\"{} {}\"'", p.type, p.name));
                }
            }
        }
        if (desc.type == TextureDesc::Type::Constant && !std::isfinite(desc.constant_value))
        {
            fail(command, "texture constant value must be finite");
        }
        if ((desc.type == TextureDesc::Type::ImageMap || desc.type == TextureDesc::Type::Checkerboard) &&
            (!std::isfinite(desc.uv_scale.x) || !std::isfinite(desc.uv_scale.y)))
        {
            fail(command, "texture UV scale must be finite");
        }
        if (desc.type == TextureDesc::Type::ImageMap && !std::isfinite(desc.image_scale))
        {
            fail(command, "imagemap texture scale must be finite");
        }
        if (desc.type == TextureDesc::Type::Checkerboard)
        {
            auto finite = [](float4 v) noexcept
            {
                return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
            };
            if ((!desc.tex1.texture && !finite(desc.tex1.constant)) ||
                (!desc.tex2.texture && !finite(desc.tex2.constant)))
            {
                fail(command, "checkerboard texture constants must be finite");
            }
        }
        m_desc.textures.emplace_back(std::move(desc));
    }

    void parse_named_material(const Token& command)
    {
        expect_world(command);
        auto name = expect_string("named material reference");
        if (!parse_parameters().empty())
        {
            fail(command, "NamedMaterial does not take parameters");
        }
        m_current_material = MaterialBinding{.named = std::move(name)};
    }

    void parse_area_light_source(const Token& command)
    {
        expect_world(command);
        auto type = expect_string("AreaLightSource type");
        if (type != "diffuse")
        {
            fail(command, luisa::format("unsupported AreaLightSource '{}'", type));
        }
        auto params = parse_parameters();
        m_current_area_light.emplace(AreaLightDesc{
            .source     = command.loc,
            .type       = AreaLightDesc::Type::Diffuse,
            .emission   = rgb(params, "L", command),
            .parameters = std::move(params),
        });
    }

    void parse_light_source(const Token& command)
    {
        expect_world(command);
        auto type = expect_string("LightSource type");
        if (type != "infinite" && type != "distant")
        {
            fail(command, luisa::format("unsupported LightSource '{}'", type));
        }
        auto params = parse_parameters();
        if (type == "distant")
        {
            for (auto i = 0u; i < params.size(); i++)
            {
                auto&& param   = params[i];
                auto supported = (param.type == "rgb" && param.name == "L") ||
                                 (param.type == "float" && (param.name == "scale" || param.name == "illuminance")) ||
                                 (param.type == "point3" && (param.name == "from" || param.name == "to"));
                if (!supported)
                {
                    fail(param.source, luisa::format("unsupported parameter '\"{} {}\"' for distant LightSource", param.type, param.name));
                }
                for (auto j = 0u; j < i; j++)
                {
                    if (params[j].name == param.name)
                    {
                        fail(param.source, luisa::format("duplicate distant LightSource parameter '\"{} {}\"'", param.type, param.name));
                    }
                }
            }
            auto L     = one_float3(params, "rgb", "L", make_float3(1.0f));
            auto scale = one_float(params, "scale", command, 1.0f);
            luisa::optional<float> illuminance;
            if (find_param(params, "float", "illuminance") != nullptr)
            {
                illuminance.emplace(one_float(params, "illuminance", command, -1.0f));
            }
            auto from   = one_float3(params, "point3", "from", make_float3(0.0f));
            auto to     = one_float3(params, "point3", "to", make_float3(0.0f, 0.0f, 1.0f));
            auto finite = [](float3 v) noexcept
            {
                return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
            };
            if (!finite(L) || L.x < 0.0f || L.y < 0.0f || L.z < 0.0f)
            {
                auto p = find_param(params, "rgb", "L");
                fail(p == nullptr ? command.loc : p->source,
                     "distant LightSource radiance must be finite and non-negative");
            }
            if (!std::isfinite(scale) || scale < 0.0f)
            {
                auto p = find_param(params, "float", "scale");
                fail(p == nullptr ? command.loc : p->source,
                     "distant LightSource scale must be finite and non-negative");
            }
            if (illuminance && !std::isfinite(*illuminance))
            {
                auto p = find_param(params, "float", "illuminance");
                fail(p == nullptr ? command.loc : p->source,
                     "distant LightSource illuminance must be finite");
            }
            auto direction      = from - to;
            auto length_squared = dot_host(direction, direction);
            if (!finite(from) || !finite(to) || !std::isfinite(length_squared) || length_squared < 1e-16f)
            {
                fail(command, "distant LightSource 'from' and 'to' must define a finite non-zero direction");
            }
            m_desc.distant_lights.emplace_back(DistantLightDesc{
                .source         = command.loc,
                .L              = L,
                .scale          = scale,
                .illuminance    = illuminance,
                .from           = from,
                .to             = to,
                .pbrt_transform = m_current_transform,
                .parameters     = std::move(params),
            });
            return;
        }
        for (auto i = 0u; i < params.size(); i++)
        {
            auto&& param   = params[i];
            auto supported = (param.type == "rgb" && param.name == "L") ||
                             (param.type == "string" && param.name == "filename") ||
                             (param.type == "float" && (param.name == "scale" || param.name == "illuminance"));
            if (!supported)
            {
                fail(param.source, luisa::format("unsupported parameter '\"{} {}\"' for infinite LightSource", param.type, param.name));
            }
            for (auto j = 0u; j < i; j++)
            {
                if (params[j].name == param.name)
                {
                    fail(param.source, luisa::format("duplicate infinite LightSource parameter '\"{} {}\"'", param.type, param.name));
                }
            }
        }
        luisa::optional<float3> L;
        if (find_param(params, "rgb", "L") != nullptr)
        {
            L.emplace(one_float3(params, "rgb", "L", make_float3(1.0f)));
        }
        auto filename = one_string(params, "filename", command, {});
        if (L && !filename.empty())
        {
            auto p = find_param(params, "string", "filename");
            fail(p == nullptr ? command.loc : p->source,
                 "infinite LightSource cannot specify both 'rgb L' and 'string filename'");
        }
        auto scale = one_float(params, "scale", command, 1.0f);
        luisa::optional<float> illuminance;
        if (find_param(params, "float", "illuminance") != nullptr)
        {
            illuminance.emplace(one_float(params, "illuminance", command, -1.0f));
        }
        auto finite = [](float3 v) noexcept
        {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        };
        if (L && (!finite(*L) || L->x < 0.0f || L->y < 0.0f || L->z < 0.0f))
        {
            auto p = find_param(params, "rgb", "L");
            fail(p == nullptr ? command.loc : p->source,
                 "infinite LightSource radiance must be finite and non-negative");
        }
        if (!std::isfinite(scale) || scale < 0.0f)
        {
            auto p = find_param(params, "float", "scale");
            fail(p == nullptr ? command.loc : p->source,
                 "infinite LightSource scale must be finite and non-negative");
        }
        if (illuminance && !std::isfinite(*illuminance))
        {
            auto p = find_param(params, "float", "illuminance");
            fail(p == nullptr ? command.loc : p->source,
                 "infinite LightSource illuminance must be finite");
        }
        if (!filename.empty() && illuminance)
        {
            auto p = find_param(params, "float", "illuminance");
            fail(p == nullptr ? command.loc : p->source,
                 "illuminance for image infinite LightSource is not supported");
        }
        m_desc.infinite_lights.emplace_back(InfiniteLightDesc{
            .source         = command.loc,
            .L              = L,
            .filename       = std::filesystem::path{std::move(filename)},
            .scale          = scale,
            .illuminance    = illuminance,
            .pbrt_transform = m_current_transform,
            .parameters     = std::move(params),
        });
    }

    void parse_attribute_begin(const Token& command)
    {
        expect_world(command);
        if (!parse_parameters().empty())
        {
            fail(command, "AttributeBegin does not take parameters");
        }
        m_attribute_stack.emplace_back(AttributeState{
            .material         = m_current_material,
            .area_light       = m_current_area_light,
            .medium_interface = m_current_medium_interface,
            .transform        = m_current_transform,
        });
    }

    void parse_attribute_end(const Token& command)
    {
        expect_world(command);
        if (!parse_parameters().empty())
        {
            fail(command, "AttributeEnd does not take parameters");
        }
        if (m_attribute_stack.empty())
        {
            fail(command, "unmatched AttributeEnd");
        }
        auto state = std::move(m_attribute_stack.back());
        m_attribute_stack.pop_back();
        m_current_material         = std::move(state.material);
        m_current_area_light       = std::move(state.area_light);
        m_current_medium_interface = std::move(state.medium_interface);
        m_current_transform        = state.transform;
    }

    void parse_shape(const Token& command)
    {
        expect_world(command);
        auto type   = expect_string("Shape type");
        auto params = parse_parameters();
        ShapeDesc shape{
            .source           = command.loc,
            .parameters       = params,
            .material         = m_current_material,
            .area_light       = m_current_area_light,
            .medium_interface = m_current_medium_interface,
            .pbrt_transform   = m_current_transform,
        };
        const RawParameter* alpha_parameter = nullptr;
        for (auto&& param : params)
        {
            if (param.name != "alpha")
            {
                continue;
            }
            if (param.type != "float" && param.type != "texture")
            {
                fail(param.source, "shape parameter 'alpha' must have type 'float' or 'texture'");
            }
            if (alpha_parameter != nullptr)
            {
                fail(param.source, "shape alpha cannot be specified more than once");
            }
            alpha_parameter = &param;
        }
        shape.alpha         = one_float(params, "alpha", command, 1.0f);
        shape.alpha_texture = optional_texture(params, "alpha");
        if (!std::isfinite(shape.alpha))
        {
            fail(alpha_parameter == nullptr ? command.loc : alpha_parameter->source,
                 "shape alpha must be finite");
        }
        if (type == "sphere")
        {
            shape.type             = ShapeDesc::Type::Sphere;
            auto radius_count      = 0u;
            auto subdivision_count = 0u;
            for (auto&& param : params)
            {
                if (param.name == "alpha")
                {
                    continue;
                }
                if (param.name == "zmin" || param.name == "zmax" || param.name == "phimax")
                {
                    fail(param.source, luisa::format("PBRT sphere clipping parameter '{}' is not supported", param.name));
                }
                if (param.name == "radius")
                {
                    if (param.type != "float")
                    {
                        fail(param.source, "sphere parameter 'radius' must have type 'float'");
                    }
                    if (radius_count++ != 0u)
                    {
                        fail(param.source, "duplicate parameter 'float radius'");
                    }
                }
                else if (param.name == "subdivision")
                {
                    if (param.type != "integer")
                    {
                        fail(param.source, "sphere parameter 'subdivision' must have type 'integer'");
                    }
                    if (subdivision_count++ != 0u)
                    {
                        fail(param.source, "duplicate parameter 'integer subdivision'");
                    }
                }
                else
                {
                    fail(param.source, luisa::format("unsupported sphere parameter '{} {}'", param.type, param.name));
                }
            }
            shape.radius             = one_float(params, "radius", command, 1.0f);
            shape.sphere_subdivision = one_uint(params, "subdivision", command, ShapeDesc::sphere_default_subdivision);
            if (!std::isfinite(shape.radius) || shape.radius <= 0.0f)
            {
                fail(command, "sphere radius must be finite and positive");
            }
            if (shape.sphere_subdivision > ShapeDesc::sphere_max_subdivision)
            {
                fail(command, luisa::format("sphere subdivision level must not exceed {}", ShapeDesc::sphere_max_subdivision));
            }
        }
        else if (type == "plymesh")
        {
            shape.type                         = ShapeDesc::Type::PlyMesh;
            const RawParameter* filename_param = nullptr;
            for (auto&& param : params)
            {
                if (param.type == "string" && param.name == "filename")
                {
                    if (filename_param != nullptr)
                    {
                        fail(param.source, "duplicate parameter 'string filename'");
                    }
                    filename_param = &param;
                }
            }
            auto filename = one_string(params, "filename", command, {});
            if (filename.empty())
            {
                fail(command, "plymesh requires a non-empty 'string filename'");
            }
            shape.filename = std::filesystem::path{std::move(filename)};
        }
        else if (type == "trianglemesh")
        {
            shape.type     = ShapeDesc::Type::TriangleMesh;
            auto positions = float3_array(params, "point3", "P", command);
            auto normals   = optional_float3_array(params, "normal", "N");
            auto uvs       = optional_float2_array(params, "point2", "uv");
            if (!normals.empty() && normals.size() != positions.size())
            {
                fail(command, "'normal N' count must match 'point3 P' count");
            }
            if (!uvs.empty() && uvs.size() != positions.size())
            {
                fail(command, "'point2 uv' count must match 'point3 P' count");
            }
            auto indices = triangle_indices(params, command, positions.size());

            shape.mesh_index = static_cast<uint>(m_desc.meshes.size());
            m_desc.meshes.emplace_back(MeshDesc{
                .source    = command.loc,
                .positions = std::move(positions),
                .normals   = std::move(normals),
                .uvs       = std::move(uvs),
                .indices   = std::move(indices),
            });
        }
        else
        {
            fail(command, luisa::format("unknown Shape '{}'", type));
        }
        m_desc.shapes.emplace_back(std::move(shape));
    }
};

[[nodiscard]] luisa::string read_file(const std::filesystem::path& path)
{
    std::ifstream input{path};
    if (!input)
    {
        auto s = luisa::format("Failed to open PBRT scene '{}'.", path.string());
        throw std::runtime_error{s.c_str()};
    }
    std::ostringstream ss;
    ss << input.rdbuf();
    auto source = ss.str();
    return luisa::string{source.c_str()};
}

} // namespace

PbrtScene PbrtParser::parse(const std::filesystem::path& path)
{
    Clock clock;
    auto source_path = std::filesystem::absolute(path);
    LUISA_INFO("Loading PBRT scene '{}'.", source_path.string());
    auto source = read_file(source_path);
    Tokenizer tokenizer{source_path, std::move(source)};
    PbrtScene desc{};
    desc.source_path       = source_path;
    desc.camera.source     = SourceLocation{source_path};
    desc.film.source       = SourceLocation{source_path};
    desc.integrator.source = SourceLocation{source_path};
    desc.sampler.source    = SourceLocation{source_path};
    desc.filter.source     = SourceLocation{source_path};
    Parser parser{std::move(desc), tokenizer.tokenize()};
    auto scene = parser.parse();
    LUISA_INFO("Parsed PBRT scene '{}' in {} ms.", source_path.string(), clock.toc());
    return scene;
}

} // namespace Yutrel
