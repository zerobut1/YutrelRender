#pragma once

#include <luisa/dsl/syntax.h>

namespace Yutrel
{
using namespace luisa;
using namespace luisa::compute;

[[nodiscard]] UInt xxhash32(Expr<uint> p) noexcept;
[[nodiscard]] UInt xxhash32(Expr<uint2> p) noexcept;
[[nodiscard]] UInt xxhash32(Expr<uint3> p) noexcept;
[[nodiscard]] UInt xxhash32(Expr<uint4> p) noexcept;

// PBRT-v4's byte-wise MurmurHash64A for packed 32-bit words.
[[nodiscard]] ULong pbrt_hash64(Expr<uint2> p) noexcept;
[[nodiscard]] ULong pbrt_hash64(Expr<uint3> p) noexcept;
[[nodiscard]] ULong pbrt_hash64(Expr<uint4> p) noexcept;
[[nodiscard]] Float pbrt_hash_float(Expr<float3> p) noexcept;
[[nodiscard]] Float pbrt_hash_float(Expr<float3> origin, Expr<float3> direction) noexcept;

class PBRTRNG
{
private:
    ULong _state;
    ULong _inc;

    [[nodiscard]] UInt _uniform_uint() noexcept;

public:
    PBRTRNG(Expr<ulong> sequence, Expr<ulong> offset) noexcept;
    [[nodiscard]] Float uniform() noexcept;
};

[[nodiscard]] Float uniform_uint_to_float(Expr<uint> u) noexcept;

[[nodiscard]] Float lcg(UInt& state) noexcept;

} // namespace Yutrel

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(Yutrel::PBRTRNG)
