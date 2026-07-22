#include "rng.h"

#include <luisa/luisa-compute.h>

namespace Yutrel
{
UInt xxhash32(Expr<uint> p) noexcept
{
    static Callable impl = [](UInt p) noexcept
    {
        constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
        constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
        auto h32 = p + PRIME32_5;
        h32      = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32      = PRIME32_2 * (h32 ^ (h32 >> 15u));
        h32      = PRIME32_3 * (h32 ^ (h32 >> 13u));
        return h32 ^ (h32 >> 16u);
    };
    return impl(p);
}

UInt xxhash32(Expr<uint2> p) noexcept
{
    static Callable impl = [](UInt2 p) noexcept
    {
        constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
        constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
        auto h32 = p.y + PRIME32_5 + p.x * PRIME32_3;
        h32      = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32      = PRIME32_2 * (h32 ^ (h32 >> 15u));
        h32      = PRIME32_3 * (h32 ^ (h32 >> 13u));
        return h32 ^ (h32 >> 16u);
    };
    return impl(p);
}

UInt xxhash32(Expr<uint3> p) noexcept
{
    static Callable impl = [](UInt3 p) noexcept
    {
        constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
        constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
        UInt h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
        h32      = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 += p.y * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
        h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
        return h32 ^ (h32 >> 16u);
    };
    return impl(p);
}

UInt xxhash32(Expr<uint4> p) noexcept
{
    static Callable impl = [](UInt4 p) noexcept
    {
        constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
        constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
        auto h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
        h32      = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 += p.y * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 += p.z * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
        h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
        return h32 ^ (h32 >> 16u);
    };
    return impl(p);
}

namespace
{
[[nodiscard]] ULong murmur_hash64_words(Expr<uint4> words, uint count) noexcept
{
    constexpr auto m = 0xc6a4a7935bd1e995ull;
    constexpr auto r = 47ull;
    auto h = def(static_cast<ulong>(count * sizeof(uint)) * m);
    auto blocks = count / 2u;
    for (auto i = 0u; i < blocks; i++)
    {
        auto k = cast<ulong>(words[i * 2u]) |
                 (cast<ulong>(words[i * 2u + 1u]) << 32ull);
        k *= m;
        k ^= k >> r;
        k *= m;
        h ^= k;
        h *= m;
    }
    if ((count & 1u) != 0u)
    {
        h ^= cast<ulong>(words[count - 1u]);
        h *= m;
    }
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}
}

ULong pbrt_hash64(Expr<uint2> p) noexcept
{
    return murmur_hash64_words(make_uint4(p, 0u, 0u), 2u);
}

ULong pbrt_hash64(Expr<uint3> p) noexcept
{
    return murmur_hash64_words(make_uint4(p, 0u), 3u);
}

ULong pbrt_hash64(Expr<uint4> p) noexcept
{
    return murmur_hash64_words(p, 4u);
}

Float pbrt_hash_float(Expr<float3> p) noexcept
{
    return cast<uint>(pbrt_hash64(as<uint3>(p))) * 0x1p-32f;
}

Float pbrt_hash_float(Expr<float3> origin, Expr<float3> direction) noexcept
{
    static Callable impl = [](Float3 origin, Float3 direction) noexcept
    {
        constexpr auto m = 0xc6a4a7935bd1e995ull;
        constexpr auto r = 47ull;
        auto o            = as<uint3>(origin);
        auto d            = as<uint3>(direction);
        auto h            = def(24ull * m);
        auto mix          = [&](UInt lo, UInt hi) noexcept
        {
            auto k = cast<ulong>(lo) | (cast<ulong>(hi) << 32ull);
            k *= m;
            k ^= k >> r;
            k *= m;
            h ^= k;
            h *= m;
        };
        mix(o.x, o.y);
        mix(o.z, d.x);
        mix(d.y, d.z);
        h ^= h >> r;
        h *= m;
        h ^= h >> r;
        return cast<uint>(h) * 0x1p-32f;
    };
    return impl(origin, direction);
}

UInt PBRTRNG::_uniform_uint() noexcept
{
    constexpr auto multiplier = 0x5851f42d4c957f2dull;
    auto old_state = def(_state);
    _state = old_state * multiplier + _inc;
    auto xorshifted = cast<uint>(((old_state >> 18ull) ^ old_state) >> 27ull);
    auto rotation = cast<uint>(old_state >> 59ull);
    return (xorshifted >> rotation) |
           (xorshifted << ((~rotation + 1u) & 31u));
}

PBRTRNG::PBRTRNG(Expr<ulong> sequence, Expr<ulong> offset) noexcept
    : _state{0ull}, _inc{(sequence << 1ull) | 1ull}
{
    static_cast<void>(_uniform_uint());
    _state += offset;
    static_cast<void>(_uniform_uint());
}

Float PBRTRNG::uniform() noexcept
{
    return uniform_uint_to_float(_uniform_uint());
}

Float uniform_uint_to_float(Expr<uint> u) noexcept
{
    return min(one_minus_epsilon, u * 0x1p-32f);
}

Float lcg(UInt& state) noexcept
{
    static Callable impl = [](UInt& state) noexcept
    {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state                = lcg_a * state + lcg_c;
        return uniform_uint_to_float(state);
    };
    return impl(state);
}
} // namespace Yutrel
