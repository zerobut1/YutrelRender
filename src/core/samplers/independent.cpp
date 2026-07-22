#include "independent.h"

#include "base/renderer.h"
#include "scene/scene_builder.h"
#include "utils/rng.h"

namespace Yutrel
{
IndependentSampler::Instance::Instance(const Renderer& renderer, const IndependentSampler* sampler) noexcept
    : Sampler::Instance{renderer, sampler}
{
}

void IndependentSampler::Instance::reset(CommandBuffer& command_buffer, uint2 resolution, uint state_count) noexcept
{
    static_cast<void>(resolution);
    if (!_states || state_count > _states.size())
    {
        _states = renderer().device().create_buffer<uint>(next_pow2(state_count));
    }
}

void IndependentSampler::Instance::start(UInt2 pixel, UInt index) noexcept
{
    _state.emplace(xxhash32(make_uint4(pixel, base<IndependentSampler>()->seed(), index)));
}

Float IndependentSampler::Instance::generate_1d() noexcept
{
    Float u = 0.0f;
    $outline { u = lcg(*_state); };
    return u;
}

Float2 IndependentSampler::Instance::generate_2d() noexcept
{
    Float2 u = make_float2();
    $outline
    {
        u.x = generate_1d();
        u.y = generate_1d();
    };
    return u;
}

luisa::unique_ptr<Sampler::Instance> IndependentSampler::build(const Renderer& renderer) const noexcept
{
    return luisa::make_unique<Instance>(renderer, this);
}

const Sampler* IndependentSamplerSpec::build(SceneBuilder& builder) const noexcept
{
    return builder.emplace<Sampler, IndependentSampler>(_spp, _seed);
}
} // namespace Yutrel
