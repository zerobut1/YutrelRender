#include "sampler.h"

#include "base/renderer.h"

namespace Yutrel
{
Sampler::Instance::Instance(const Renderer& renderer, const Sampler* sampler) noexcept
    : _renderer{renderer}, _sampler{sampler}
{
}
} // namespace Yutrel
