#pragma once

#include <luisa/dsl/syntax.h>

namespace Yutrel
{
class Renderer;
class CommandBuffer;

using namespace luisa;
using namespace luisa::compute;

class SPD
{
private:
    const Renderer& m_renderer;
    uint m_buffer_id;
    float m_sample_interval;

public:
    SPD(Renderer& renderer, uint buffer_id, float sample_interval) noexcept;
    [[nodiscard]] static SPD create_cie_x(Renderer& renderer, CommandBuffer& cb) noexcept;
    [[nodiscard]] static SPD create_cie_y(Renderer& renderer, CommandBuffer& cb) noexcept;
    [[nodiscard]] static SPD create_cie_z(Renderer& renderer, CommandBuffer& cb) noexcept;
    [[nodiscard]] static SPD create_cie_d65(Renderer& renderer, CommandBuffer& cb) noexcept;
    [[nodiscard]] static float cie_y_integral() noexcept;
    [[nodiscard]] Float sample(Expr<float> lambda) const noexcept;
};

} // namespace Yutrel
