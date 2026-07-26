#include "light.h"

#include <luisa/core/logging.h>

namespace Yutrel
{

Light::Closure::EmissionSample Light::Closure::sample_le(
    Expr<uint> /*instance_id*/, Expr<float2> /*u_position*/, Expr<float2> /*u_direction*/) const noexcept
{
    LUISA_WARNING_WITH_LOCATION("sample_le() not implemented for this light type.");
    return EmissionSample::zero(swl().dimension());
}

} // namespace Yutrel
