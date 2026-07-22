#include "rgb2spec.h"

namespace Yutrel
{

RGB2SpectrumTable RGB2SpectrumTable::srgb() noexcept
{
    return RGB2SpectrumTable{sRGBToSpectrumTable_Data};
}

} // namespace Yutrel