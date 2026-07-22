#include "command_buffer.h"

#include <luisa/core/logging.h>

namespace Yutrel
{
CommandBuffer::CommandBuffer(Stream& stream) noexcept
    : _stream{&stream} {}

CommandBuffer::~CommandBuffer() noexcept
{
    LUISA_ASSERT(_list.empty(),
                 "Command buffer not empty when destroyed. "
                 "Did you forget to commit?");
}

} // namespace Yutrel
