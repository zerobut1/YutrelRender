#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace Yutrel
{
// credit: https://github.com/AirGuanZ/agz-utils/blob/master/include/agz-utils/console/pbar.h#L137
class ProgressBar
{
public:
    static constexpr auto complete_char   = '=';
    static constexpr auto heading_char    = '>';
    static constexpr auto incomplete_char = ' ';
    using clock_type                      = std::chrono::steady_clock;

private:
    double _progress;
    double _last_drawn_progress;
    uint32_t _width;
    size_t _last_line_size;
    bool _interactive;
    clock_type::time_point _start;
    clock_type::time_point _last_draw;

public:
    explicit ProgressBar(uint32_t width = 32u) noexcept;
    void reset() noexcept;
    void update(double progress) noexcept;
    void done() noexcept;
    void cancel() noexcept;

private:
    void draw(bool cancelled) noexcept;
};

} // namespace Yutrel
