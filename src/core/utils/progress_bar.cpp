#include "progress_bar.h"

#include <luisa/core/stl.h>

#include <cstdio>
#include <iostream>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace Yutrel
{
namespace
{
[[nodiscard]] bool stderr_is_terminal() noexcept
{
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
}
} // namespace

ProgressBar::ProgressBar(uint32_t width) noexcept
    : _progress{0.0},
      _last_drawn_progress{-1.0},
      _width{width},
      _last_line_size{0u},
      _interactive{stderr_is_terminal()},
      _start{clock_type::now()},
      _last_draw{_start} {}

void ProgressBar::reset() noexcept
{
    _start                 = clock_type::now();
    _last_draw             = _start;
    _progress              = 0.0;
    _last_drawn_progress   = -1.0;
    _last_line_size        = 0u;
}

void ProgressBar::done() noexcept
{
    _progress = 1.0;
    draw(false);
    if (_interactive)
    {
        std::cerr << '\n';
    }
    std::cerr.flush();
}

void ProgressBar::cancel() noexcept
{
    draw(true);
    if (_interactive)
    {
        std::cerr << '\n';
    }
    std::cerr.flush();
}

void ProgressBar::update(double progress) noexcept
{
    using namespace std::chrono_literals;
    _progress = std::clamp(std::max(_progress, progress), 0.0, 1.0);
    auto now  = clock_type::now();
    if (_last_drawn_progress >= 0.0)
    {
        if (_interactive && now - _last_draw < 100ms && _progress < 1.0)
        {
            return;
        }
        if (!_interactive && _progress - _last_drawn_progress < 0.1 && _progress < 1.0)
        {
            return;
        }
    }
    draw(false);
}

void ProgressBar::draw(bool cancelled) noexcept
{
    using namespace std::chrono_literals;

    auto pos = static_cast<uint32_t>(_width * _progress);
    auto dt  = static_cast<double>((clock_type::now() - _start) / 1ns) * 1e-9;

    std::string bar;
    bar.reserve(_width);
    for (auto i = 0; i < _width; ++i)
    {
        if (i < pos)
        {
            bar.push_back(complete_char);
        }
        else if (i == pos)
        {
            bar.push_back(heading_char);
        }
        else
        {
            bar.push_back(incomplete_char);
        }
    }

    luisa::string line;
    if (cancelled)
    {
        line = luisa::format("[{}] {:5.1f}% | {:.1f}s | cancelled", bar, _progress * 100.0, dt);
    }
    else if (_progress > 0.0 && _progress < 1.0) [[likely]]
    {
        auto eta = (1.0 - _progress) / _progress * dt;
        line     = luisa::format("[{}] {:5.1f}% | {:.1f}s | ETA {:.1f}s", bar, _progress * 100.0, dt, eta);
    }
    else
    {
        line = luisa::format("[{}] {:5.1f}% | {:.1f}s", bar, _progress * 100.0, dt);
    }

    if (_interactive)
    {
        std::cerr << '\r' << line;
        if (_last_line_size > line.size())
        {
            std::cerr << std::string(_last_line_size - line.size(), ' ');
        }
        std::cerr.flush();
    }
    else
    {
        std::cerr << line << '\n';
    }

    _last_line_size      = line.size();
    _last_drawn_progress = _progress;
    _last_draw           = clock_type::now();
}

} // namespace Yutrel
