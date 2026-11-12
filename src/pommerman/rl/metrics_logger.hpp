#pragma once
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>

namespace pommerman::rl {

// Appends one JSON object per line ("JSON Lines") to a metrics file, flushed
// after every record so external monitors (scripts/rl.py) see it immediately.
// Non-finite numbers are written as NaN / Infinity / -Infinity, which Python's
// json module parses.
class MetricsLogger {
public:
    using Field = std::pair<const char*, double>;

    MetricsLogger() = default;
    explicit MetricsLogger(const std::string& path) { open(path); }

    void open(const std::string& path) {
        if (path.empty()) return;
        out_.open(path, std::ios::app);
        start_ = std::chrono::steady_clock::now();
    }
    bool enabled() const { return out_.is_open(); }

    // kind: "train", "eval", "event", ...
    void log(const char* kind, std::initializer_list<Field> fields) {
        if (!enabled()) return;
        double t = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_).count();
        std::ostringstream os;
        os.precision(7);
        os << "{\"kind\":\"" << kind << "\",\"wall\":" << t;
        for (auto& [k, v] : fields) {
            os << ",\"" << k << "\":";
            if (std::isnan(v))      os << "NaN";
            else if (std::isinf(v)) os << (v > 0 ? "Infinity" : "-Infinity");
            else                    os << v;
        }
        os << "}\n";
        out_ << os.str();
        out_.flush();
    }

private:
    std::ofstream out_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

// ── Graceful stop ────────────────────────────────────────────────────────────
// SIGINT / SIGTERM set a flag; training loops poll stop_requested(), break,
// and save a final checkpoint.
namespace detail {
inline volatile std::sig_atomic_t g_stop = 0;
inline void on_stop_signal(int) { g_stop = 1; }
} // namespace detail

inline void install_stop_handler() {
    std::signal(SIGINT,  detail::on_stop_signal);
    std::signal(SIGTERM, detail::on_stop_signal);
}
inline bool stop_requested() { return detail::g_stop != 0; }

} // namespace pommerman::rl
