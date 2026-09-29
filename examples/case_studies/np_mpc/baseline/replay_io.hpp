#pragma once

// Shared by replay_laopt.cpp and replay_casadi.cpp: the instance file (JSON, read with yaml-cpp), the
// result file (JSON), and which IPOPT library the process actually loaded.
//
// Instance file:
//   {"method": "neural" | "equation", "horizon": 12, "dt": 0.02,
//    "mpc_config": "<mpc_config.yaml or mpc_config_equation.yaml, scripts/export_mpc_config.py format>",
//    "weights": "<np_weights.yaml, scripts/export_np_weights.py format; neural only>",
//    "instances": [{"x0": [4], "x_guess": [[4] x (N+1)], "u_guess": [[1] x N]}, ...]}
// Other keys (z, p, terminal_P, source, ...) are ignored.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <dlfcn.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#else
#include <link.h>
#endif

#include <yaml-cpp/yaml.h>

namespace replay {

struct Instance {
    std::vector<double> x0;                   // (NX)
    std::vector<std::vector<double>> xGuess;  // (N+1) x (NX)
    std::vector<double> uGuess;               // (N), NU = 1
};

struct InstanceFile {
    std::string method;
    int horizon{};
    double dt{};
    std::string mpcConfig;
    std::string weights;
    std::vector<Instance> instances;
};

inline InstanceFile loadInstances(const std::string& path, int nx, int n)
{
    const YAML::Node root = YAML::LoadFile(path);
    auto fail = [&](const std::string& msg) { throw std::runtime_error("loadInstances(" + path + "): " + msg); };
    InstanceFile f;
    f.method = root["method"].as<std::string>();
    f.horizon = root["horizon"].as<int>();
    f.dt = root["dt"].as<double>();
    f.mpcConfig = root["mpc_config"].as<std::string>();
    if (root["weights"] && !root["weights"].IsNull()) { f.weights = root["weights"].as<std::string>(); }
    if (f.horizon != n) { fail("horizon " + std::to_string(f.horizon) + ", compiled for N = " + std::to_string(n)); }
    for (const YAML::Node& node : root["instances"]) {
        Instance inst;
        inst.x0 = node["x0"].as<std::vector<double>>();
        inst.xGuess = node["x_guess"].as<std::vector<std::vector<double>>>();
        for (const YAML::Node& uk : node["u_guess"]) {  // [[u_0], [u_1], ...] or [u_0, u_1, ...]
            inst.uGuess.push_back(uk.IsSequence() ? uk[0].as<double>() : uk.as<double>());
        }
        if (inst.x0.size() != static_cast<size_t>(nx) || inst.xGuess.size() != static_cast<size_t>(n + 1) ||
            inst.uGuess.size() != static_cast<size_t>(n)) {
            fail("instance " + std::to_string(f.instances.size()) + " has the wrong shape");
        }
        for (const auto& xk : inst.xGuess) {
            if (xk.size() != static_cast<size_t>(nx)) { fail("x_guess rows must have " + std::to_string(nx) + " entries"); }
        }
        f.instances.push_back(std::move(inst));
    }
    if (f.instances.empty()) { fail("no instances"); }
    return f;
}

struct Step {
    double solveMs{};
    bool converged{};
    std::string status;
    int iter{-1};
    int qpIter{-1};        // SQP only
    double solverMs{-1.0}; // the solver's own wall time where it reports one (CasADi's t_wall_total)
    std::vector<std::vector<double>> x;  // (N+1) x (NX)
    std::vector<double> u;               // (N)
    std::vector<double> slack;           // (NX)
    double objective{};
};

// Top-level result; `fields` are extra key/value pairs, already JSON-encoded.
struct Run {
    std::vector<std::pair<std::string, std::string>> fields;
    std::vector<Step> steps;
};

inline std::string jsonString(const std::string& s)
{
    std::ostringstream out;
    out << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') { out << '\\' << c; }
        else if (c == '\n') { out << "\\n"; }
        else if (static_cast<unsigned char>(c) < 0x20) { out << ' '; }
        else { out << c; }
    }
    out << '"';
    return out.str();
}

inline std::string jsonNumber(double v)
{
    if (!std::isfinite(v)) { return "null"; }
    std::ostringstream out;
    out << std::setprecision(17) << v;
    return out.str();
}

inline std::string jsonArray(const std::vector<double>& v)
{
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) { s += (i ? ", " : "") + jsonNumber(v[i]); }
    return s + "]";
}

inline void writeRun(const std::string& path, const Run& run)
{
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    out << "{\n";
    for (const auto& [key, value] : run.fields) { out << "  " << jsonString(key) << ": " << value << ",\n"; }
    out << "  \"steps\": [";
    for (size_t k = 0; k < run.steps.size(); ++k) {
        const Step& s = run.steps[k];
        out << (k ? ",\n    " : "\n    ") << "{\"solve_ms\": " << jsonNumber(s.solveMs)
            << ", \"converged\": " << (s.converged ? "true" : "false") << ", \"status\": " << jsonString(s.status)
            << ", \"iter\": " << s.iter;
        if (s.qpIter >= 0) { out << ", \"qp_iter\": " << s.qpIter; }
        if (s.solverMs >= 0.0) { out << ", \"solver_ms\": " << jsonNumber(s.solverMs); }
        out << ", \"objective\": " << jsonNumber(s.objective) << ", \"u\": " << jsonArray(s.u)
            << ", \"slack\": " << jsonArray(s.slack) << ", \"x\": [";
        for (size_t i = 0; i < s.x.size(); ++i) { out << (i ? ", " : "") << jsonArray(s.x[i]); }
        out << "]}";
    }
    out << "\n  ]\n}\n";
}

// The IPOPT library loaded into this process: its path and version, or an empty path if none is
// loaded yet. Found among the loaded images by file name, so it also sees an IPOPT pulled in by a
// dlopen'ed plugin (CasADi's nlpsol_ipopt). The version comes from that library's GetIpoptVersion()
// where it has one (Scaly's 3.14.19 does; the casadi wheel's 3.14.11 does not), else, on macOS, from
// its LC_ID_DYLIB current_version: IPOPT 3.14.N installs libipopt with current_version 18.N.0.
struct LoadedIpopt {
    std::string path;
    std::string version;
};

inline LoadedIpopt loadedIpopt()
{
    struct Image {
        std::string path;
        std::string dylibVersion; // "3.14.N" derived from current_version 18.N.0, macOS only
    };
    std::vector<Image> images;
#if defined(__APPLE__)
    for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
        Image image{_dyld_get_image_name(i), ""};
        const auto* header = reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(i));
        if (header != nullptr && header->magic == MH_MAGIC_64) {
            const auto* cursor = reinterpret_cast<const uint8_t*>(header) + sizeof(mach_header_64);
            for (uint32_t c = 0; c < header->ncmds; ++c) {
                const auto* command = reinterpret_cast<const load_command*>(cursor);
                if (command->cmd == LC_ID_DYLIB) {
                    const uint32_t v = reinterpret_cast<const dylib_command*>(command)->dylib.current_version;
                    if ((v >> 16) == 18) { image.dylibVersion = "3.14." + std::to_string((v >> 8) & 0xff); }
                }
                cursor += command->cmdsize;
            }
        }
        images.push_back(std::move(image));
    }
#else
    dl_iterate_phdr([](dl_phdr_info* info, size_t, void* data) {
        static_cast<std::vector<Image>*>(data)->push_back({info->dlpi_name ? info->dlpi_name : "", ""});
        return 0;
    }, &images);
#endif
    for (const Image& image : images) {
        const std::string leaf = image.path.substr(image.path.find_last_of('/') + 1);
        if (leaf.rfind("libipopt", 0) != 0) { continue; }
        LoadedIpopt found{image.path, image.dylibVersion.empty() ? "?" : image.dylibVersion};
        if (void* handle = dlopen(image.path.c_str(), RTLD_LAZY | RTLD_NOLOAD)) {
            using GetIpoptVersionFn = void (*)(int*, int*, int*);
            if (auto fn = reinterpret_cast<GetIpoptVersionFn>(dlsym(handle, "GetIpoptVersion"))) {
                int major = 0, minor = 0, release = 0;
                fn(&major, &minor, &release);
                found.version = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(release);
            }
            dlclose(handle);
        }
        return found;
    }
    return {};
}

// Minimal "--key value" / "--flag" parsing.
struct Args {
    std::vector<std::string> argv;
    Args(int argc, char** argv_) : argv(argv_ + 1, argv_ + argc) {}
    bool flag(const std::string& name) const
    {
        for (const auto& a : argv) { if (a == name) { return true; } }
        return false;
    }
    std::string value(const std::string& name, const std::string& fallback = "") const
    {
        for (size_t i = 0; i + 1 < argv.size(); ++i) { if (argv[i] == name) { return argv[i + 1]; } }
        if (fallback.empty()) { throw std::invalid_argument("missing " + name); }
        return fallback;
    }
};

} // namespace replay
