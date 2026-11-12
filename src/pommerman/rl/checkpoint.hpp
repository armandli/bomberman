#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <string>

namespace pommerman::rl {

// Model weights go to `path`, optimizer state to `path + ".optim"`. Both are
// written to a temp file and renamed so a kill mid-save never leaves a
// truncated checkpoint behind.
template <typename ModuleHolder>
void save_checkpoint(ModuleHolder& model, torch::optim::Optimizer& opt,
                     const std::string& path) {
    namespace fs = std::filesystem;
    const std::string tmp_model = path + ".tmp";
    const std::string tmp_optim = path + ".optim.tmp";
    torch::save(model, tmp_model);
    torch::save(opt, tmp_optim);
    fs::rename(tmp_model, path);
    fs::rename(tmp_optim, path + ".optim");
}

// Loads model (and optimizer state if present). The optimizer's learning rate
// is then reset to `lr`, so a resumed run picks up a changed --lr instead of
// the one stored in the checkpoint. Returns false if `path` does not exist.
template <typename ModuleHolder>
bool load_checkpoint(ModuleHolder& model, torch::optim::Optimizer& opt,
                     const std::string& path, double lr, torch::Device device) {
    namespace fs = std::filesystem;
    if (!fs::exists(path)) return false;
    torch::load(model, path, device);
    const std::string optim_path = path + ".optim";
    if (fs::exists(optim_path)) {
        try {
            torch::load(opt, optim_path, device);
        } catch (const std::exception& e) {
            std::cerr << "Warning: could not load optimizer state (" << e.what()
                      << "); starting with a fresh optimizer\n";
        }
    }
    for (auto& group : opt.param_groups())
        group.options().set_lr(lr);
    return true;
}

} // namespace pommerman::rl
