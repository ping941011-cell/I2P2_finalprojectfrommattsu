#pragma once
#include "base_state.hpp"
#include "search_params.hpp"
#include <vector>
#include <cstdint>
#include <functional>
#include <chrono>

class State;

struct RootUpdate {
    Move best_move;
    int score;
    int depth;
    int move_number;
    int total_moves;
};

struct SearchContext {
    uint64_t nodes = 0;
    int seldepth = 0;
    bool stop = false;
    ParamMap params;
    std::function<void(const RootUpdate&)> on_root_update;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();

    void reset(){
        nodes = 0;
        seldepth = 0;
    }
};

struct SearchResult {
    Move best_move;
    int score = 0;
    int depth = 0;
    int seldepth = 0;
    uint64_t nodes = 0;
    double time_ms = 0;
    std::vector<Move> pv;
};
