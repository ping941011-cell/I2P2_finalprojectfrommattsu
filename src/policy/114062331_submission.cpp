#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>
#include "state.hpp"
#include "114062331_submission.hpp"

/*============================================================
 * Static tables (MD7)
 * killer_moves[ply][0/1] : two killer moves per ply
 * history_table[player][fr][fc][tr][tc] : quiet-move bonus
 *============================================================*/
static const int SUB_MAX_PLY = 100;
static Move killer_moves[SUB_MAX_PLY][2];
static int  history_table[2][BOARD_H][BOARD_W][BOARD_H][BOARD_W];

/*============================================================
 * Transposition Table (MD11)
 * TTFlag: EXACT (precise) | LOWERBOUND (beta cutoff) | UPPERBOUND (all-node)
 *============================================================*/
enum TTFlag114 : uint8_t { TT114_EXACT = 0, TT114_LOWERBOUND = 1, TT114_UPPERBOUND = 2 };

struct TTEntry114 {
    uint64_t   key       = 0;
    int16_t    depth     = -1;
    int32_t    score     = 0;
    TTFlag114  flag      = TT114_EXACT;
    Move       best_move;
};

static const int TT114_SIZE = 1048583; // prime near 1M
static std::vector<TTEntry114> TT114(TT114_SIZE);

void Submission114::clear_tables(){
    std::memset(killer_moves,   0, sizeof(killer_moves));
    std::memset(history_table,  0, sizeof(history_table));
    std::fill(TT114.begin(), TT114.end(), TTEntry114{});
}

/*============================================================
 * score_move  (MD6 + MD7 combined)
 * Priority: captures (MVV-LVA) > killer[0] > killer[1] > history
 *============================================================*/
static int score_move(State* state, const Move& action, int ply){
    // 0. TT best move: highest priority (MD11)
    uint64_t hash_key = state->hash();
    TTEntry114& tte = TT114[hash_key % TT114_SIZE];
    if(tte.key == hash_key && tte.best_move == action) return 20000000;

    Point from = action.first,  to = action.second;
    int attacker = state->piece_at(    state->player, from.first, from.second);
    int victim   = state->piece_at(1 - state->player, to.first,  to.second);

    // 1. MVV-LVA captures
    if(victim != 0){
        return 10000000 + (100 * PIECE_VALUES[victim]) - PIECE_VALUES[attacker];
    }
    // 2. Killer heuristic
    int safe_ply = (ply < SUB_MAX_PLY) ? ply : SUB_MAX_PLY - 1;
    if(action == killer_moves[safe_ply][0]) return 9000000;
    if(action == killer_moves[safe_ply][1]) return 8000000;
    // 3. History heuristic
    return std::min(
        history_table[state->player][from.first][from.second][to.first][to.second],
        5000000
    );
}

/*============================================================
 * Submission114 — quiescence_search  (MD8)
 * Only searches captures; stand-pat provides a lower bound.
 *============================================================*/
int Submission114::quiescence_search(
    State* state, int alpha, int beta,
    GameHistory& history,
    SearchContext& ctx, const Sub114Params& p
){
    ctx.nodes++;
    if(ctx.stop) return 0;

    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    if(state->game_state == WIN)  return P_MAX;
    if(state->game_state == DRAW) return 0;

    // Stand-pat: assume doing nothing is at least this good
    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    if(stand_pat >= beta) return beta;
    if(alpha < stand_pat) alpha = stand_pat;

    // Collect captures only
    std::vector<Move> captures;
    captures.reserve(16);
    for(auto& action : state->legal_actions){
        if(state->piece_at(1 - state->player, action.second.first, action.second.second) != 0)
            captures.push_back(action);
    }
    if(captures.empty()) return alpha;

    // Sort captures by MVV-LVA
    std::sort(captures.begin(), captures.end(), [&](const auto& a, const auto& b){
        return score_move(state, a, 0) > score_move(state, b, 0);
    });

    for(auto& action : captures){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int next_alpha = same ? alpha : -beta;
        int next_beta  = same ? beta  : -alpha;
        int raw   = quiescence_search(next, next_alpha, next_beta, history, ctx, p);
        int score = same ? raw : -raw;
        delete next;

        if(score >= beta) return beta;
        if(score > alpha) alpha = score;
    }
    return alpha;
}

/*============================================================
 * Submission114 — eval_ctx  (MD5 + MD6 + MD7 + MD8 + MD10)
 *
 * History contract: the caller pushes state->hash() before
 * calling eval_ctx and pops it after.  eval_ctx itself pushes
 * each child's hash before recursing and pops after.
 *============================================================*/
int Submission114::eval_ctx(
    State* state, int depth, int alpha, int beta,
    GameHistory& history, int ply,
    SearchContext& ctx, const Sub114Params& p,
    bool allow_null
){
    ctx.nodes++;

    // Periodic time check (every 2048 nodes)
    if((ctx.nodes & 2047) == 0){
        if(std::chrono::steady_clock::now() >= ctx.deadline) ctx.stop = true;
    }
    if(ctx.stop) return 0;

    if(ply > ctx.seldepth) ctx.seldepth = ply;

    // --- Terminal / repetition checks (before generating moves) ---
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        // Contempt: treat draw as slightly worse than 0 to avoid repetition draws
        if(rep_score == 0) rep_score = -25;
        return rep_score;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    if(state->game_state == WIN)  return P_MAX  - ply;
    if(state->game_state == DRAW) return 0;
    if(state->legal_actions.empty()) return 0;

    // --- Quiescence at leaf ---
    if(depth <= 0)
        return quiescence_search(state, alpha, beta, history, ctx, p);

    // --- Transposition Table lookup (MD11) ---
    uint64_t hash_key     = state->hash();
    int      original_alpha = alpha;
    TTEntry114& tte = TT114[hash_key % TT114_SIZE];

    if(tte.key == hash_key && tte.depth >= depth){
        int tt_score = tte.score;
        if(tt_score > P_MAX - 1000) tt_score -= ply;
        if(tt_score < M_MAX + 1000) tt_score += ply;

        if(tte.flag == TT114_EXACT){
            return tt_score;
        } else if(tte.flag == TT114_LOWERBOUND){
            alpha = std::max(alpha, tt_score);
        } else if(tte.flag == TT114_UPPERBOUND){
            beta  = std::min(beta,  tt_score);
        }
        if(alpha >= beta) return tt_score;
    }

    // --- Null Move Pruning (MD10) ---
    // Adaptive R: 3 for deep, 2 for shallow
    int R = (depth >= 4) ? 3 : 2;
    if(allow_null && depth >= R + 1 && ply > 0){
        int stand_pat = state->evaluate(p.use_kp_eval, false, nullptr);
        if(stand_pat >= beta){
            State* null_state = static_cast<State*>(state->create_null_state());
            if(null_state != nullptr){
                // Pass allow_null=false to prevent recursive null moves
                history.push(null_state->hash());
                int null_score = -eval_ctx(
                    null_state, depth - 1 - R,
                    -beta, -beta + 1,
                    history, ply + 1, ctx, p, false
                );
                history.pop(null_state->hash());
                delete null_state;
                if(null_score >= beta) return beta;
            }
        }
    }

    // --- Move ordering (MD6 + MD7) ---
    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [&](const auto& a, const auto& b){
            return score_move(state, a, ply) > score_move(state, b, ply);
        }
    );

    // Futility pruning disabled — too aggressive; remove if it hurts tactical play

    int best_score = M_MAX;
    bool first_move = true;
    int  move_index = 0;

    // MD11: track best move for TT storage
    Move current_best_move;
    if(!state->legal_actions.empty())
        current_best_move = state->legal_actions[0];

    for(auto& action : state->legal_actions){
        int safe_ply = (ply < SUB_MAX_PLY) ? ply : SUB_MAX_PLY - 1;
        bool is_capture = state->piece_at(1 - state->player,
                              action.second.first, action.second.second) != 0;
        bool is_killer  = !is_capture
                          && (action == killer_moves[safe_ply][0]
                           || action == killer_moves[safe_ply][1]);

        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        history.push(next->hash());

        int score;
        if(first_move){
            // MD5/MD6: full window on first (expected best) move
            int na = same ? alpha : -beta;
            int nb = same ? beta  : -alpha;
            int raw = eval_ctx(next, depth - 1, na, nb, history, ply + 1, ctx, p);
            score = same ? raw : -raw;
        } else {
            // LMR: reduce late quiet moves (depth >= 4, index >= 4 — conservative)
            int reduction = 0;
            if(depth >= 4 && !is_capture && !is_killer && move_index >= 4){
                reduction = 1;
            }

            // MD6 PVS: zero-window probe at (possibly reduced) depth
            int na = same ? alpha : -(alpha + 1);
            int nb = same ? (alpha + 1) : -alpha;
            int raw = eval_ctx(next, depth - 1 - reduction, na, nb,
                               history, ply + 1, ctx, p);
            score = same ? raw : -raw;

            // LMR fail: re-search at full depth with zero window
            if(reduction > 0 && score > alpha){
                raw   = eval_ctx(next, depth - 1, na, nb, history, ply + 1, ctx, p);
                score = same ? raw : -raw;
            }

            // PVS fail: re-search with full window if probe beats alpha
            if(score > alpha && score < beta){
                int ra = same ? alpha : -beta;
                int rb = same ? beta  : -alpha;
                raw   = eval_ctx(next, depth - 1, ra, rb, history, ply + 1, ctx, p);
                score = same ? raw : -raw;
            }
        }

        history.pop(next->hash());
        delete next;

        if(ctx.stop) break;

        if(score > best_score){
            best_score        = score;
            current_best_move = action; // MD11: remember best move for TT
        }
        if(best_score > alpha) alpha = best_score;

        if(alpha >= beta){
            // MD7: update killer and history for quiet cutoff moves
            if(!is_capture){
                if(action != killer_moves[safe_ply][0]){
                    killer_moves[safe_ply][1] = killer_moves[safe_ply][0];
                    killer_moves[safe_ply][0] = action;
                }
                Point from = action.first, to = action.second;
                history_table[state->player]
                              [from.first][from.second]
                              [to.first][to.second] += depth * depth;
            }
            break;
        }
        first_move = false;
        move_index++;
    }

    // --- Transposition Table store (MD11) ---
    if(!ctx.stop){
        int store_score = best_score;
        if(store_score > P_MAX - 1000) store_score += ply;
        if(store_score < M_MAX + 1000) store_score -= ply;

        if(tte.key != hash_key || depth >= (int)tte.depth){
            tte.key       = hash_key;
            tte.depth     = (int16_t)depth;
            tte.score     = (int32_t)store_score;
            tte.best_move = current_best_move;

            if(best_score <= original_alpha) tte.flag = TT114_UPPERBOUND;
            else if(best_score >= beta)      tte.flag = TT114_LOWERBOUND;
            else                             tte.flag = TT114_EXACT;
        }
    }

    return best_score;
}

/*============================================================
 * Submission114 — search  (MD9: Iterative Deepening)
 *
 * Searches from depth 1 up to target_depth.
 * If interrupted (ctx.stop), returns the last fully-completed
 * depth result (never a partial result).
 * PV move ordering: best move from depth d is placed first for d+1.
 *============================================================*/
SearchResult Submission114::search(
    State* state, int target_depth,
    GameHistory& history, SearchContext& ctx
){
    ctx.reset();
    Sub114Params p = Sub114Params::from_map(ctx.params);
    SearchResult best_result;
    best_result.depth = 0;

    if(!state->legal_actions.size())
        state->get_legal_actions();

    if(state->legal_actions.empty()){
        best_result.best_move = Move();
        return best_result;
    }

    // Initial move ordering: MVV-LVA / killer / history at root (ply=1)
    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [&](const auto& a, const auto& b){
            return score_move(state, a, 1) > score_move(state, b, 1);
        }
    );

    Move current_best_move = state->legal_actions[0];

    for(int d = 1; d <= target_depth; d++){

        SearchResult cur;
        cur.depth = d;
        int best_score  = M_MAX - 10;
        int alpha       = M_MAX;
        int beta        = P_MAX;
        int move_index  = 0;
        int total_moves = (int)state->legal_actions.size();
        bool root_first = true;

        for(auto& action : state->legal_actions){
            State* child = state->next_state(action);
            bool same = child->same_player_as_parent();
            history.push(child->hash());

            int score;
            if(root_first){
                // Full window for first (PV) move
                int na = same ? alpha : -beta;
                int nb = same ? beta  : -alpha;
                int raw = eval_ctx(child, d - 1, na, nb, history, 1, ctx, p);
                score = same ? raw : -raw;
                root_first = false;
            } else {
                // PVS at root: zero-window probe
                int na = same ? alpha : -(alpha + 1);
                int nb = same ? (alpha + 1) : -alpha;
                int raw = eval_ctx(child, d - 1, na, nb, history, 1, ctx, p);
                score = same ? raw : -raw;
                // Re-search with full window if probe beats alpha
                if(score > alpha && score < beta){
                    int ra = same ? alpha : -beta;
                    int rb = same ? beta  : -alpha;
                    raw   = eval_ctx(child, d - 1, ra, rb, history, 1, ctx, p);
                    score = same ? raw : -raw;
                }
            }

            history.pop(child->hash());
            delete child;

            if(ctx.stop) break;

            if(score > best_score){
                best_score      = score;
                cur.score       = score;
                cur.best_move   = action;
                current_best_move = action;
                if(p.report_partial && ctx.on_root_update){
                    ctx.on_root_update({cur.best_move, best_score, d,
                                        move_index + 1, total_moves});
                }
            }
            if(best_score > alpha) alpha = best_score;
            move_index++;
        }

        // Discard incomplete depth result if interrupted
        if(ctx.stop) break;

        best_result          = cur;
        best_result.nodes    = ctx.nodes;
        best_result.seldepth = ctx.seldepth;
        best_result.pv       = {best_result.best_move};

        // PV ordering: move current best to front for next iteration (MD9)
        auto it = std::find(state->legal_actions.begin(),
                            state->legal_actions.end(), current_best_move);
        if(it != state->legal_actions.end())
            std::rotate(state->legal_actions.begin(), it, it + 1);

        // Early exit on forced win
        if(best_score > P_MAX - 1000) break;
    }

    return best_result;
}

/*============================================================
 * Registry interface
 *============================================================*/
ParamMap Submission114::default_params(){
    return {
        {"UseKPEval",       "true"},
        {"UseEvalMobility", "false"},
        {"ReportPartial",   "true"},
    };
}

std::vector<ParamDef> Submission114::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true",  0, 1},
        {"UseEvalMobility", ParamDef::CHECK, "false", 0, 1},
        {"ReportPartial",   ParamDef::CHECK, "true",  0, 1},
    };
}
