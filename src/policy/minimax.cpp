#include <utility>
#include "state.hpp"
#include "minimax.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>

static int history_table[2][6][5][6][5] = {};

// Aspiration window: score and depth from the last completed search()
static int g_asp_score = 0;
static int g_asp_depth = 0;

enum TTFlag : uint8_t { TT_EXACT, TT_LOWERBOUND, TT_UPPERBOUND };

static inline uint16_t pack_move(const Move& m) {
    return (uint16_t)(((m.first.first & 7) << 9) | ((m.first.second & 7) << 6) |
                      ((m.second.first & 7) << 3) | (m.second.second & 7));
}
static inline Move unpack_move(uint16_t p) {
    return Move(Point((p >> 9) & 7, (p >> 6) & 7), Point((p >> 3) & 7, p & 7));
}

struct TTEntry {
    uint64_t zobrist_key = 0;
    int16_t  score       = 0;
    int8_t   depth       = -1;
    TTFlag   flag        = TT_EXACT;
    uint16_t move_packed = 0xFFFF;
};

const int TT_SIZE = 1048576;
std::vector<TTEntry> tt(TT_SIZE);

const int MAX_PLY = 128;
Move killer_moves[MAX_PLY][2];

void MiniMax::clear_tables(){
    std::fill(tt.begin(), tt.end(), TTEntry{});
    std::memset(killer_moves, 0, sizeof(killer_moves));
    std::memset(history_table, 0, sizeof(history_table));
    g_asp_score = 0;
    g_asp_depth = 0;
}

// Returns true if piece of type `piece` (player `player`) at (fr,fc) attacks (tr,tc).
// Checks the moved piece only — fast enough for per-node extension detection.
static bool piece_attacks_sq(
    int piece, int player,
    int fr, int fc, int tr, int tc,
    State* s
) {
    int dr = tr - fr, dc = tc - fc;
    switch (piece) {
        case 1: { int dir = (player == 0) ? -1 : 1; return (dr == dir) && (std::abs(dc) == 1); }
        case 3: { int a = std::abs(dr), b = std::abs(dc); return (a==2&&b==1)||(a==1&&b==2); }
        case 6: return std::abs(dr) <= 1 && std::abs(dc) <= 1 && (dr || dc);
        case 2: case 4: case 5: {
            bool straight = (dr == 0 || dc == 0) && (dr || dc);
            bool diag     = dr != 0 && std::abs(dr) == std::abs(dc);
            if (piece == 2 && !straight) return false;
            if (piece == 4 && !diag)     return false;
            if (piece == 5 && !straight && !diag) return false;
            int sr = dr == 0 ? 0 : (dr > 0 ? 1 : -1);
            int sc = dc == 0 ? 0 : (dc > 0 ? 1 : -1);
            int cr = fr + sr, cc = fc + sc;
            while (cr != tr || cc != tc) {
                if (s->board.board[0][cr][cc] || s->board.board[1][cr][cc]) return false;
                cr += sr; cc += sc;
            }
            return true;
        }
    }
    return false;
}

/*============================================================
 * MiniMax — eval_ctx
 *============================================================*/
int MiniMax::eval_ctx(
    State *state, int depth, int alpha, int beta,
    GameHistory& history, int ply, SearchContext& ctx, const MMParams& p, bool allow_null
){
    ctx.nodes++;

    if ((ctx.nodes & 2047) == 0) {
        if (std::chrono::steady_clock::now() >= ctx.deadline) ctx.stop = true;
    }

    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    int original_alpha = alpha;
    uint64_t current_hash = state->hash();
    int tt_index = current_hash % TT_SIZE;

    Move tt_move;
    bool has_tt_move = false;

    if (tt[tt_index].zobrist_key == current_hash) {
        if (tt[tt_index].move_packed != 0xFFFF) {
            tt_move = unpack_move(tt[tt_index].move_packed);
            has_tt_move = true;
        }
        if (tt[tt_index].depth >= depth) {
            int tt_score = tt[tt_index].score;
            if (tt_score > P_MAX - MAX_PLY) tt_score -= ply;
            else if (tt_score < M_MAX + MAX_PLY) tt_score += ply;

            if (tt[tt_index].flag == TT_EXACT)      return tt_score;
            else if (tt[tt_index].flag == TT_LOWERBOUND) alpha = std::max(alpha, tt_score);
            else if (tt[tt_index].flag == TT_UPPERBOUND) beta  = std::min(beta,  tt_score);

            if (alpha >= beta) return tt_score;
        }
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;

    history.push(state->hash());

    if(depth <= 0){
        int score = quiescence(state, alpha, beta, ply, ctx, p);
        history.pop(state->hash());
        return score;
    }

    // [1] NMP: use mobility=false to avoid create_null_state() inside evaluate()
    // [6] Adaptive R: deeper searches prune more aggressively
    int R = (depth >= 6) ? 3 : 2;
    if (allow_null && depth >= R + 1 && ply > 0) {
        int stand_pat = state->evaluate(p.use_kp_eval, false, nullptr);
        if (stand_pat >= beta) {
            State* null_state = static_cast<State*>(state->create_null_state());
            if (null_state != nullptr) {
                int null_score = -eval_ctx(null_state, depth - 1 - R, -beta, -beta + 1, history, ply + 1, ctx, p, false);
                delete null_state;
                if (null_score >= beta) {
                    history.pop(state->hash());
                    return beta;
                }
            }
        }
    }

    // Move ordering: precompute scores, insertion-sort
    const int n_moves = (int)state->legal_actions.size();
    int mscores[64];
    for (int i = 0; i < n_moves; i++) {
        const Move& m = state->legal_actions[i];
        const int att = state->piece_at(    state->player, m.first.first,  m.first.second);
        const int vic = state->piece_at(1 - state->player, m.second.first, m.second.second);
        if (has_tt_move && m == tt_move)                       mscores[i] = 3000000;
        else if (vic != 0)                                     mscores[i] = 1000000 + (PIECE_VALUES[vic] * 10 - PIECE_VALUES[att]);
        else if (ply < MAX_PLY && m == killer_moves[ply][0])   mscores[i] = 900000;
        else if (ply < MAX_PLY && m == killer_moves[ply][1])   mscores[i] = 800000;
        else mscores[i] = history_table[state->player][m.first.first][m.first.second][m.second.first][m.second.second];
    }
    for (int i = 1; i < n_moves; i++) {
        int ks = mscores[i]; Move km = state->legal_actions[i]; int j = i - 1;
        while (j >= 0 && mscores[j] < ks) {
            mscores[j+1] = mscores[j]; state->legal_actions[j+1] = state->legal_actions[j]; j--;
        }
        mscores[j+1] = ks; state->legal_actions[j+1] = km;
    }

    // [3] Futility pruning: at depth 1-2, skip quiet moves that can't raise alpha.
    // Margins in kp_material units (Rook=60, Knight=70, Bishop=80).
    const int FUTILITY_MARGIN[3] = {0, 100, 200};
    bool futility_prune = false;
    if (depth <= 2 && alpha > M_MAX + MAX_PLY && beta < P_MAX - MAX_PLY) {
        int feval = state->evaluate(p.use_kp_eval, false, nullptr);
        futility_prune = (feval + FUTILITY_MARGIN[depth] <= alpha);
    }

    int best_score = M_MAX;
    Move current_best_move = state->legal_actions[0];
    bool first_move = true;
    int move_count = 0;

    for(auto& action : state->legal_actions){
        move_count++;

        // Compute tactical flags once per move (simple array lookups)
        const int vic = state->piece_at(1 - state->player, action.second.first, action.second.second);
        const int mpc = state->piece_at(    state->player, action.first.first,  action.first.second);
        const bool is_capture = (vic != 0);
        const bool is_promo   = (mpc == 1 && (action.second.first == 0 || action.second.first == (int)BOARD_H - 1));

        // [3] Futility: skip quiet moves when static eval is far below alpha
        if (futility_prune && !first_move && !is_capture && !is_promo) continue;

        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        int raw, score;

        // Check extension: if moved piece directly attacks the opponent's king, search 1 ply deeper.
        int ext = 0;
        if (depth >= 2 && !same) {
            int to_r = action.second.first, to_c = action.second.second;
            int moved_piece = (int)next->board.board[state->player][to_r][to_c];
            int opp = 1 - state->player;
            for (int er = 0; er < BOARD_H && !ext; er++)
                for (int ec = 0; ec < BOARD_W && !ext; ec++)
                    if (next->board.board[opp][er][ec] == 6)
                        if (piece_attacks_sq(moved_piece, state->player, to_r, to_c, er, ec, next))
                            ext = 1;
        }

        // [6] LMR: reduce late quiet moves; never reduce when giving check
        const int reduction = (!first_move && depth >= 3 && move_count >= 4 && !is_capture && !is_promo && !ext) ? 1 : 0;

        if (first_move) {
            if (same) raw = eval_ctx(next, depth,             alpha,       beta,   history, ply + 1, ctx, p);
            else      raw = eval_ctx(next, depth - 1 + ext,   -beta,      -alpha,  history, ply + 1, ctx, p);
            score = same ? raw : -raw;
            first_move = false;
        } else {
            if (same) raw = eval_ctx(next, depth - reduction,              alpha,       alpha + 1, history, ply + 1, ctx, p);
            else      raw = eval_ctx(next, depth - 1 - reduction + ext, -(alpha + 1), -alpha,      history, ply + 1, ctx, p);
            score = same ? raw : -raw;

            // LMR fail-high: re-search at full depth
            if (reduction > 0 && score > alpha && !ctx.stop) {
                if (same) raw = eval_ctx(next, depth,            alpha,       alpha + 1, history, ply + 1, ctx, p);
                else      raw = eval_ctx(next, depth - 1 + ext, -(alpha + 1), -alpha,   history, ply + 1, ctx, p);
                score = same ? raw : -raw;
            }

            // PVS: zero-window raised alpha → full-window re-search
            if (score > alpha && score < beta && !ctx.stop) {
                if (same) raw = eval_ctx(next, depth,            score,  beta,   history, ply + 1, ctx, p);
                else      raw = eval_ctx(next, depth - 1 + ext,  -beta, -score,  history, ply + 1, ctx, p);
                score = same ? raw : -raw;
            }
        }
        delete next;

        if (score > best_score) {
            best_score = score;
            current_best_move = action;
        }
        alpha = std::max(alpha, best_score);

        if (alpha >= beta) {
            if (!is_capture) {
                if (ply < MAX_PLY) {
                    if (killer_moves[ply][0] != action) {
                        killer_moves[ply][1] = killer_moves[ply][0];
                        killer_moves[ply][0] = action;
                    }
                }
                int& hist = history_table[state->player][action.first.first][action.first.second][action.second.first][action.second.second];
                hist += depth * depth;
                if (hist > 500000) hist = 500000;
            }
            break;
        }
    }

    TTFlag flag;
    if (best_score <= original_alpha) flag = TT_UPPERBOUND;
    else if (best_score >= beta)      flag = TT_LOWERBOUND;
    else                              flag = TT_EXACT;

    int store_score = best_score;
    if (store_score > P_MAX - MAX_PLY) store_score += ply;
    else if (store_score < M_MAX + MAX_PLY) store_score -= ply;

    tt[tt_index].zobrist_key = current_hash;
    tt[tt_index].depth       = (int8_t)depth;
    tt[tt_index].score       = (int16_t)store_score;
    tt[tt_index].flag        = flag;
    tt[tt_index].move_packed = pack_move(current_best_move);

    history.pop(state->hash());
    return best_score;
}

/*============================================================
 * MiniMax — quiescence
 *============================================================*/
int MiniMax::quiescence(
    State *state, int alpha, int beta, int ply, SearchContext& ctx, const MMParams& p,
    bool allow_threat_ext
){
    ctx.nodes++;

    if ((ctx.nodes & 2047) == 0) {
        if (std::chrono::steady_clock::now() >= ctx.deadline) ctx.stop = true;
    }

    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    int opp = 1 - state->player;

    bool opp_promo_threat = false;
    if (allow_threat_ext) {
        int opp_promo_row = (opp == 0) ? 1 : (BOARD_H - 2);
        for(int c = 0; c < BOARD_W && !opp_promo_threat; c++){
            if(state->board.board[opp][opp_promo_row][c] == 1) opp_promo_threat = true;
        }
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();
    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    // [2] Use mobility=false to avoid create_null_state() at every leaf node
    int stand_pat = state->evaluate(p.use_kp_eval, false, nullptr);

    if (!opp_promo_threat) {
        if (stand_pat >= beta) return beta;
        if (alpha < stand_pat) alpha = stand_pat;
    }

    // [4] Collect only captures/promotions/(quiet extensions) and sort just those.
    //     Avoids std::sort over all ~20 moves when only ~3-5 are tactical.
    int  qscores[64];
    Move qmoves[64];
    bool q_is_quiet_ext[64];
    int  qn = 0;

    for (auto& action : state->legal_actions) {
        const int mpc = state->piece_at(state->player, action.first.first, action.first.second);
        const int vic = state->piece_at(opp, action.second.first, action.second.second);
        const bool is_promo    = (mpc == 1 && (action.second.first == 0 || action.second.first == (int)BOARD_H - 1));
        const bool is_quiet_ext = (opp_promo_threat && vic == 0 && !is_promo);

        if (vic == 0 && !is_promo && !is_quiet_ext) continue;

        qscores[qn]        = (vic != 0) ? (PIECE_VALUES[vic] * 10 - PIECE_VALUES[mpc]) : (is_promo ? 500 : 0);
        qmoves[qn]         = action;
        q_is_quiet_ext[qn] = is_quiet_ext;
        qn++;
    }
    // Insertion sort over captures only (usually 3-8 moves)
    for (int i = 1; i < qn; i++) {
        int ks = qscores[i]; Move km = qmoves[i]; bool ke = q_is_quiet_ext[i]; int j = i - 1;
        while (j >= 0 && qscores[j] < ks) {
            qscores[j+1] = qscores[j]; qmoves[j+1] = qmoves[j]; q_is_quiet_ext[j+1] = q_is_quiet_ext[j]; j--;
        }
        qscores[j+1] = ks; qmoves[j+1] = km; q_is_quiet_ext[j+1] = ke;
    }

    int best_score = stand_pat;

    for (int i = 0; i < qn; i++) {
        const Move& action = qmoves[i];
        bool child_allow_ext = !q_is_quiet_ext[i];

        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        int raw, score;

        if (same) raw = quiescence(next,  alpha,  beta, ply + 1, ctx, p, child_allow_ext);
        else      raw = quiescence(next, -beta, -alpha, ply + 1, ctx, p, child_allow_ext);
        score = same ? raw : -raw;
        delete next;

        if (score > best_score) best_score = score;
        alpha = std::max(alpha, best_score);
        if (alpha >= beta) break;
    }

    return best_score;
}

/*============================================================
 * MiniMax — search  (with Aspiration Windows)
 *============================================================*/
SearchResult MiniMax::search(
    State *state, int depth, GameHistory& history, SearchContext& ctx
){
    ctx.reset();

    // History aging: halve all values so stale moves from prior depths don't dominate ordering
    {
        int* ht = &history_table[0][0][0][0][0];
        for (int i = 0, n = 2*6*5*6*5; i < n; i++) ht[i] >>= 1;
    }

    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()) state->get_legal_actions();

    // Build root move order once — reused by all aspiration retries
    Move pv_move = state->legal_actions[0];
    {
        uint64_t root_hash = state->hash();
        int root_tt_idx = root_hash % TT_SIZE;
        if (tt[root_tt_idx].zobrist_key == root_hash && tt[root_tt_idx].move_packed != 0xFFFF)
            pv_move = unpack_move(tt[root_tt_idx].move_packed);
    }
    {
        const int n = (int)state->legal_actions.size();
        int mscores[64];
        for (int i = 0; i < n; i++) {
            const Move& m = state->legal_actions[i];
            const int att = state->piece_at(    state->player, m.first.first,  m.first.second);
            const int vic = state->piece_at(1 - state->player, m.second.first, m.second.second);
            if (m == pv_move)                          mscores[i] = 2000000;
            else if (vic != 0)                         mscores[i] = 1000000 + (PIECE_VALUES[vic] * 10 - PIECE_VALUES[att]);
            else if (m == killer_moves[1][0])          mscores[i] = 900000;
            else if (m == killer_moves[1][1])          mscores[i] = 800000;
            else mscores[i] = history_table[state->player][m.first.first][m.first.second][m.second.first][m.second.second];
        }
        for (int i = 1; i < n; i++) {
            int ks = mscores[i]; Move km = state->legal_actions[i]; int j = i - 1;
            while (j >= 0 && mscores[j] < ks) {
                mscores[j+1] = mscores[j]; state->legal_actions[j+1] = state->legal_actions[j]; j--;
            }
            mscores[j+1] = ks; state->legal_actions[j+1] = km;
        }
    }

    // [5] Aspiration window: narrow window around previous depth's score (depth >= 5 only)
    int asp_delta = 50;
    int asp_alpha = (depth >= 5 && g_asp_depth == depth - 1)
                    ? std::max(g_asp_score - asp_delta, (int)M_MAX) : (int)M_MAX;
    int asp_beta  = (depth >= 5 && g_asp_depth == depth - 1)
                    ? std::min(g_asp_score + asp_delta, (int)P_MAX) : (int)P_MAX;

    int best_score = M_MAX - 10;
    Move best_move = state->legal_actions[0];
    int asp_retry  = 0;

    while (true) {
        best_score = M_MAX - 10;
        bool first_move = true;
        int alpha = asp_alpha;
        int beta  = asp_beta;
        Move iter_best = state->legal_actions[0];

        for(auto& action : state->legal_actions){
            if (std::chrono::steady_clock::now() >= ctx.deadline) ctx.stop = true;
            if (ctx.stop) break;

            State* next = static_cast<State*>(state->next_state(action));
            bool same = next->same_player_as_parent();
            int raw, score;

            if (first_move) {
                if (same) raw = eval_ctx(next, depth,     alpha,       beta,   history, 1, ctx, p);
                else      raw = eval_ctx(next, depth - 1, -beta,      -alpha,  history, 1, ctx, p);
                score = same ? raw : -raw;
                first_move = false;
            } else {
                if (same) raw = eval_ctx(next, depth,     alpha,   alpha + 1, history, 1, ctx, p);
                else      raw = eval_ctx(next, depth - 1, -(alpha + 1), -alpha, history, 1, ctx, p);
                score = same ? raw : -raw;

                if (score > alpha && score < beta && !ctx.stop) {
                    if (same) raw = eval_ctx(next, depth,     score,  beta,   history, 1, ctx, p);
                    else      raw = eval_ctx(next, depth - 1, -beta, -score,  history, 1, ctx, p);
                    score = same ? raw : -raw;
                }
            }
            delete next;
            if (ctx.stop) break;

            if(score > best_score){ best_score = score; iter_best = action; }
            alpha = std::max(alpha, best_score);
        }

        // Accept result if this iteration completed something meaningful
        if (!first_move) best_move = iter_best;
        if (ctx.stop) break;

        // Check aspiration bounds
        bool fail_low  = (best_score <= asp_alpha) && (asp_alpha > (int)M_MAX);
        bool fail_high = (best_score >= asp_beta)  && (asp_beta  < (int)P_MAX);

        if (!fail_low && !fail_high) break;

        asp_retry++;
        if (asp_retry >= 3) {
            // Widen to full window for final retry
            asp_alpha = M_MAX;
            asp_beta  = P_MAX;
        } else if (fail_low) {
            asp_delta *= 2;
            asp_alpha = std::max(best_score - asp_delta, (int)M_MAX);
        } else {
            asp_delta *= 2;
            asp_beta = std::min(best_score + asp_delta, (int)P_MAX);
        }
    }

    g_asp_score = best_score;
    g_asp_depth = depth;

    result.score     = best_score;
    result.best_move = best_move;
    result.nodes     = ctx.nodes;
    result.seldepth  = ctx.seldepth;
    result.pv        = { best_move };

    return result;
}

ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval",       "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial",   "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial",   ParamDef::CHECK, "true"},
    };
}
