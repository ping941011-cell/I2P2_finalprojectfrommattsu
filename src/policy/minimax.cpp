#include <utility>
#include "state.hpp"
#include "minimax.hpp"
#include <algorithm>

enum TTFlag {
    TT_EXACT,
    TT_LOWERBOUND,
    TT_UPPERBOUND
};

struct TTEntry {
    uint64_t zobrist_key = 0;
    int depth = -1;
    int score = 0;
    TTFlag flag;
};

const int TT_SIZE = 1048576;
std::vector<TTEntry> tt(TT_SIZE);

/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax without pruning. Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    int original_alpha = alpha;
    uint64_t current_hash = state->hash();
    int tt_index = current_hash % TT_SIZE;

    if (tt[tt_index].zobrist_key == current_hash && tt[tt_index].depth >= depth) {
        if (tt[tt_index].flag == TT_EXACT) {
            return tt[tt_index].score;
        } else if (tt[tt_index].flag == TT_LOWERBOUND) {
            alpha = std::max(alpha, tt[tt_index].score);
        } else if (tt[tt_index].flag == TT_UPPERBOUND) {
            beta = std::min(beta, tt[tt_index].score);
        }

        if (alpha >= beta) {
            return tt[tt_index].score;
        }
    }


    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = quiescence(state, alpha, beta, ply, ctx, p); 
        history.pop(state->hash());
        return score;
    }

    std::sort(state->legal_actions.begin(), state->legal_actions.end(), [&state](const Move& a, const Move& b) {
        int a_attacker = state->piece_at(state->player, a.first.first, a.first.second);
        int a_victim = state->piece_at(1 - state->player, a.second.first, a.second.second);
        
        int b_attacker = state->piece_at(state->player, b.first.first, b.first.second);
        int b_victim = state->piece_at(1 - state->player, b.second.first, b.second.second);

        int score_a = (a_victim != 0) ? (100 * a_victim - a_attacker) : 0;
        int score_b = (b_victim != 0) ? (100 * b_victim - b_attacker) : 0;

        return score_a > score_b;
    });
    
    /* === Negamax loop === */
    int best_score = M_MAX;
    bool first_move = true;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        
        int raw;
        int score;

        if (first_move) {
            if (same) {
                raw = eval_ctx(next, depth, alpha, beta, history, ply + 1, ctx, p);
            } else {
                raw = eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
            }
            score = same ? raw : -raw;
            first_move = false;
            
        } else {
            if (same) {
                raw = eval_ctx(next, depth, alpha, alpha + 1, history, ply + 1, ctx, p);
            } else {
                raw = eval_ctx(next, depth - 1, -(alpha + 1), -alpha, history, ply + 1, ctx, p);
            }
            score = same ? raw : -raw;

            if (score > alpha && score < beta) {
                if (same) {
                    raw = eval_ctx(next, depth, score, beta, history, ply + 1, ctx, p);
                } else {
                    raw = eval_ctx(next, depth - 1, -beta, -score, history, ply + 1, ctx, p);
                }
                score = same ? raw : -raw;
            }
        }

        delete next;

        if (score > best_score) {
            best_score = score;
        }
        alpha = std::max(alpha, best_score);
        if (alpha >= beta) {
            break;
        }
    }

    TTFlag flag;
    if (best_score <= original_alpha) {
        flag = TT_UPPERBOUND;
    } else if (best_score >= beta) {
        flag = TT_LOWERBOUND;
    } else {
        flag = TT_EXACT;
    }

    tt[tt_index].zobrist_key = current_hash;
    tt[tt_index].depth = depth;
    tt[tt_index].score = best_score;
    tt[tt_index].flag = flag;

    history.pop(state->hash());
    return best_score;
}

/*============================================================
 * MiniMax — quiescence
 *============================================================*/
int MiniMax::quiescence(
    State *state,
    int alpha,
    int beta,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, nullptr);

    if (stand_pat >= beta) {
        return beta;
    }
    if (alpha < stand_pat) {
        alpha = stand_pat;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    std::sort(state->legal_actions.begin(), state->legal_actions.end(), [&state](const Move& a, const Move& b) {
        int a_attacker = state->piece_at(state->player, a.first.first, a.first.second);
        int a_victim = state->piece_at(1 - state->player, a.second.first, a.second.second);
        
        int b_attacker = state->piece_at(state->player, b.first.first, b.first.second);
        int b_victim = state->piece_at(1 - state->player, b.second.first, b.second.second);

        int score_a = (a_victim != 0) ? (100 * a_victim - a_attacker) : 0;
        int score_b = (b_victim != 0) ? (100 * b_victim - b_attacker) : 0;

        return score_a > score_b;
    });

    int best_score = stand_pat;

    for(auto& action : state->legal_actions){
        int to_r = action.second.first;
        int to_c = action.second.second;
        int opp_player = 1 - state->player;
        
        if (state->piece_at(opp_player, to_r, to_c) == 0) {
            continue;
        }

        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        
        int raw;
        if (same) {
            raw = quiescence(next, alpha, beta, ply + 1, ctx, p);
        } else {
            raw = quiescence(next, -beta, -alpha, ply + 1, ctx, p);
        }

        int score = same ? raw : -raw;
        delete next;

        if (score > best_score) {
            best_score = score;
        }
        alpha = std::max(alpha, best_score);
        if (alpha >= beta) {
            break;
        }
    }

    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    std::sort(state->legal_actions.begin(), state->legal_actions.end(), [&state](const Move& a, const Move& b) {
        int a_attacker = state->piece_at(state->player, a.first.first, a.first.second);
        int a_victim = state->piece_at(1 - state->player, a.second.first, a.second.second);
        
        int b_attacker = state->piece_at(state->player, b.first.first, b.first.second);
        int b_victim = state->piece_at(1 - state->player, b.second.first, b.second.second);

        int score_a = (a_victim != 0) ? (100 * a_victim - a_attacker) : 0;
        int score_b = (b_victim != 0) ? (100 * b_victim - b_attacker) : 0;

        return score_a > score_b;
    });

    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    int alpha = M_MAX;
    int beta = P_MAX;
    bool first_move = true;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        
        int raw;
        int score;

        if (first_move) {
            if (same) raw = eval_ctx(next, depth, alpha, beta, history, 1, ctx, p);
            else      raw = eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
            score = same ? raw : -raw;
            first_move = false;
        } else {
            if (same) raw = eval_ctx(next, depth, alpha, alpha + 1, history, 1, ctx, p);
            else      raw = eval_ctx(next, depth - 1, -(alpha + 1), -alpha, history, 1, ctx, p);
            score = same ? raw : -raw;

            if (score > alpha && score < beta) {
                if (same) raw = eval_ctx(next, depth, score, beta, history, 1, ctx, p);
                else      raw = eval_ctx(next, depth - 1, -beta, -score, history, 1, ctx, p);
                score = same ? raw : -raw;
            }
        }
        
        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }  
        alpha = std::max(alpha, best_score);
        move_index++;
    }

    result.score = best_score;
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;
    result.pv = { result.best_move }; // Store the best move as the principal variation

    return result;
} 


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
