#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

struct Sub114Params {
    bool use_kp_eval       = true;
    bool use_eval_mobility = false;
    bool report_partial    = true;

    static Sub114Params from_map(const ParamMap& m){
        Sub114Params p;
        p.use_kp_eval       = param_bool(m, "UseKPEval",       true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", false);
        p.report_partial    = param_bool(m, "ReportPartial",   true);
        return p;
    }
};

class Submission114 {
public:
    /* MD5+MD6+MD7+MD8+MD10: alpha-beta / PVS / killer / history / QS / NMP */
    static int eval_ctx(
        State*          state,
        int             depth,
        int             alpha,
        int             beta,
        GameHistory&    history,
        int             ply,
        SearchContext&  ctx,
        const Sub114Params& p,
        bool            allow_null = true
    );

    /* MD8: quiescence search */
    static int quiescence_search(
        State*          state,
        int             alpha,
        int             beta,
        GameHistory&    history,
        SearchContext&  ctx,
        const Sub114Params& p
    );

    /* MD9: iterative deepening entry point */
    static SearchResult search(
        State*          state,
        int             depth,
        GameHistory&    history,
        SearchContext&  ctx
    );

    static void                   clear_tables();
    static ParamMap               default_params();
    static std::vector<ParamDef>  param_defs();
};
