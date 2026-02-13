function R = paper_power_eval(varargin)
% PAPER_POWER_EVAL
% One-click Monte-Carlo evaluation for paper figures/tables:
%   Compare:
%     (1) NN  (Transformer, TRF)
%     (2) Feasibility-search(Feasibility-search governor, nonn)   [optional]
%     (3) Baseline (Budget-agnostic governor, nolimit)
% By default, we include the feasibility-search nonn controller. To disable it:
%   opts.include_nonn_feasible = false;
% across payload mass only (trajectory fixed by PB.traj/cfg by default).
% NOTE (2026-01-21): By default, power metrics are computed on a moving-average filtered
% Pgrid signal ("red curve" in demo plots, default window opts.power_eval_win_s=0.50s). Set opts.power_eval_mode='raw' to use the
% instantaneous Pgrid ("blue curve").
%
% USAGE:
%   R = paper_power_eval();            % one-click: PB -> cfg -> ranges -> run
%   R = paper_power_eval(cfg);         % use provided cfg, ranges still from PB/opts
%   R = paper_power_eval(opts);        % cfg from PB, user options
%   R = paper_power_eval(cfg, opts);   % cfg + opts
%
% DEFAULTS (payload-only sweep):
%   - Trajectory is FIXED (q0/qf): opts.q_margin_deg = 0, opts.swap_prob = 0
%   - Power budget is FIXED:       opts.pmax_mode_eval = 'fixed'
% To restore trajectory randomization, set for example:
%   opts.q_margin_deg = 10;  opts.swap_prob = 0.5;
% To sample P_TOTAL_MAX across trials, set:
%   opts.pmax_mode_eval = 'sample';  opts.pmax_range_W = [Pmin Pmax];
%
% OUTPUT (struct R):
%   R.trials_table, R.summary_table, R.paired_stats_table
%   R.export_dir, R.q0_range_deg, R.qf_range_deg, R.mass_range
%
% Notes:
% - This script closes figures created by demos by default (good for large N).
% - Required demo functions:
%     spinn3d_demo_ctcgov_online_nn
%     spinn3d_demo_ctcgov_online_nonn_nolimit
%     spinn3d_demo_ctcgov_online_nonn  (optional third method)
% - Required PB->cfg helper (for one-click):
%     spinn3d_params_block
%     spinn3d_params_to_cfg

%% ---------------- Parse inputs ----------------
[cfg_base, opts] = parse_inputs(varargin{:});

%% ---------------- Required functions ----------------
needFns = {'spinn3d_demo_ctcgov_online_nn','spinn3d_demo_ctcgov_online_nonn_nolimit'};
% Optional third method: feasibility-search nonn controller
include_nonn_feasible = true;
if isfield(opts,'include_nonn_feasible')
    try, include_nonn_feasible = logical(opts.include_nonn_feasible); catch, end
end
if include_nonn_feasible
    needFns{end+1} = 'spinn3d_demo_ctcgov_online_nonn';
end
for k = 1:numel(needFns)
    if exist(needFns{k},'file') ~= 2
        error('paper_power_eval:MissingFunction', ...
            'Missing function: %s. Please add SPINN3D code to MATLAB path.', needFns{k});
    end
end

%% ---------------- Load PB (for ranges / sanity prints) ----------------
PB = [];
if exist('spinn3d_params_block','file')==2
    try
        PB = spinn3d_params_block();
    catch
        PB = [];
    end
end

%% ---------------- Generate sampling ranges HERE (no build_* functions) ----------------
nJ = numel(cfg_base.Q0_DEG(:));

% ---------- q0/qf ranges ----------
q0_range_deg = [];
qf_range_deg = [];

% Priority 1: explicit ranges from opts
if isfield(opts,'q0_range_deg') && ~isempty(opts.q0_range_deg) && ...
   isfield(opts,'qf_range_deg') && ~isempty(opts.qf_range_deg)
    q0_range_deg = opts.q0_range_deg;
    qf_range_deg = opts.qf_range_deg;

else
    % Do we have margin settings?
    hasMargin = false;
    if isfield(opts,'q_margin_deg')  && ~isempty(opts.q_margin_deg),  hasMargin = true; end
    if isfield(opts,'q0_margin_deg') && ~isempty(opts.q0_margin_deg), hasMargin = true; end
    if isfield(opts,'qf_margin_deg') && ~isempty(opts.qf_margin_deg), hasMargin = true; end

    if hasMargin
        % Center from PB.traj (preferred) else cfg
        q0c = [];
        qfc = [];
        if ~isempty(PB) && isfield(PB,'traj') && isstruct(PB.traj)
            if isfield(PB.traj,'q0_deg') && ~isempty(PB.traj.q0_deg), q0c = PB.traj.q0_deg(:); end
            if isfield(PB.traj,'qf_deg') && ~isempty(PB.traj.qf_deg), qfc = PB.traj.qf_deg(:); end
        end
        if isempty(q0c), q0c = cfg_base.Q0_DEG(:); end
        if isempty(qfc), qfc = cfg_base.QF_DEG(:); end

        % Margins (deg)
        m0 = get_opt_def(opts,'q0_margin_deg',[]);
        mf = get_opt_def(opts,'qf_margin_deg',[]);
        m  = get_opt_def(opts,'q_margin_deg',[]);
        if ~isempty(m), m0 = m; mf = m; end
        if isempty(m0), m0 = 10; end
        if isempty(mf), mf = 10; end

        % Expand scalar -> per-joint vector
        if isscalar(m0), m0 = repmat(m0, nJ, 1); else, m0 = m0(:); end
        if isscalar(mf), mf = repmat(mf, nJ, 1); else, mf = mf(:); end

        q0_range_deg = [q0c - m0, q0c + m0];
        qf_range_deg = [qfc - mf, qfc + mf];

    else
        % No margin specified: prefer PB.alpha_budget ranges directly (one-click)
        if ~isempty(PB) && isfield(PB,'alpha_budget') && isstruct(PB.alpha_budget) && ...
           isfield(PB.alpha_budget,'q0_range_deg') && isfield(PB.alpha_budget,'qref_range_deg') && ...
           ~isempty(PB.alpha_budget.q0_range_deg) && ~isempty(PB.alpha_budget.qref_range_deg)
            q0_range_deg = PB.alpha_budget.q0_range_deg;
            qf_range_deg = PB.alpha_budget.qref_range_deg;
        else
            % Fallback: cfg centers +- 10 deg
            q0c = cfg_base.Q0_DEG(:);
            qfc = cfg_base.QF_DEG(:);
            q0_range_deg = [q0c - 10, q0c + 10];
            qf_range_deg = [qfc - 10, qfc + 10];
        end
    end
end

% ---------- payload mass range ----------
mass_range = [];

% Priority 1: explicit from opts
if isfield(opts,'payload_mass_range') && ~isempty(opts.payload_mass_range)
    mass_range = double(opts.payload_mass_range(:).');

else
    % If user gives mass margin -> use PB.payload.demo.mass (center) +- margin
    dm = get_opt_def(opts,'mass_margin_kg',[]);
    if isempty(dm), dm = get_opt_def(opts,'payload_mass_margin',[]); end

    if ~isempty(dm)
        mc = [];
        if ~isempty(PB) && isfield(PB,'payload') && isstruct(PB.payload) && ...
           isfield(PB.payload,'demo') && isstruct(PB.payload.demo) && ...
           isfield(PB.payload.demo,'mass') && ~isempty(PB.payload.demo.mass)
            mc = PB.payload.demo.mass;
        end
        if isempty(mc)
            % fallback to cfg demo mass
            mc = 0.40;
            if isfield(cfg_base,'PAYLOAD') && isstruct(cfg_base.PAYLOAD) && ...
               isfield(cfg_base.PAYLOAD,'DEMO') && isstruct(cfg_base.PAYLOAD.DEMO) && ...
               isfield(cfg_base.PAYLOAD.DEMO,'mass') && ~isempty(cfg_base.PAYLOAD.DEMO.mass)
                mc = cfg_base.PAYLOAD.DEMO.mass;
            end
        end
        mass_range = [mc - dm, mc + dm];

    else
        % No mass margin specified: prefer PB.payload.range.mass (one-click)
        if ~isempty(PB) && isfield(PB,'payload') && isstruct(PB.payload) && ...
           isfield(PB.payload,'range') && isstruct(PB.payload.range) && ...
           isfield(PB.payload.range,'mass') && ~isempty(PB.payload.range.mass)
            mass_range = double(PB.payload.range.mass(:).');
        elseif isfield(cfg_base,'PAYLOAD') && isstruct(cfg_base.PAYLOAD) && ...
               isfield(cfg_base.PAYLOAD,'RANGE') && isstruct(cfg_base.PAYLOAD.RANGE) && ...
               isfield(cfg_base.PAYLOAD.RANGE,'mass') && ~isempty(cfg_base.PAYLOAD.RANGE.mass)
            mass_range = double(cfg_base.PAYLOAD.RANGE.mass(:).');
        else
            % fallback: demo mass +- 50%
            mc = 0.40;
            if isfield(cfg_base,'PAYLOAD') && isstruct(cfg_base.PAYLOAD) && ...
               isfield(cfg_base.PAYLOAD,'DEMO') && isstruct(cfg_base.PAYLOAD.DEMO) && ...
               isfield(cfg_base.PAYLOAD.DEMO,'mass') && ~isempty(cfg_base.PAYLOAD.DEMO.mass)
                mc = cfg_base.PAYLOAD.DEMO.mass;
            end
            mass_range = [max(0, 0.5*mc), 1.5*mc];
        end
    end
end

% Ensure sane shapes
q0_range_deg = ensure_range_mat(q0_range_deg, nJ);
qf_range_deg = ensure_range_mat(qf_range_deg, nJ);
mass_range   = ensure_range_vec(mass_range);

%% ---------------- Print ranges (so you know exactly what ran) ----------------
if opts.verbose
    fprintf('\n[paper_power_eval] Using ranges:\n');
    fprintf('  q0_range_deg (nJx2):\n'); disp(q0_range_deg);
    fprintf('  qf_range_deg (nJx2):\n'); disp(qf_range_deg);
    fprintf('  mass_range_kg (1x2): [%.4f, %.4f]\n\n', mass_range(1), mass_range(2));
end

%% ---------------- Export dir ----------------
ensure_dir(opts.export_dir);

%% ---------------- RNG ----------------
if ~isempty(opts.seed) && isfinite(opts.seed)
    rng(opts.seed);
end

%% ---------------- Evaluation settings ----------------
Pmax_nom = cfg_base.P_TOTAL_MAX;  % nominal budget (from PB.power.P_TOTAL_MAX)

% Resolve evaluation-time sampling for P_TOTAL_MAX (optional)
pmax_mode_eval = get_opt_def(opts,'pmax_mode_eval','fromPB');
try, pmax_mode_eval = lower(char(pmax_mode_eval)); catch, pmax_mode_eval = 'fromPB'; end

pmax_sample_side = get_opt_def(opts,'pmax_sample_side','tighten_only');
try, pmax_sample_side = lower(char(pmax_sample_side)); catch, pmax_sample_side = 'tighten_only'; end

pmax_range_W = get_opt_def(opts,'pmax_range_W',[]);
if isempty(pmax_range_W)
    if ~isempty(PB) && isfield(PB,'alpha_budget') && isstruct(PB.alpha_budget) && ...
       isfield(PB.alpha_budget,'pmax_range') && ~isempty(PB.alpha_budget.pmax_range)
        pmax_range_W = double(PB.alpha_budget.pmax_range(:).');
    else
        pmax_range_W = [Pmax_nom, Pmax_nom];
    end
end
pmax_range_W = ensure_range_vec(pmax_range_W);

% Map 'fromPB' -> PB.alpha_budget.pmax_mode if present, else fixed
if any(strcmp(pmax_mode_eval, {'frompb','pb','auto'}))
    if ~isempty(PB) && isfield(PB,'alpha_budget') && isstruct(PB.alpha_budget) && ...
       isfield(PB.alpha_budget,'pmax_mode') && ~isempty(PB.alpha_budget.pmax_mode)
        try
            pmax_mode_eval = lower(char(PB.alpha_budget.pmax_mode));
        catch
            pmax_mode_eval = 'fixed';
        end
    else
        pmax_mode_eval = 'fixed';
    end
end

if opts.verbose
    fprintf('[paper_power_eval] P_TOTAL_MAX eval: mode=%s, side=%s, range=[%.3f, %.3f] W, nominal=%.3f W, Pmax_margin_frac=%.3f\n', ...
        pmax_mode_eval, pmax_sample_side, pmax_range_W(1), pmax_range_W(2), Pmax_nom, get_opt_def(opts,'Pmax_margin_frac',0));
end


% Metric profile selection (see default_opts)
metric_profile = 'bus_smooth';
if isfield(opts,'metric_profile') && ~isempty(opts.metric_profile)
    try
        metric_profile = lower(char(opts.metric_profile));
    catch
        metric_profile = 'bus_smooth';
    end
end

if any(strcmp(metric_profile, {'legacy','ripple','detrended'}))
    % Legacy (more sensitive to high-frequency spikes)
    metricList = { ...
        't_reach', ...
        'P_peak','P_p99', ...
        'E_excess', ...
        'P_std_detrended', ...
        'ramp_rms','ramp_p95','ramp_max' ...
    };
else
    % Bus-smoothing profile: evaluate low-pass bus power smoothness (ignoring fast spikes)
    metricList = { ...
        't_reach', ...
        'P_peak','P_p99', ...
        'E_excess', ...
        'P_std','P_std_norm','P_rms_to_Pmax', ...
        'ramp_rms','ramp_p95' ...
    };
end

%% ---------------- Run N successful trials (with attempts cap) ----------------
N  = opts.N;
maxAttempts = max(N, round(N * opts.max_attempts_factor));

rows = cell(N,1);
ok = 0; attempts = 0; fail = 0;

% Representative trial logs for paper Fig. 4 (first successful trial)
out_rep_nn = [];
out_rep_np = [];
out_rep_nf = [];
Pmax_rep   = NaN;


tStartAll = tic;

while ok < N && attempts < maxAttempts
    attempts = attempts + 1;

    % --- sample q0/qf in degree box ---
    q0_deg = sample_box(q0_range_deg);
    qf_deg = sample_box(qf_range_deg);

    % optional swap (match augmentation style)
    if rand() < opts.swap_prob
        tmp = q0_deg; q0_deg = qf_deg; qf_deg = tmp;
    end

    % --- sample payload mass ---
    m = mass_range(1) + (mass_range(2)-mass_range(1))*rand();

    % --- construct cfg_i ---
    cfg_i = cfg_base;
    cfg_i.Q0_DEG = q0_deg(:);
    cfg_i.QF_DEG = qf_deg(:);

    % payload: override mass (optionally scale inertia)
    cfg_i = set_payload_mass(cfg_i, m, opts);

    % optional: zero noise to reduce randomness
    if opts.zero_noise
        if isfield(cfg_i,'NOISE') && isstruct(cfg_i.NOISE)
            if isfield(cfg_i.NOISE,'sigma_q'),  cfg_i.NOISE.sigma_q  = 0; end
            if isfield(cfg_i.NOISE,'sigma_dq'), cfg_i.NOISE.sigma_dq = 0; end
            if isfield(cfg_i.NOISE,'tau_std'),  cfg_i.NOISE.tau_std  = 0; end
        end
    end

        % --- sample power budget (P_TOTAL_MAX) for this trial ---
    Pmax_trial = Pmax_nom;
    if any(strcmp(pmax_mode_eval, {'sample','rand','random','uniform'}))
        lo = pmax_range_W(1);
        hi = pmax_range_W(2);

        % one-sided sampling: avoid splitting above/below nominal (per user request)
        if any(strcmp(pmax_sample_side, {'tighten','tighten_only','down','lower','lower_only'}))
            hi = min(hi, Pmax_nom);      % only tighter-or-equal budgets (more exceedance)
        elseif any(strcmp(pmax_sample_side, {'loosen','loosen_only','up','upper','upper_only'}))
            lo = max(lo, Pmax_nom);      % only looser-or-equal budgets
        end

        % fallback if invalid
        if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
            lo = min(pmax_range_W);
            hi = max(pmax_range_W);
        end
        Pmax_trial = lo + (hi-lo)*rand();
    end
    Pmax_trial = max(Pmax_trial, eps);
    cfg_i.P_TOTAL_MAX = Pmax_trial;

% --- run methods ---
    try
        tOne = tic;
        out_nn = run_demo_closefig(@spinn3d_demo_ctcgov_online_nn,           cfg_i, opts.close_figs);

        out_nf = [];
        if include_nonn_feasible
            out_nf = run_demo_closefig(@spinn3d_demo_ctcgov_online_nonn,     cfg_i, opts.close_figs);
        end

        out_np = run_demo_closefig(@spinn3d_demo_ctcgov_online_nonn_nolimit, cfg_i, opts.close_figs);

        % --- metrics ---
        Mnn = compute_power_metrics(out_nn, Pmax_trial, opts);
        Mnp = compute_power_metrics(out_np, Pmax_trial, opts);
        Mnf = [];
        if include_nonn_feasible
            Mnf = compute_power_metrics(out_nf, Pmax_trial, opts);
        end
        % --- pack row ---
        ok = ok + 1;
        row = struct();
        row.trial_id = ok;
        row.attempt_id = attempts;
        row.runtime_s = toc(tOne);

        % Store representative telemetry for paper Fig. 4
        if isempty(out_rep_nn) && isempty(out_rep_np) && isempty(out_rep_nf)
            out_rep_nn = out_nn;
            out_rep_np = out_np;
            if include_nonn_feasible
                out_rep_nf = out_nf;
            end
            Pmax_rep   = Pmax_trial;
        end


        for j=1:nJ
            row.(sprintf('q0_deg_j%d',j)) = q0_deg(j);
            row.(sprintf('qf_deg_j%d',j)) = qf_deg(j);
        end
        row.payload_mass_kg = m;
        row.P_total_max_W = Pmax_trial;
        row.P_thr_W = Pmax_trial * (1 + get_opt_def(opts,'Pmax_margin_frac',0));


        for mi=1:numel(metricList)
            fn = metricList{mi};
            v_nn = Mnn.(fn);
            v_np = Mnp.(fn);

            row.(sprintf('nn_%s',fn)) = v_nn;
            row.(sprintf('np_%s',fn)) = v_np;

            if include_nonn_feasible
                v_nf = Mnf.(fn);
                row.(sprintf('nf_%s',fn)) = v_nf;
            end

            % improvement vs baseline np: imp = np - nn (positive means TRF is better for "smaller-is-better" metrics)
            row.(sprintf('imp_%s',fn)) = v_np - v_nn;

            % percent reduction vs baseline np: 100*(1 - nn/np)
            row.(sprintf('imp_pct_%s',fn)) = safe_pct_reduction(v_np, v_nn);

            if include_nonn_feasible
                % baseline vs feasible-search nonn: np - nf
                row.(sprintf('imp_np_minus_nf_%s',fn)) = v_np - v_nf;
                row.(sprintf('imp_pct_np_minus_nf_%s',fn)) = safe_pct_reduction(v_np, v_nf);

                % feasible-search nonn vs TRF: nf - nn
                row.(sprintf('imp_nf_minus_nn_%s',fn)) = v_nf - v_nn;
                row.(sprintf('imp_pct_nf_minus_nn_%s',fn)) = safe_pct_reduction(v_nf, v_nn);
            end
        end

        rows{ok} = row;

        if opts.verbose
            fprintf('[OK %d/%d] attempt=%d  mass=%.3f kg  runtime=%.2fs\n', ok, N, attempts, m, row.runtime_s);
        end

    catch ME
        fail = fail + 1;
        if opts.verbose
            fprintf('[FAIL] attempt=%d  (%s)\n', attempts, ME.message);
        end
    end
end

if ok < N
    warning('paper_power_eval:PartialSuccess', ...
        'Only %d/%d trials succeeded (attempts=%d, fails=%d). Narrow ranges or increase max_attempts_factor.', ...
        ok, N, attempts, fail);
    rows = rows(1:ok);
else
    rows = rows(1:N);
end

T = struct2table(cell2mat(rows));

%% ---------------- Summary tables ----------------
S  = make_summary_table(T, metricList);
PS = make_paired_stats_table(T, metricList, opts);

%% ---------------- Export ----------------
tag = 'nn_vs_nolimit';
if include_nonn_feasible
    tag = 'nn_vs_nolimit_vs_nonn';
end

csv_trials   = fullfile(opts.export_dir, sprintf('trials_%s.csv', tag));
csv_summary  = fullfile(opts.export_dir, sprintf('summary_%s.csv', tag));
csv_paired   = fullfile(opts.export_dir, sprintf('paired_stats_%s.csv', tag));
writetable(T,  csv_trials);
writetable(S,  csv_summary);
writetable(PS, csv_paired);

save(fullfile(opts.export_dir, sprintf('results_%s_mc.mat', tag)), ...
    'T','S','PS','cfg_base','opts','q0_range_deg','qf_range_deg','mass_range');

%% ---------------- Plots (aggregate) ----------------
set_paper_style(opts);

% ---- Paper figures (Fig. 4–6) ----
% Fig. 4: representative time-domain telemetry (normalized raw+filtered power, and a_post)
if ~isempty(out_rep_nn) && ~isempty(out_rep_np) && isfinite(Pmax_rep)
    fig4 = plot_representative_timehist(out_rep_nn, out_rep_np, out_rep_nf, Pmax_rep, opts);
    save_figure(fig4, fullfile(opts.export_dir, 'Fig4_representative_timehist'), opts);
end

% Fig. 5: Monte-Carlo metric summary (mean ± std, paired trials)
fig5 = plot_mc_bar_meanstd(T, metricList, opts);
save_figure(fig5, fullfile(opts.export_dir, 'Fig5_mc_metrics_mean_std'), opts);

% Fig. 6: Paired trade-off scatter (Baseline -> TRF)
fig6 = plot_tradeoff_scatter(T, opts, 'np', 'nn');
save_figure(fig6, fullfile(opts.export_dir, 'Fig6_tradeoff_baseline_vs_trf'), opts);

% Optional: additional trade-off plots if feasible-search nonn is included
if include_nonn_feasible
    fig7 = plot_tradeoff_scatter(T, opts, 'np', 'nf');
    save_figure(fig7, fullfile(opts.export_dir, 'Fig7_tradeoff_baseline_vs_feasible'), opts);

    fig8 = plot_tradeoff_scatter(T, opts, 'nf', 'nn');
    save_figure(fig8, fullfile(opts.export_dir, 'Fig8_tradeoff_feasible_vs_trf'), opts);
end
totalRuntime = toc(tStartAll);

%% ---------------- Return ----------------
R = struct();
R.trials_table       = T;
R.summary_table      = S;
R.paired_stats_table = PS;
R.export_dir         = opts.export_dir;
R.total_runtime_s    = totalRuntime;
R.n_success          = height(T);
R.n_attempts         = attempts;
R.n_fail             = fail;

R.q0_range_deg       = q0_range_deg;
R.qf_range_deg       = qf_range_deg;
R.mass_range         = mass_range;
R.include_nonn_feasible = include_nonn_feasible;
R.tag = tag;

disp(['--- paper_power_eval finished (MC: ', tag, ') ---']);
disp(['Export dir: ', opts.export_dir]);
disp(['Trials CSV:  ', csv_trials]);
disp(['Summary CSV: ', csv_summary]);
disp(['Paired CSV:  ', csv_paired]);
fprintf('Success=%d  Attempts=%d  Fails=%d  TotalRuntime=%.1fs\n', ...
    R.n_success, R.n_attempts, R.n_fail, R.total_runtime_s);

end

%% ======================================================================
function [cfg, opts] = parse_inputs(varargin)
cfg  = [];
opts = [];

if nargin == 0
    cfg  = make_default_cfg();
    opts = default_opts();
    return;
end

if nargin == 1
    a1 = varargin{1};

    if isstruct(a1) && isfield(a1,'Q0_DEG') && isfield(a1,'QF_DEG')
        % treat as cfg
        cfg  = a1;
        opts = default_opts();
        return;
    end

    if isstruct(a1)
        % treat as opts
        cfg  = make_default_cfg();
        base = default_opts();
        opts = merge_struct(base, a1);
        return;
    end

    error('paper_power_eval:BadInputs', 'Single input must be cfg struct or opts struct.');
end

% nargin >= 2
cfg  = varargin{1};
opts = varargin{2};

if isempty(cfg),  cfg  = make_default_cfg(); end
if isempty(opts), opts = struct(); end

base = default_opts();
opts = merge_struct(base, opts);
end

function cfg = make_default_cfg()
if exist('spinn3d_params_block','file')==2 && exist('spinn3d_params_to_cfg','file')==2
    PB  = spinn3d_params_block();
    cfg = spinn3d_params_to_cfg(PB,'demo_alpha');
else
    error('paper_power_eval:CannotAutoBuildCFG', ...
        'Cannot auto-build cfg: missing spinn3d_params_block or spinn3d_params_to_cfg.');
end
end

function opts = default_opts()
opts = struct();

opts.N = 100;
opts.seed = 1;
opts.verbose = true;

% methods
opts.include_nonn_feasible = true;   % include feasibility-search nonn baseline (spinn3d_demo_ctcgov_online_nonn)

% sampling behavior
opts.swap_prob = 0.0;                 % DEFAULT: 0 -> do NOT swap start/goal (keep the same trajectory). Set >0 for augmentation.
opts.max_attempts_factor = 3.0;       % attempts cap = N * factor

% ranges (user overrides)
opts.q_margin_deg  = 0;   % DEFAULT: 0 -> do NOT randomize q0/qf (trajectory fixed to PB.traj/cfg). Set >0 to randomize in a box.
opts.q0_margin_deg = [];
opts.qf_margin_deg = [];
opts.q0_range_deg  = [];
opts.qf_range_deg  = [];

opts.mass_margin_kg      = []; % if set: center from PB.payload.demo.mass, range = center +- margin
opts.payload_mass_margin = []; % alias
opts.payload_mass_range  = [];

% payload inertia scaling (geometry-fixed approximation)
opts.scale_inertia_with_mass = true;

% metric profile:
%   'bus_smooth' (default): focus on smoothing of the low-pass bus power seen by the upstream supply
%   'legacy'     : keep older "ripple-focused" metrics (std of detrended power, ramp_max, etc.)
opts.metric_profile = 'bus_smooth';

% metrics options
opts.power_eval_mode  = 'filtered'; % 'filtered' (LPF / "red curve") or 'raw' (instantaneous / "blue curve")
opts.power_eval_win_s = 0.50;       % moving-average window [s] for filtered evaluation (bus smoothing view)
opts.detrend_win_s = 0.20;    % legacy only: window for std(detrended)

% Exceedance evaluation tolerance:
% For exceedance-related metric (E_excess), we use an effective threshold:
%   P_thr = Pmax * (1 + Pmax_margin_frac)
% Set to 0 to disable.
opts.Pmax_margin_frac = 0.05;  % 5%*Pmax tolerance (default)

% (Optional) Sample the power budget P_TOTAL_MAX across trials.
% DEFAULT: 'fixed' -> keep cfg.P_TOTAL_MAX (from PB.power.P_TOTAL_MAX).
% Set pmax_mode_eval='sample' to randomize within pmax_range_W (or PB.alpha_budget.pmax_range).
%   pmax_mode_eval    : 'fixed' (default), 'fromPB', or 'sample'
%   pmax_sample_side  : 'tighten_only' biases P_TOTAL_MAX <= nominal (more exceedance)
%   pmax_range_W      : override [Pmin Pmax] (W); if empty uses PB.alpha_budget.pmax_range
opts.pmax_mode_eval    = 'fixed';
opts.pmax_sample_side  = 'tighten_only';   % 'tighten_only'|'symmetric'|'loosen_only'
opts.pmax_range_W      = [];               % [Pmin Pmax] in W

opts.trim_lo_pct   = 1;
opts.trim_hi_pct   = 99;

% paired stats options
opts.bootstrap_B     = 2000;
opts.bootstrap_alpha = 0.05;

% runtime niceties
opts.close_figs = true;
opts.zero_noise = false;

% plot labels (used in paper figures)
opts.label_proposed = 'Look-ahead Transformer Gov';   % nn
opts.label_feasible = 'Feasibility-search Gov';       % nf (optional)
opts.label_baseline = 'Budget-agnostic Gov';          % np

% export
opts.export_dir = fullfile(pwd, 'paper_mc_payload_only_nn_vs_nolimit_vs_nonn');
opts.save_png   = true;
opts.save_pdf   = true;
opts.font_name  = 'Times New Roman';
opts.line_width = 1.6;
end

function out = merge_struct(a,b)
out = a;
if isempty(b), return; end
f = fieldnames(b);
for i=1:numel(f)
    out.(f{i}) = b.(f{i});
end
end

function v = get_opt_def(opts, fn, def)
if isstruct(opts) && isfield(opts,fn) && ~isempty(opts.(fn))
    v = opts.(fn);
else
    v = def;
end
end

%% ======================================================================
function R = ensure_range_mat(R, nJ)
R = double(R);
if size(R,1) ~= nJ || size(R,2) ~= 2
    error('paper_power_eval:BadRange', 'Range must be nJ×2 (nJ=%d).', nJ);
end
lo = min(R(:,1), R(:,2));
hi = max(R(:,1), R(:,2));
R = [lo, hi];
end

function r = ensure_range_vec(r)
r = double(r(:).');
if numel(r) ~= 2
    error('paper_power_eval:BadMassRange', 'Mass range must be 1×2.');
end
r = [min(r), max(r)];
r(1) = max(r(1), 0);
if ~isfinite(r(1)) || ~isfinite(r(2)) || r(2) <= r(1)
    error('paper_power_eval:BadMassRange', 'Mass range invalid.');
end
end

function x = sample_box(R)
x = R(:,1) + (R(:,2)-R(:,1)).*rand(size(R,1),1);
end

%% ======================================================================
function cfg = set_payload_mass(cfg, m, opts)
% Override cfg.PAYLOAD.DEMO.mass; optionally scale inertia linearly with mass
if ~isfield(cfg,'PAYLOAD') || ~isstruct(cfg.PAYLOAD)
    cfg.PAYLOAD = struct();
end
if ~isfield(cfg.PAYLOAD,'DEMO') || ~isstruct(cfg.PAYLOAD.DEMO) || isempty(cfg.PAYLOAD.DEMO)
    cfg.PAYLOAD.DEMO = struct('mass',0,'com',[0 0 0],'inertia',[0 0 0 0 0 0]);
end

p = cfg.PAYLOAD.DEMO;
base_m = getfield_def(p,'mass',0);

com = getfield_def(p,'com',[0 0 0]);
I6  = getfield_def(p,'inertia',[0 0 0 0 0 0]);
I6  = I6(:).';

if opts.scale_inertia_with_mass && isfinite(base_m) && base_m > 0
    ratio = m / base_m;
    I6 = ratio * I6;
end

p.mass    = m;
p.com     = com;
p.inertia = I6;

cfg.PAYLOAD.DEMO = p;
end

function x = getfield_def(S, fn, def)
if isstruct(S) && isfield(S,fn) && ~isempty(S.(fn))
    x = S.(fn);
else
    x = def;
end
end

%% ======================================================================
function out = run_demo_closefig(fhandle, cfg, doClose)
fig_before = findall(0,'Type','figure');
out = fhandle(cfg);
fig_after  = findall(0,'Type','figure');

if doClose
    new_figs = fig_after(~ismember(fig_after, fig_before));
    if ~isempty(new_figs)
        try, close(new_figs); catch, end %#ok<CTCH>
    end
end
end

%% ======================================================================
function M = compute_power_metrics(out, Pmax, opts)
if ~isfield(out,'log') || ~isfield(out.log,'t') || ~isfield(out.log,'Pgrid')
    error('paper_power_eval:BadOut', 'Output missing required fields: out.log.t and out.log.Pgrid');
end

t = out.log.t(:);
P_raw = out.log.Pgrid(:);
P = P_raw;  % may be replaced by filtered version below

if isfield(out,'t_reach') && ~isempty(out.t_reach) && isfinite(out.t_reach)
    t_reach = out.t_reach;
else
    t_reach = t(end);
end

dt = median(diff(t));
if ~isfinite(dt) || dt <= 0
    dt = (t(end)-t(1))/max(numel(t)-1,1);
end

% Optional evaluation filtering (to match the red curve in demo BUS power plots)
% When enabled, ALL power-based metrics (peak/p99/exceedance/ramp/etc.) are computed
% on a moving-average filtered Pgrid, intended to emulate bus capacitance / measurement filtering.
mode = 'raw';
if isfield(opts,'power_eval_mode') && ~isempty(opts.power_eval_mode)
    try
        mode = lower(char(opts.power_eval_mode));
    catch
        mode = 'raw';
    end
end
if any(strcmp(mode, {'filtered','avg','ma','movmean','red'}))
    win_s = 0.10;
    if isfield(opts,'power_eval_win_s') && ~isempty(opts.power_eval_win_s) && isfinite(opts.power_eval_win_s)
        win_s = double(opts.power_eval_win_s);
    end
    win = max(1, round(win_s / dt));
    P = movmean_fallback(P_raw, win);
else
    mode = 'raw';
    P = P_raw;
end

% detrend
win = max(3, round(opts.detrend_win_s / dt));
P_ma  = movmean_fallback(P, win);
P_det = P - P_ma;

% ramp-rate
ramp = diff(P) / dt;
ramp_abs = abs(ramp);

% Effective threshold for exceedance-related metrics (tolerance margin)
Pthr = Pmax;
mfrac = 0;
if isfield(opts,'Pmax_margin_frac') && ~isempty(opts.Pmax_margin_frac) && isfinite(opts.Pmax_margin_frac)
    mfrac = double(opts.Pmax_margin_frac);
    Pthr = Pmax * (1 + mfrac);
end

% compliance (time-based, robust to non-uniform dt)
dt_vec = diff(t);
if isempty(dt_vec)
    T_total = 0;
    T_excess = 0;
else
    T_total = sum(dt_vec);
    T_excess = sum(dt_vec .* double(P(1:end-1) > Pthr));
end
r_viol   = T_excess / max(T_total, eps);
E_excess = trapz(t, max(P - Pthr, 0));

% pack
M = struct();
M.Pmax_nom = Pmax;
M.Pthr = Pthr;
M.Pmax_margin_frac = mfrac;

M.t_reach = t_reach;
M.P_peak_raw = max(P_raw);
M.P_p99_raw  = pctile(P_raw, 99);
% Primary (evaluated) metrics are based on P (raw or filtered depending on opts.power_eval_mode)
M.P_peak  = max(P);
M.P_p99   = pctile(P, 99);

M.r_viol   = r_viol;
M.T_excess = T_excess;
M.E_excess = E_excess;

% Smoothness metrics of evaluated power signal P (raw or filtered):
M.P_std = std(P,'omitnan');
M.P_std_norm = M.P_std / max(Pmax, eps);
M.P_rms_to_Pmax = sqrt(mean((P - Pmax).^2,'omitnan'));

% Legacy ripple metric (high-pass residual)
M.P_std_detrended = std(P_det);

M.ramp_rms = sqrt(mean(ramp.^2,'omitnan'));
M.ramp_p95 = pctile(ramp_abs, 95);
M.ramp_max = max(ramp_abs);
end

function y = movmean_fallback(x, win)
x = x(:);
win = max(1, round(win));
if exist('movmean','file')==2
    y = movmean(x, win);
else
    k = ones(win,1)/win;
    y = conv(x, k, 'same');
end
end

function p = pctile(x, q)
x = x(isfinite(x));
x = sort(x(:));
if isempty(x), p = NaN; return; end
q = max(0,min(100,q));
if q == 0,   p = x(1);   return; end
if q == 100, p = x(end); return; end
pos = 1 + (numel(x)-1)*(q/100);
lo = floor(pos); hi = ceil(pos);
if lo == hi
    p = x(lo);
else
    p = x(lo) + (pos-lo)*(x(hi)-x(lo));
end
end

function S = safe_pct_reduction(base_val, new_val)
% 100*(1 - new/base). Positive means reduced (better if smaller-is-better).
if ~isfinite(base_val) || base_val == 0 || ~isfinite(new_val)
    S = NaN; return;
end
S = 100*(1 - new_val/base_val);
end

%% ======================================================================
function S = make_summary_table(T, metricList)
% Summary statistics per metric for all available methods.
% Always includes:
%   - nn_* : Transformer (TRF)
%   - np_* : budget-agnostic baseline (nolimit)
% Optionally includes (if present in T):
%   - nf_* : feasibility-search governor (nonn)

rows = {};
for mi=1:numel(metricList)
    m = metricList{mi};

    nn = T.(sprintf('nn_%s',m));
    np = T.(sprintf('np_%s',m));
    imp = T.(sprintf('imp_%s',m));          % np - nn
    imp_pct = T.(sprintf('imp_pct_%s',m));  % 100*(1 - nn/np)

    has_nf = any(strcmp(T.Properties.VariableNames, sprintf('nf_%s',m)));

    r = struct();
    r.metric = m;

    r.nn_mean = mean(nn,'omitnan');
    r.nn_std  = std(nn,'omitnan');
    r.nn_med  = pctile(nn,50);

    if has_nf
        nf = T.(sprintf('nf_%s',m));
        r.nf_mean = mean(nf,'omitnan');
        r.nf_std  = std(nf,'omitnan');
        r.nf_med  = pctile(nf,50);
    else
        r.nf_mean = NaN; r.nf_std = NaN; r.nf_med = NaN;
    end

    r.np_mean = mean(np,'omitnan');
    r.np_std  = std(np,'omitnan');
    r.np_med  = pctile(np,50);

    % Primary paired diff: baseline - TRF
    r.imp_mean = mean(imp,'omitnan');          % np - nn
    r.imp_std  = std(imp,'omitnan');
    r.imp_med  = pctile(imp,50);

    r.imp_pct_mean = mean(imp_pct,'omitnan');
    r.imp_pct_med  = pctile(imp_pct,50);

    if has_nf
        imp_np_nf = T.(sprintf('imp_np_minus_nf_%s',m));              % np - nf
        imp_pct_np_nf = T.(sprintf('imp_pct_np_minus_nf_%s',m));      % 100*(1 - nf/np)
        imp_nf_nn = T.(sprintf('imp_nf_minus_nn_%s',m));              % nf - nn
        imp_pct_nf_nn = T.(sprintf('imp_pct_nf_minus_nn_%s',m));      % 100*(1 - nn/nf)

        r.imp_np_minus_nf_mean = mean(imp_np_nf,'omitnan');
        r.imp_np_minus_nf_std  = std(imp_np_nf,'omitnan');
        r.imp_np_minus_nf_med  = pctile(imp_np_nf,50);

        r.imp_pct_np_minus_nf_mean = mean(imp_pct_np_nf,'omitnan');
        r.imp_pct_np_minus_nf_med  = pctile(imp_pct_np_nf,50);

        r.imp_nf_minus_nn_mean = mean(imp_nf_nn,'omitnan');
        r.imp_nf_minus_nn_std  = std(imp_nf_nn,'omitnan');
        r.imp_nf_minus_nn_med  = pctile(imp_nf_nn,50);

        r.imp_pct_nf_minus_nn_mean = mean(imp_pct_nf_nn,'omitnan');
        r.imp_pct_nf_minus_nn_med  = pctile(imp_pct_nf_nn,50);
    else
        r.imp_np_minus_nf_mean = NaN; r.imp_np_minus_nf_std = NaN; r.imp_np_minus_nf_med = NaN;
        r.imp_pct_np_minus_nf_mean = NaN; r.imp_pct_np_minus_nf_med = NaN;
        r.imp_nf_minus_nn_mean = NaN; r.imp_nf_minus_nn_std = NaN; r.imp_nf_minus_nn_med = NaN;
        r.imp_pct_nf_minus_nn_mean = NaN; r.imp_pct_nf_minus_nn_med = NaN;
    end

    rows{end+1,1} = r; %#ok<AGROW>
end
S = struct2table(cell2mat(rows));
end

%% ======================================================================
function PS = make_paired_stats_table(T, metricList, opts)
% Paired-trial statistics.
%
% Primary comparison (always present):
%   imp = np - nn   (baseline - TRF)
%
% Optional additional comparisons (if feasible-search nonn is present):
%   imp_np_minus_nf = np - nf   (baseline - feasible)
%   imp_nf_minus_nn = nf - nn   (feasible - TRF)

rows = {};
for mi=1:numel(metricList)
    m = metricList{mi};

    % Primary: baseline - TRF
    imp = T.(sprintf('imp_%s',m));   % np - nn
    st1 = paired_stats_from_imp(imp, opts);

    r = struct();
    r.metric = m;

    % Keep legacy column names for primary comparison
    r.n     = st1.n;
    r.mean  = st1.mean;
    r.std   = st1.std;
    r.ci_lo = st1.ci_lo;
    r.ci_hi = st1.ci_hi;
    r.frac_nn_better = st1.frac_better;   % fraction where (np - nn) > 0
    r.p_sign_2s      = st1.p_sign_2s;

    % Optional: include feasible-search nonn if present
    has_nf = any(strcmp(T.Properties.VariableNames, sprintf('imp_np_minus_nf_%s',m))) && ...
             any(strcmp(T.Properties.VariableNames, sprintf('imp_nf_minus_nn_%s',m)));

    if has_nf
        imp_np_nf = T.(sprintf('imp_np_minus_nf_%s',m));   % np - nf
        imp_nf_nn = T.(sprintf('imp_nf_minus_nn_%s',m));   % nf - nn

        st2 = paired_stats_from_imp(imp_np_nf, opts);
        st3 = paired_stats_from_imp(imp_nf_nn, opts);

        % baseline - feasible
        r.n_np_minus_nf        = st2.n;
        r.mean_np_minus_nf     = st2.mean;
        r.std_np_minus_nf      = st2.std;
        r.ci_lo_np_minus_nf    = st2.ci_lo;
        r.ci_hi_np_minus_nf    = st2.ci_hi;
        r.frac_nf_better       = st2.frac_better;    % fraction where (np - nf) > 0
        r.p_sign_2s_np_minus_nf= st2.p_sign_2s;

        % feasible - TRF
        r.n_nf_minus_nn        = st3.n;
        r.mean_nf_minus_nn     = st3.mean;
        r.std_nf_minus_nn      = st3.std;
        r.ci_lo_nf_minus_nn    = st3.ci_lo;
        r.ci_hi_nf_minus_nn    = st3.ci_hi;
        r.frac_nn_better_vs_nf = st3.frac_better;    % fraction where (nf - nn) > 0
        r.p_sign_2s_nf_minus_nn= st3.p_sign_2s;

    else
        r.n_np_minus_nf        = NaN;
        r.mean_np_minus_nf     = NaN;
        r.std_np_minus_nf      = NaN;
        r.ci_lo_np_minus_nf    = NaN;
        r.ci_hi_np_minus_nf    = NaN;
        r.frac_nf_better       = NaN;
        r.p_sign_2s_np_minus_nf= NaN;

        r.n_nf_minus_nn        = NaN;
        r.mean_nf_minus_nn     = NaN;
        r.std_nf_minus_nn      = NaN;
        r.ci_lo_nf_minus_nn    = NaN;
        r.ci_hi_nf_minus_nn    = NaN;
        r.frac_nn_better_vs_nf = NaN;
        r.p_sign_2s_nf_minus_nn= NaN;
    end

    rows{end+1,1} = r; %#ok<AGROW>
end
PS = struct2table(cell2mat(rows));
end

function st = paired_stats_from_imp(imp, opts)
imp = imp(:);
imp = imp(isfinite(imp));

st = struct();
st.n = numel(imp);

if isempty(imp)
    st.mean = NaN; st.std = NaN;
    st.ci_lo = NaN; st.ci_hi = NaN;
    st.frac_better = NaN;
    st.p_sign_2s = NaN;
    return;
end

st.mean = mean(imp);
st.std  = std(imp);

[st.ci_lo, st.ci_hi] = bootstrap_ci_mean(imp, opts.bootstrap_B, opts.bootstrap_alpha);
st.frac_better = mean(imp > 0);
st.p_sign_2s = sign_test_p_two_sided(imp);
end


function [ci_lo, ci_hi] = bootstrap_ci_mean(x, B, alpha)
x = x(:);
n = numel(x);
B = max(200, round(B));
alpha = max(1e-6, min(0.5, alpha));

bm = zeros(B,1);
for b=1:B
    idx = randi(n, [n,1]);
    bm(b) = mean(x(idx));
end
ci_lo = pctile(bm, 100*(alpha/2));
ci_hi = pctile(bm, 100*(1-alpha/2));
end

function p = sign_test_p_two_sided(x)
x = x(:);
x = x(isfinite(x));
x = x(x ~= 0);
n = numel(x);
if n == 0
    p = 1;
    return;
end
k = sum(x > 0);
p_lower = binom_cdf(k, n, 0.5);
p_upper = 1 - binom_cdf(k-1, n, 0.5);
p = 2 * min(p_lower, p_upper);
p = min(max(p,0),1);
end

function F = binom_cdf(k, n, p0)
if k < 0
    F = 0; return;
end
if k >= n
    F = 1; return;
end
F = betainc(1-p0, n-k, k+1);
end

%% ======================================================================
function set_paper_style(opts)
set(groot,'defaultFigureColor','w');
set(groot,'defaultAxesFontName',opts.font_name);
set(groot,'defaultTextFontName',opts.font_name);
set(groot,'defaultAxesLineWidth',1.0);
set(groot,'defaultLineLineWidth',opts.line_width);
end

function ensure_dir(d)
if exist(d,'dir')~=7, mkdir(d); end
end

function save_figure(fig, path_no_ext, opts)
if opts.save_pdf
    try
        exportgraphics(fig, [path_no_ext,'.pdf'], 'ContentType','vector');
    catch
        print(fig, [path_no_ext,'.pdf'], '-dpdf', '-painters');
    end
end
if opts.save_png
    try
        exportgraphics(fig, [path_no_ext,'.png'], 'Resolution', 300);
    catch
        print(fig, [path_no_ext,'.png'], '-dpng', '-r300');
    end
end
end

%% ======================================================================
function fig = plot_representative_timehist(out_nn, out_np, out_nf, Pmax, opts)
% Representative servo-rate telemetry (time histories)
%  - Normalized DC-bus input power: \tilde{P}(t) = P_grid(t) / Pmax
%  - Show both raw and moving-average filtered signals
%  - Show applied outer-loop progress rate a_post (logged as log.alpha)
%
% Methods:
%   nn : Transformer (TRF)
%   nf : feasibility-search nonn (optional; pass [] to skip)
%   np : budget-agnostic baseline (nolimit)

check_out_log(out_nn, 'out_nn');
check_out_log(out_np, 'out_np');

has_nf = ~isempty(out_nf);
if has_nf
    check_out_log(out_nf, 'out_nf');
end

t_nn = out_nn.log.t(:);   P_nn = out_nn.log.Pgrid(:);
t_np = out_np.log.t(:);   P_np = out_np.log.Pgrid(:);

t_nf = []; P_nf = [];
if has_nf
    t_nf = out_nf.log.t(:);
    P_nf = out_nf.log.Pgrid(:);
end

% Optional: applied progress-rate (a_post)
a_nn = []; a_np = []; a_nf = [];
if isfield(out_nn.log,'alpha'), a_nn = out_nn.log.alpha(:); end
if isfield(out_np.log,'alpha'), a_np = out_np.log.alpha(:); end
if has_nf && isfield(out_nf.log,'alpha'), a_nf = out_nf.log.alpha(:); end

% Moving-average filter window (match evaluation window)
dt = median(diff(t_nn));
if ~isfinite(dt) || dt <= 0
    dt = (t_nn(end)-t_nn(1))/max(numel(t_nn)-1,1);
end
win_s = get_opt_def(opts,'power_eval_win_s',0.50);
win = max(1, round(win_s / max(dt, eps)));

P_nn_f = movmean_fallback(P_nn, win);
P_np_f = movmean_fallback(P_np, win);
if has_nf
    P_nf_f = movmean_fallback(P_nf, win);
end

% Normalize by Pmax
Pmax = max(double(Pmax), eps);
P_nn_n  = P_nn   / Pmax;
P_np_n  = P_np   / Pmax;
P_nn_fn = P_nn_f / Pmax;
P_np_fn = P_np_f / Pmax;

if has_nf
    P_nf_n  = P_nf   / Pmax;
    P_nf_fn = P_nf_f / Pmax;
end

% Threshold lines
delta = get_opt_def(opts,'Pmax_margin_frac',0.05);
y_budget = 1;
y_thr   = 1 + double(delta);

tmin = min([t_nn(1), t_np(1)]);
tmax = max([t_nn(end), t_np(end)]);
if has_nf
    tmin = min([tmin, t_nf(1)]);
    tmax = max([tmax, t_nf(end)]);
end

fig = figure('Name','Representative telemetry','Color','w');
useTL = exist('tiledlayout','file')==2 && exist('nexttile','file')==2;
if useTL
    tlo = tiledlayout(fig,2,1,'Padding','compact','TileSpacing','compact');
    ax1 = nexttile(tlo,1);
    ax2 = nexttile(tlo,2);
else
    ax1 = subplot(2,1,1,'Parent',fig);
    ax2 = subplot(2,1,2,'Parent',fig);
end

% --- Power panel ---
hold(ax1,'on'); grid(ax1,'on');

plot(ax1, t_nn, P_nn_n,  '--', 'DisplayName','TRF raw');
plot(ax1, t_nn, P_nn_fn, '-',  'DisplayName','TRF filtered');

if has_nf
    plot(ax1, t_nf, P_nf_n,  '--', 'DisplayName','feasible raw');
    plot(ax1, t_nf, P_nf_fn, '-',  'DisplayName','feasible filtered');
end

plot(ax1, t_np, P_np_n,  '--', 'DisplayName','Baseline raw');
plot(ax1, t_np, P_np_fn, '-',  'DisplayName','Baseline filtered');

plot(ax1, [tmin tmax], [y_budget y_budget], ':', 'DisplayName','Budget');
plot(ax1, [tmin tmax], [y_thr   y_thr],   ':', 'DisplayName','Threshold');

xlim(ax1,[tmin tmax]);
xlabel(ax1,'t (s)');
ylabel(ax1,'Normalized power $\tilde{P}_{\mathrm{grid}}$', 'Interpreter','latex');
title(ax1,'DC-bus input power (normalized)');
legend(ax1,'Location','best');

% --- Progress-rate panel ---
hold(ax2,'on'); grid(ax2,'on');

if ~isempty(a_nn)
    plot(ax2, t_nn, a_nn, '-', 'DisplayName','TRF a_{post}');
end
if has_nf && ~isempty(a_nf)
    plot(ax2, t_nf, a_nf, '-', 'DisplayName','Feasible a_{post}');
end
if ~isempty(a_np)
    plot(ax2, t_np, a_np, '-', 'DisplayName','Baseline a_{post}');
end

xlim(ax2,[tmin tmax]);
xlabel(ax2,'t (s)');
ylabel(ax2,'a');
title(ax2,'Outer-loop applied progress rate');
legend(ax2,'Location','best');
end

function check_out_log(out, name)
if ~isfield(out,'log') || ~isstruct(out.log)
    error('paper_power_eval:BadOut', '%s missing out.log struct.', name);
end
if ~isfield(out.log,'t') || ~isfield(out.log,'Pgrid')
    error('paper_power_eval:BadOut', '%s.log must contain t and Pgrid', name);
end
end

function fig = plot_mc_bar_meanstd(T, metricList, opts)
% Fig. 5: Monte-Carlo metric summary (mean ± std) for paired trials
% Default: normalize power-based metrics by the trial power budget P_total_max_W:
%   \tilde{P} = P / Pmax,  \tilde{E}_{excess} = E_excess / Pmax (unit: s),
%   \widetilde{ramp} = ramp / Pmax (unit: 1/s).

% Choose what to show in the compact 2×3 summary figure
metric_profile = 'bus_smooth';
if isfield(opts,'metric_profile') && ~isempty(opts.metric_profile)
    try, metric_profile = lower(char(opts.metric_profile)); catch, end
end

doNorm = get_opt_def(opts,'plot_norm_power', true);
Pmax_vec = [];
try
    if any(strcmp(T.Properties.VariableNames,'P_total_max_W'))
        Pmax_vec = T.P_total_max_W(:);
    end
catch
    Pmax_vec = [];
end

if any(strcmp(metric_profile, {'legacy','ripple','detrended'}))
    keys = {'t_reach','P_peak','P_p99','E_excess','P_std_detrended','ramp_rms'};
    if doNorm && ~isempty(Pmax_vec)
        titles = {'$t_{\mathrm{reach}}$ (s)','$\tilde{P}_{\mathrm{peak}}$','$\tilde{P}_{\mathrm{p99}}$','$\tilde{E}_{\mathrm{excess}}$ (s)','$\mathrm{Std}(\tilde{P})$','$\mathrm{RMS}(|\mathrm{d}\tilde{P}/\mathrm{d}t|)$ (1/s)'};
    else
        titles = {'$t_{\mathrm{reach}}$ (s)','$P_{\mathrm{peak}}$','$P_{\mathrm{p99}}$','$E_{\mathrm{excess}}\,(\mathrm{W\cdot s})$','Std (detrended)','$\mathrm{RMS}(|\mathrm{d}P/\mathrm{d}t|)\,(\mathrm{W/s})$'};
    end
else
    keys = {'t_reach','P_peak','P_p99','E_excess','P_std','ramp_p95'};
    if doNorm && ~isempty(Pmax_vec)
        titles = {'$t_{\mathrm{reach}}$ (s)','$\tilde{P}_{\mathrm{peak}}$','$\tilde{P}_{\mathrm{p99}}$','$\tilde{E}_{\mathrm{excess}}$ (s)','$\mathrm{Std}(\tilde{P})$','$p_{95}(|\mathrm{d}\tilde{P}/\mathrm{d}t|)$ (1/s)'};
    else
        titles = {'$t_{\mathrm{reach}}$ (s)','$P_{\mathrm{peak}}$','$P_{\mathrm{p99}}$','$E_{\mathrm{excess}}\,(\mathrm{W\cdot s})$','Std (LP bus power)','$p_{95}(|\mathrm{d}P/\mathrm{d}t|)\,(\mathrm{W/s})$'};
    end
end

fig = figure('Name','Fig5 MC metrics','Color','w');
% Make figure size deterministic for export (helps avoid tick-label overlap)
try
    set(fig,'Units','inches','Position',[1 1 7.2 3.9]);
catch
end
useTL = exist('tiledlayout','file')==2 && exist('nexttile','file')==2;
if useTL
    tlo = tiledlayout(fig,2,3,'Padding','compact','TileSpacing','compact');
end

for i=1:numel(keys)
    m = keys{i};
    nn = T.(sprintf('nn_%s',m));
    np = T.(sprintf('np_%s',m));
    has_nf = any(strcmp(T.Properties.VariableNames, sprintf('nf_%s',m)));
    nf = [];
    if has_nf
        nf = T.(sprintf('nf_%s',m));
    end

    % Normalize selected power-based metrics by trial Pmax
    if doNorm && ~isempty(Pmax_vec)
        den = max(Pmax_vec, eps);

        do_div = false;
        if any(strcmp(m, {'P_peak','P_p99','P_std','P_std_detrended'}))
            do_div = true;
        elseif strcmp(m, 'E_excess')
            do_div = true;
        elseif ~isempty(strfind(m,'ramp'))
            do_div = true;
        end

        if do_div
            nn = nn ./ den;
            np = np ./ den;
            if has_nf
                nf = nf ./ den;
            end
        end
    end

    if has_nf
        mu = [mean(nn,'omitnan'), mean(nf,'omitnan'), mean(np,'omitnan')];
        sd = [std(nn,'omitnan'),  std(nf,'omitnan'),  std(np,'omitnan')];
        xtlbl = {'TRF','Feasible','Baseline'};
    else
        mu = [mean(nn,'omitnan'), mean(np,'omitnan')];
        sd = [std(nn,'omitnan'),  std(np,'omitnan')];
        xtlbl = {'TRF','Baseline'};
    end

    if useTL
        ax = nexttile(tlo, i);
    else
        ax = subplot(2,3,i,'Parent',fig);
    end
    hold(ax,'on'); grid(ax,'on');

    % Bars: use grayscale for publication (TRF: dark gray, Baseline: light gray)
    try
        b = bar(ax, mu, 'FaceColor','flat');
    catch
        b = bar(ax, mu);
    end
    try
        if numel(mu) == 3
            b.CData = [0.35 0.35 0.35; 0.65 0.65 0.65; 0.85 0.85 0.85];
        else
            b.CData = [0.45 0.45 0.45; 0.85 0.85 0.85];
        end
        b.EdgeColor = [0 0 0];
    catch
        % fallback (older MATLAB): keep default face color
    end

    er = errorbar(ax, 1:numel(mu), mu, sd, '.');
    try
        er.Color = [0 0 0];
        er.LineStyle = 'none';
    catch
    end

    % Tick labels: keep readable in compact tiles (avoid overlap)
rot = 0;
fs  = 10;
if numel(mu) >= 3
    rot = 20;
    fs  = 9;
end
set(ax,'XTick',1:numel(mu),'XTickLabel',xtlbl,'XTickLabelRotation',rot);
try, ax.FontSize = fs; catch, end
    title(ax, titles{i}, 'Interpreter','latex');
end
end

function fig = plot_tradeoff_scatter(T, opts, pref_base, pref_other)
% Paired trade-off scatter (t_reach vs exceedance) for two methods.
%
% Column prefixes in T:
%   nn_* : Transformer (TRF)
%   nf_* : feasibility-search nonn (optional)
%   np_* : budget-agnostic baseline (nolimit)
%
% Usage:
%   fig = plot_tradeoff_scatter(T, opts);              % default: np vs nn
%   fig = plot_tradeoff_scatter(T, opts, 'np','nf');   % baseline vs feasible
%   fig = plot_tradeoff_scatter(T, opts, 'nf','nn');   % feasible vs TRF

if nargin < 3 || isempty(pref_base),  pref_base  = 'np'; end
if nargin < 4 || isempty(pref_other), pref_other = 'nn'; end

fig = figure('Name','Trade-off scatter','Color','w');
ax = axes(fig); hold(ax,'on'); grid(ax,'on');

% Pull paired data
x_other = T.(sprintf('%s_t_reach',pref_other)); y_other = T.(sprintf('%s_E_excess',pref_other));
x_base  = T.(sprintf('%s_t_reach',pref_base));  y_base  = T.(sprintf('%s_E_excess',pref_base));

% Optional normalization by trial P_total_max_W
doNorm = get_opt_def(opts,'plot_norm_power', true);
ylab = '$E_{\mathrm{excess}}$ (W$\cdot$s)';
if doNorm
    try
        if any(strcmp(T.Properties.VariableNames,'P_total_max_W'))
            den = max(T.P_total_max_W(:), eps);
            y_other = y_other ./ den;
            y_base  = y_base  ./ den;
            ylab = '$\tilde{E}_{\mathrm{excess}}$ (s)';
        end
    catch
    end
end

% Labels
lbl_base  = tradeoff_label_from_prefix(opts, pref_base);
lbl_other = tradeoff_label_from_prefix(opts, pref_other);

% Colors (consistent across all figures)
c_base  = tradeoff_color_from_prefix(opts, pref_base);
c_other = tradeoff_color_from_prefix(opts, pref_other);

% Optional paired lines (Base -> Other) for each trial
doPair = get_opt_def(opts,'plot_tradeoff_paired_lines', true);
if doPair
    pairColor = get_opt_def(opts,'plot_tradeoff_pair_color', [0.85 0.85 0.85]);
    pairLW    = get_opt_def(opts,'plot_tradeoff_pair_linewidth', 0.75);
    for k=1:numel(x_other)
        if isfinite(x_other(k)) && isfinite(x_base(k)) && isfinite(y_other(k)) && isfinite(y_base(k))
            plot(ax, [x_base(k), x_other(k)], [y_base(k), y_other(k)], '-', ...
                'Color', pairColor, 'LineWidth', pairLW, 'HandleVisibility','off');
        end
    end
    % Single legend handle for paired segments
    plot(ax, [NaN NaN], [NaN NaN], '-', 'Color', pairColor, ...
         'LineWidth', pairLW, 'DisplayName','Paired trials');
end

% Scatter points (base: open circle, other: filled circle)
ms = get_opt_def(opts,'plot_tradeoff_marker_size', 18);          % scatter marker area (pt^2)
lw = get_opt_def(opts,'plot_tradeoff_marker_linewidth', 0.9);

s_base  = scatter(ax, x_base,  y_base,  ms, 'o', 'DisplayName', lbl_base);
s_other = scatter(ax, x_other, y_other, ms, 'o', 'filled', 'DisplayName', lbl_other);

try
    % Base: open circle in its color
    s_base.MarkerFaceColor = 'none';
    s_base.MarkerEdgeColor = c_base;
    s_base.LineWidth = lw;

    % Other: filled circle in its color
    s_other.MarkerFaceColor = c_other;
    s_other.MarkerEdgeColor = c_other;
    s_other.LineWidth = lw;

    % Optional alpha (newer MATLAB)
    if isprop(s_other,'MarkerFaceAlpha')
        s_base.MarkerEdgeAlpha  = get_opt_def(opts,'plot_tradeoff_marker_edge_alpha', 0.95);
        s_other.MarkerFaceAlpha = get_opt_def(opts,'plot_tradeoff_marker_face_alpha', 0.85);
        s_other.MarkerEdgeAlpha = get_opt_def(opts,'plot_tradeoff_marker_edge_alpha', 0.95);
    end
catch
end

xlabel(ax,'$t_{\mathrm{reach}}$ (s)','Interpreter','latex');
ylabel(ax, ylab, 'Interpreter','latex');
% --- Disable axis exponent (×10^n) to avoid overlapping with title ---
try
    ax.YAxis.Exponent = 0;      % Newer MATLAB
catch
    try
        ax.YRuler.Exponent = 0; % Older MATLAB
    catch
    end
end
title(ax, sprintf('Paired trade-off: %s vs %s', lbl_base, lbl_other), 'Interpreter','none');

% Legend placement: keep Fig. 6 as-is (NW), move Fig. 7/8 to NE to avoid covering data
legLoc = 'northwest';
try
    if ~(strcmpi(pref_base,'np') && strcmpi(pref_other,'nn'))
        legLoc = 'northeast';
    end
catch
end
leg = legend(ax,'Location',legLoc,'Box','off');
try, leg.FontSize = 10; catch, end
end
function lbl = tradeoff_label_from_prefix(opts, pref)
% Map method prefix -> plot label (customizable via opts)
lbl = char(pref);
try
    pref = lower(char(pref));
catch
    return;
end

if strcmp(pref,'np')
    lbl = get_opt_def(opts,'label_baseline','Budget-agnostic Gov');
elseif strcmp(pref,'nn')
    lbl = get_opt_def(opts,'label_proposed','Look-ahead Transformer Gov');
elseif strcmp(pref,'nf')
    lbl = get_opt_def(opts,'label_Feasible','Feasibility-search Gov');
end
end

function c = tradeoff_color_from_prefix(opts, pref)
% Map method prefix -> RGB color (customizable via opts).
% Defaults use MATLAB's standard color order (good contrast):
%   np (baseline):   blue
%   nf (feasible):   green
%   nn (TRF):        orange
c = [0 0 0];
try
    pref = lower(char(pref));
catch
    return;
end

if strcmp(pref,'np')
    c = get_opt_def(opts,'color_baseline', [0.0000 0.4470 0.7410]);
elseif strcmp(pref,'nf')
    c = get_opt_def(opts,'color_Feasible', [0.4660 0.6740 0.1880]);
elseif strcmp(pref,'nn')
    c = get_opt_def(opts,'color_proposed', [0.8500 0.3250 0.0980]);
else
    % Fallback: gray
    c = [0.25 0.25 0.25];
end
end

