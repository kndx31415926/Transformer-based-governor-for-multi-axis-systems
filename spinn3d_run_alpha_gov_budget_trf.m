function out = spinn3d_run_alpha_gov_budget_trf()
% 生成 look-ahead 序列数据集 → 分片写盘 → 主表聚合 → 训练 Transformer
%
% 与 spinn3d_run_alpha_gov_budget 保持同一口径：
%   - 同一 robot / caps / BUS 功率口径
%   - Pmax 与 payload：逐轨迹采样；单条轨迹内保持不变
%   - teacher：alpha_feasible + alpha_des_teacher（governor-aware）
%   - 在线 NN 不加硬约束，只在训练期加 PINN/soft regularization

assert(exist('spinn3d_params_block','file')==2, '缺少 spinn3d_params_block.m');
PB = spinn3d_params_block();

% ===== 校验 =====
req(PB, {'paths','sample','power','alpha','caps','fric','alpha_budget','time','payload','train_trf'});
req(PB.paths, {'root','model_alpha'});
req(PB.sample, {'dt'});
req(PB.time,   {'Ts'});
req(PB.alpha,  {'alpha_min','itmax'});
req(PB.power,  {'P_TOTAL_MAX'});
req(PB.fric,   {'B','Fc','vel_eps'});
req(PB.alpha_budget, {'n_traj','chunk_traj','pmax_mode','pmax_range', ...
                      'q0_range_deg','qref_range_deg', ...
                      'dls_lambda','dls_w_ori','bez_lift_z','bez_gamma','ns_min'});
req(PB.payload, {'range'});
req(PB.train_trf, {'enable','K','H','epochs','batch','lr'});

N_TRAJ     = double(PB.alpha_budget.n_traj);
CHUNK_TRAJ = double(PB.alpha_budget.chunk_traj);
DT         = double(PB.sample.dt);
TS         = double(PB.time.Ts);
ALPHA_MIN  = double(PB.alpha.alpha_min);
ITMAX      = double(PB.alpha.itmax);

% caps + 总功率预算 + 可选 BUS 口径
caps0 = PB.caps; caps0.P_total_max = double(PB.power.P_TOTAL_MAX);
if isfield(PB,'bus') && isstruct(PB.bus)
    if isfield(PB.bus,'eta_share') && ~isempty(PB.bus.eta_share), caps0.eta_share=double(PB.bus.eta_share); end
    if isfield(PB.bus,'P_dump_max') && ~isempty(PB.bus.P_dump_max), caps0.P_dump_max=double(PB.bus.P_dump_max); end
    if isfield(PB.bus,'P_brk_peak') && ~isempty(PB.bus.P_brk_peak), caps0.P_brk_peak=double(PB.bus.P_brk_peak); end
end

% governor 参数
areg = PB.alpha_reg;   % 要求 PB.alpha_reg 完整
req(areg, {'beta','a_dot_up','a_dot_dn'});

% Robot
robot = resolve_robot_from_PB(PB); try, robot.DataFormat='row'; end

% 输出路径
SAVE_DIR = char(PB.paths.root);
if ~exist(SAVE_DIR,'dir'), mkdir(SAVE_DIR); end
chunk_pat = fullfile(SAVE_DIR, 'alpha_gov_seq_ds_chunk_*.mat');
existing  = dir(chunk_pat);
used_ids  = parse_ids({existing.name}, 'alpha_gov_seq_ds_chunk_(\d+)\.mat');
if isempty(used_ids)
    next_id = 1;
else
    next_id = max(used_ids) + 1;
end

fprintf('[alpha-gov-trf] Total %d traj, chunk=%d, out=%s | Ts=%.6f | K=%d, H=%.3g\n', ...
        N_TRAJ, CHUNK_TRAJ, SAVE_DIR, TS, PB.train_trf.K, PB.train_trf.H);

% ===== 分片生成 =====
left    = N_TRAJ;
nChunks = ceil(N_TRAJ / CHUNK_TRAJ);

for c = 1:nChunks
    m = min(CHUNK_TRAJ, left);
    left = left - m;
    if m<=0, break; end

    Xseq_c = [];
    y_c    = [];

    for i = 1:m
        % 起终位形（deg → rad）
        q0d   = sample_box(PB.alpha_budget.q0_range_deg);
        qrefd = sample_box(PB.alpha_budget.qref_range_deg);
        q0 = deg2rad_local(q0d);
        qf = deg2rad_local(qrefd);

        % Bézier + DLS-IK（几何域）
        cfgGeom = struct('DT',DT, ...
                         'DLS_LAMBDA', double(PB.alpha_budget.dls_lambda), ...
                         'DLS_W_ORI',  double(PB.alpha_budget.dls_w_ori), ...
                         'BEZ_LIFT_Z', double(PB.alpha_budget.bez_lift_z), ...
                         'BEZ_GAMMA',  double(PB.alpha_budget.bez_gamma), ...
                         'NS_MIN',     double(PB.alpha_budget.ns_min));
        TRAJ = make_traj_cart_bezier_dls_geom(robot, q0, qf, cfgGeom);

        % 逐轨迹采样 Pmax 和 payload
        if strcmpi(PB.alpha_budget.pmax_mode,'fixed')
            pmax = PB.power.P_TOTAL_MAX;
        else
            pmax = PB.alpha_budget.pmax_range(1) + ...
                   rand()*(PB.alpha_budget.pmax_range(2)-PB.alpha_budget.pmax_range(1));
        end
        caps_use = caps0;
        caps_use.P_total_max = pmax;

        payload_struct = sample_payload_struct(PB.payload.range);

        % —— 生成 governor-aware look-ahead DS ——
        opts = struct('dt',DT, 'Ts',TS, 'alpha_min',ALPHA_MIN, 'itmax',ITMAX, ...
                      'fric', struct('B',double((PB.fric.B(:)).'), ...
                                     'Fc',double((PB.fric.Fc(:)).'), ...
                                     'vel_eps',double(PB.fric.vel_eps)), ...
                      'payload', payload_struct, ...
                      'BETA',double(areg.beta), ...
                      'A_DOT_UP',double(areg.a_dot_up), ...
                      'A_DOT_DN',double(areg.a_dot_dn), ...
                      'K', double(PB.train_trf.K), ...
                      'H', double(PB.train_trf.H));
        DSi = spinn3d_make_dataset_alpha_gov_geom_seq(robot, TRAJ, caps_use, opts);

        Xseq_c = cat(1, Xseq_c, double(DSi.Xseq)); %#ok<AGROW>
        y_c    = [y_c; double(DSi.y)]; %#ok<AGROW>
    end

    chunk_id = next_id + (c-1);
    fn_chunk = fullfile(SAVE_DIR, sprintf('alpha_gov_seq_ds_chunk_%d.mat', chunk_id));
    save(fn_chunk, 'Xseq_c','y_c','caps0','PB','-v7.3');
    fprintf('[alpha-gov-trf] wrote chunk #%d: %s (samples=%d, K=%d)\n', ...
            chunk_id, fn_chunk, size(Xseq_c,1), size(Xseq_c,2));
end

% ===== 聚合主表 =====
files = dir(chunk_pat);
assert(~isempty(files), '未生成任何分片；检查 PB.alpha_budget.* 是否齐全。');
Xseq=[]; y=[];
for i=1:numel(files)
    S = load(fullfile(files(i).folder, files(i).name), 'Xseq_c','y_c');
    Xseq = cat(1, Xseq, S.Xseq_c); %#ok<AGROW>
    y    = [y; S.y_c]; %#ok<AGROW>
end
Xseq(~isfinite(Xseq)) = 0;
y(~isfinite(y)) = ALPHA_MIN; y = min(1.0, max(ALPHA_MIN, y));

master_mat = fullfile(SAVE_DIR, 'spinn3d_alpha_gov_seq_dataset_master.mat');
save(master_mat, 'Xseq','y','caps0','PB','-v7.3');
fprintf('[alpha-gov-trf] 样本: %d | 主表: %s\n', size(Xseq,1), master_mat);

% ===== 训练 Transformer =====
model_path = char(PB.paths.model_alpha);
if PB.train_trf.enable
    Klook = double(PB.train_trf.K);
    if Klook <= 1
        ds = 0;
    else
        ds = double(PB.train_trf.H) / (Klook - 1);
    end
    DS = struct('Xseq',Xseq,'y',y,'alpha_min',ALPHA_MIN,'K',Klook,'ds',ds); %#ok<NASGU>

    topts = struct();
    topts.MaxEpochs        = PB.train_trf.epochs;
    topts.MiniBatchSize    = PB.train_trf.batch;
    topts.InitialLearnRate = PB.train_trf.lr;

    % soft constraint weights
    if isfield(PB.train_trf,'lambda_over'),  topts.LambdaOver  = PB.train_trf.lambda_over;  end
    if isfield(PB.train_trf,'lambda_under'), topts.LambdaUnder = PB.train_trf.lambda_under; end

    % horizon-PINN / speed / end
    if isfield(PB.train_trf,'lambda_pinn'),  topts.LambdaPINN  = PB.train_trf.lambda_pinn;  end
    if isfield(PB.train_trf,'lambda_speed'), topts.LambdaSpeed = PB.train_trf.lambda_speed; end
    if isfield(PB.train_trf,'lambda_end'),   topts.LambdaEnd   = PB.train_trf.lambda_end;   end
    if isfield(PB.train_trf,'end_phase'),    topts.EndPhase    = PB.train_trf.end_phase;    end
    if isfield(PB.train_trf,'end_width'),    topts.EndWidth    = PB.train_trf.end_width;    end
    if isfield(PB.train_trf,'pinn_token_decay'), topts.PinnTokenDecay = PB.train_trf.pinn_token_decay; end

    % arch
    if isfield(PB.train_trf,'d_model'),   topts.DModel   = PB.train_trf.d_model; end
    if isfield(PB.train_trf,'n_heads'),   topts.NumHeads = PB.train_trf.n_heads; end
    if isfield(PB.train_trf,'ffn_dim'),   topts.FFNDim   = PB.train_trf.ffn_dim; end
    if isfield(PB.train_trf,'n_layers'),  topts.NumLayers= PB.train_trf.n_layers; end
    if isfield(PB.train_trf,'dropout'),   topts.Dropout  = PB.train_trf.dropout; end

    % physics constants
    topts.CAPS = caps0;
    topts.FRIC = PB.fric;
    fprintf('[alpha-gov-trf] Training TRF: epochs=%d, batch=%d, lr=%.3g | model=%s\n', ...
        topts.MaxEpochs, topts.MiniBatchSize, topts.InitialLearnRate, model_path);

    spinn3d_train_alpha_gov_trf(DS, model_path, topts);
end

out = struct('dataset_master', master_mat, 'model', model_path);
end

%% ============= helpers =============
function req(S, keys), if ischar(keys), keys={keys}; end
for i=1:numel(keys), assert(isfield(S,keys{i}) && ~isempty(S.(keys{i})), 'PB 缺少字段：%s', keys{i}); end
end
function q = sample_box(Qdeg), lo=Qdeg(:,1).'; hi=Qdeg(:,2).'; q=lo+(hi-lo).*rand(size(lo)); end
function ids = parse_ids(names, pat)
    ids = [];
    for i = 1:numel(names)
        m = regexp(names{i}, pat, 'tokens', 'once');
        if ~isempty(m), ids(end+1) = str2double(m{1}); end %#ok<AGROW>
    end
end
function robot = resolve_robot_from_PB(PB)
    robot = [];
    if isfield(PB,'robot') && isfield(PB.robot,'object') && ~isempty(PB.robot.object), robot = PB.robot.object; return; end
    if isfield(PB,'robot') && isfield(PB.robot,'builder') && ~isempty(PB.robot.builder), robot = feval(PB.robot.builder); return; end
    error('PB.robot 需要提供 object 或 builder。');
end
function y = deg2rad_local(x), y=(pi/180).*x; end
function pay = sample_payload_struct(R)
% R: PB.payload.range
% 输出 pay.inertia 采用 Robotics System Toolbox 口径：
%   [Ixx Iyy Izz Iyz Ixz Ixy]，并且默认表示“关于 COM”的惯量

    pay = struct('mass',0,'com',[0 0 0],'inertia',zeros(1,6), ...
                 'about','com','mode','replace');

    if ~isstruct(R), return; end

    % --- mass ---
    if isfield(R,'mass') && numel(R.mass)>=2
        pay.mass = R.mass(1) + rand()*(R.mass(2)-R.mass(1));
    end

    % --- com ---
    if isfield(R,'com') && all(size(R.com)==[3 2])
        lo = R.com(:,1).'; hi = R.com(:,2).';
        pay.com = lo + (hi-lo).*rand(1,3);
    end

    % --- inertia ---
    % 1) 若显式给了 inertia 采样范围：直接采（注意应为 RST 顺序）
    if isfield(R,'inertia') && all(size(R.inertia)==[6 2])
        lo = R.inertia(:,1).'; hi = R.inertia(:,2).';
        pay.inertia = lo + (hi-lo).*rand(1,6);
        return;
    end

    % 2) 否则：用 box_dims 生成一个物理可行惯量（盒子，轴对齐，off-diagonal=0）
    if isfield(R,'box_dims') && all(size(R.box_dims)==[3 2]) && pay.mass > 0
        lo = R.box_dims(:,1).'; hi = R.box_dims(:,2).';
        dims = lo + (hi-lo).*rand(1,3);  % [a b c]
        a = dims(1); b = dims(2); c = dims(3);

        Ixx = (1/12)*pay.mass*(b^2 + c^2);
        Iyy = (1/12)*pay.mass*(a^2 + c^2);
        Izz = (1/12)*pay.mass*(a^2 + b^2);

        pay.inertia = [Ixx Iyy Izz 0 0 0]; % [Ixx Iyy Izz Iyz Ixz Ixy]
    end
end
