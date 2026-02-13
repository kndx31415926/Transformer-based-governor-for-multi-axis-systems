function [out, fn_chunk] = spinn3d_add_nn_rollout_as_chunk(cfg)
%SPINN3D_ADD_NN_ROLLOUT_AS_CHUNK
% 一键：运行 spinn3d_demo_ctcgov_online_nn() 出图
%      + 将该 NN rollout 重新用 teacher 标注成一个新的训练 chunk 写入 PB.paths.root
%
% 关键保证：
%   (1) 标签 y 不是 NN 输出，而是 teacher：
%       a_star = alpha_feasible(...)
%       a_des  = alpha_des_teacher(a_star, a_prev, ...)
%   (2) robot 动力学用于 teacher 与特征时，会按 demo 的 payload 口径注入（并在末尾 restore）
%   (3) s 对齐：demo 的 log.s(k) 是推进后的 s_{k+1}，本脚本用 s_used(k) = (k==1?0:log.s(k-1))

    if nargin < 1
        cfg = [];
    end
    out = spinn3d_demo_ctcgov_online_nn(cfg);

    assert(isfield(out,'log') && isstruct(out.log), 'out.log 缺失');
    req(out.log, {'s','alpha'});
    PB = spinn3d_params_block();
    req(PB, {'paths','sample','train_trf','alpha_reg'});
    req(PB.paths, {'root'});
    req(PB.sample, {'dt'});
    req(PB.train_trf, {'K','H'});

    SAVE_DIR = char(PB.paths.root);
    if ~exist(SAVE_DIR,'dir'), mkdir(SAVE_DIR); end

    % 下一个 chunk id：扫描已有 alpha_gov_seq_ds_chunk_*.mat → max+1
    chunk_pat = fullfile(SAVE_DIR, 'alpha_gov_seq_ds_chunk_*.mat');
    existing  = dir(chunk_pat);
    used_ids  = parse_ids_local({existing.name}, 'alpha_gov_seq_ds_chunk_(\d+)\.mat');
    if isempty(used_ids), next_id = 1; else, next_id = max(used_ids) + 1; end

    % ============================================================
    % 3) 训练口径参数：K/H/ds、dt
    % ============================================================
    K  = double(PB.train_trf.K);
    H  = double(PB.train_trf.H);
    dt = double(PB.sample.dt);
    if K <= 1, ds = 0; else, ds = H / (K-1); end

    % governor 参数：默认用 PB.alpha_reg（训练口径）
    beta = double(PB.alpha_reg.beta);
    up   = double(PB.alpha_reg.a_dot_up);
    dn   = double(PB.alpha_reg.a_dot_dn);

    % 若 out.cfg 里显式带了 ALPHA_REG（用户自定义），则采用它，并提示可能与 PB 不一致
    cfg_run = out.cfg;
    if isfield(cfg_run,'ALPHA_REG') && isstruct(cfg_run.ALPHA_REG) && ...
       all(isfield(cfg_run.ALPHA_REG,{'BETA','A_DOT_UP','A_DOT_DN'}))
        beta2 = double(cfg_run.ALPHA_REG.BETA);
        up2   = double(cfg_run.ALPHA_REG.A_DOT_UP);
        dn2   = double(cfg_run.ALPHA_REG.A_DOT_DN);
        if any(abs([beta2-beta, up2-up, dn2-dn]) > 1e-12)
            warning('ALPHA_REG in cfg != PB.alpha_reg: 用 cfg.ALPHA_REG 标注；确保后续训练也用同一组参数。');
        end
        beta = beta2; up = up2; dn = dn2;
    end

    % ============================================================
    % 4) 从 demo 输出拿 robot/caps/fric/payload，并对齐 BUS 字段
    % ============================================================
    req(cfg_run, {'ROBOT','FRIC','TS','ALPHA_MIN','ALPHA_ITMAX','Q0_DEG','QF_DEG'});
    Ts        = double(cfg_run.TS);
    alpha_min = double(cfg_run.ALPHA_MIN);
    itmax     = double(cfg_run.ALPHA_ITMAX);
    fric      = cfg_run.FRIC;

    limits = out.limits;  % demo 已经保证含 tau_max/qd_max/qdd_max/P_axis_max/P_total_max
    req(limits, {'tau_max','qd_max','qdd_max','P_axis_max','P_total_max'});
    Pmax = double(limits.P_total_max);

    % BUS 口径字段（alpha_feasible 里缺省也能跑，但这里补齐以对齐 demo/控制器）
    bus = get_bus_from_cfg(cfg_run, Pmax);
    bus = spinn3d_bus_defaults(bus);
    caps_use = limits;
    if isfield(bus,'eta_share') && ~isempty(bus.eta_share),   caps_use.eta_share  = double(bus.eta_share); end
    if isfield(bus,'P_brk_peak') && ~isempty(bus.P_brk_peak), caps_use.P_brk_peak = double(bus.P_brk_peak); end

    % payload：demo 返回 out.payload（mass/com/inertia），但不一定带 about/mode
    payload = out.payload;
    if isfield(cfg_run,'PAYLOAD') && isstruct(cfg_run.PAYLOAD) && isfield(cfg_run.PAYLOAD,'DEMO') && isstruct(cfg_run.PAYLOAD.DEMO)
        pc = cfg_run.PAYLOAD.DEMO;
        if isfield(pc,'about') && ~isempty(pc.about), payload.about = pc.about; end
        if isfield(pc,'mode')  && ~isempty(pc.mode),  payload.mode  = pc.mode;  end
    end

    % ============================================================
    % 5) 构造 time-domain 参考轨迹（与 demo 同一生成器）
    % ============================================================
    robot = cfg_run.ROBOT;  try, robot.DataFormat='row'; end
    q0 = unwrap_to_near_local(0, deg2rad_local(row(cfg_run.Q0_DEG)));
    qf = unwrap_to_near_local(q0, deg2rad_local(row(cfg_run.QF_DEG)));

    traj = make_traj_cart_bezier_dls(robot, q0, qf, limits, cfg_run);
    Tf   = double(traj.Tf);

    % ============================================================
    % 6) ★物理一致性关键：把 payload 注入到 robot（动力学一致），并在结束时 restore
    % ============================================================
    [robot_use, restore] = inject_payload_last_body_rst(robot, payload);
    cleaner = onCleanup(@() restore_robot_last_body(robot, restore)); %#ok<NASGU>

    % ============================================================
    % 7) ★逻辑一致性关键：重建“本步控制器使用的 s”
    %    demo 记录的是推进后的 s_{k+1}，所以：
    %       s_used(1)=0; s_used(k)=log.s(k-1)
    % ============================================================
    s_log = double(out.log.s(:));
    a_log = double(out.log.alpha(:));
    assert(numel(s_log)==numel(a_log), 'log.s 与 log.alpha 长度不一致');

    Nstep = numel(a_log);
    s_used = zeros(Nstep,1);
    if Nstep >= 2
        s_used(2:end) = s_log(1:end-1);
    end
    a_prev_vec = zeros(Nstep,1);
    a_prev_vec(1) = alpha_min;
    if Nstep >= 2
        a_prev_vec(2:end) = a_log(1:end-1);
    end

    % ============================================================
    % 8) 预分配：Nmax ≈ Tf/dt
    % ============================================================
    Nmax = max(2, ceil(max(Tf,0)/max(dt,1e-12)) + 5);

    % 先试算 Dtot
    [q_tmp, dq_tmp, dd_tmp] = traj.eval(0);
    xb0 = spinn3d_features_alpha_plus_payload(row(q_tmp), row(dq_tmp), row(dd_tmp), ...
                                              robot_use, payload, Pmax, caps_use);
    x0  = double([row(xb0), a_prev_vec(1), beta, up, dn, Ts, 0]);
    Dtot = numel(x0);

    % 如果已有 chunk，做一个维度兼容检查（避免把不兼容数据写进去）
    if ~isempty(existing)
        mf = matfile(fullfile(SAVE_DIR, existing(1).name));
        sz = size(mf, 'Xseq_c');  % [N,K,D]
        if numel(sz) ~= 3 || sz(2) ~= K
            error('已有 chunk 的 K=%d，与当前 PB.train_trf.K=%d 不一致，先统一 K 再追加。', sz(2), K);
        end
        if sz(3) ~= Dtot
            error('已有 chunk 的 Dtot=%d，与本次生成 Dtot=%d 不一致，特征维度/顺序不一致，禁止追加。', sz(3), Dtot);
        end
    end

    Xseq_c = zeros(Nmax, K, Dtot, 'double');
    y_c    = zeros(Nmax, 1, 'double');

    % ============================================================
    % 9) 生成样本：按 dt 沿 s_used 下采样；标签用 teacher 重新标注
    % ============================================================
    idx = 0;
    s_next = 0;

    for k = 1:Nstep
        s0 = s_used(k);
        if s0 < s_next - 1e-12
            continue;
        end

        s0 = min(Tf, max(0, s0));
        a_prev = clamp_local(double(a_prev_vec(k)), 0.0, 1.0);  % a_prev 是 rollout 的真实状态，别强行抬到 alpha_min

        % teacher label
        [q_r, dq_r, dd_r] = traj.eval(s0);
        a_star = alpha_feasible(row(q_r), row(dq_r), row(dd_r), 1.0, robot_use, caps_use, ...
                                Pmax, alpha_min, itmax, fric);
        a_star = clamp_local(double(a_star), alpha_min, 1.0);

        a_des = alpha_des_teacher(a_star, a_prev, beta, up, dn, Ts);
        a_des = clamp_local(double(a_des), alpha_min, 1.0);

        % tokens
        idx = idx + 1;
        for ti = 1:K
            s_i = min(Tf, s0 + (ti-1)*ds);
            [qi, dqi, ddi] = traj.eval(s_i);

            xb = spinn3d_features_alpha_plus_payload(row(qi), row(dqi), row(ddi), ...
                                                     robot_use, payload, Pmax, caps_use);
            s_norm = min(1.0, s_i / max(Tf, 1e-9));
            Xseq_c(idx,ti,:) = double([row(xb), a_prev, beta, up, dn, Ts, s_norm]);
        end
        y_c(idx,1) = a_des;

        s_next = s_next + dt;
        if s_next > Tf + 1e-9
            break;
        end
        if idx >= Nmax
            break;
        end
    end

    if idx < 1
        error('没有生成任何样本：检查 log.s/log.alpha 或 dt 设置是否异常。');
    end

    Xseq_c = Xseq_c(1:idx,:,:);
    y_c    = y_c(1:idx,:);

    % ============================================================
    % 10) 保存 chunk：命名与 run 脚本一致
    % ============================================================
    fn_chunk = fullfile(SAVE_DIR, sprintf('alpha_gov_seq_ds_chunk_%d.mat', next_id));
    caps0 = caps_use; %#ok<NASGU>
    save(fn_chunk, 'Xseq_c','y_c','caps0','PB','-v7.3');

    fprintf('[add-nn-rollout-chunk] wrote: %s | samples=%d | K=%d | Dtot=%d\n', ...
            fn_chunk, size(Xseq_c,1), size(Xseq_c,2), size(Xseq_c,3));

end

% ===================== local helpers =====================
function ids = parse_ids_local(names, pattern)
    ids = [];
    if isempty(names), return; end
    for i = 1:numel(names)
        tok = regexp(names{i}, pattern, 'tokens', 'once');
        if ~isempty(tok)
            v = str2double(tok{1});
            if isfinite(v), ids(end+1) = v; end %#ok<AGROW>
        end
    end
end

function v = row(v), v = v(:).'; end
function r = deg2rad_local(d), r = d * pi / 180; end

function qf = unwrap_to_near_local(q0, qf)
    dq = qf - q0;
    qf = q0 + mod(dq + pi, 2*pi) - pi;
end

function x = clamp_local(x, lo, hi)
    x = min(hi, max(lo, x));
end

% ---- payload injection (RST order) ----
function [robot_out, restore] = inject_payload_last_body_rst(robot_in, payload)
    robot_out = robot_in;
    restore = struct('has',false);

    if isempty(payload), return; end

    % parse payload
    if isnumeric(payload)
        pv = double(payload(:)).';
        if numel(pv) < 10, pv = [pv, zeros(1,10-numel(pv))]; end
        m = pv(1);  com = pv(2:4);  I6 = pv(5:10);
        about = 'com';  mode = 'replace';
    elseif isstruct(payload)
        m = getfield_def(payload,'mass',0);
        com = getfield_def(payload,'com',[0 0 0]);
        I6 = getfield_def(payload,'inertia',[0 0 0 0 0 0]);
        about = lower(getfield_def(payload,'about','com'));
        mode  = lower(getfield_def(payload,'mode','replace'));
    else
        return;
    end

    m = double(m);
    com = double(com(:)).';
    I6 = double(I6(:)).';
    if numel(I6) < 6, I6 = [I6, zeros(1,6-numel(I6))]; end
    if ~isfinite(m) || m <= 0, return; end
    if numel(com)~=3 || any(~isfinite(com)), com = [0 0 0]; end

    try
        b = robot_out.Bodies{end};

        % cache for restore
        restore.has = true;
        restore.Mass = b.Mass;
        restore.CenterOfMass = b.CenterOfMass;
        restore.Inertia = b.Inertia;

        % current inertia about origin (RST)
        I0 = I6ToMat_rst(double(b.Inertia(:)).');
        m0 = double(b.Mass);
        c0 = double(b.CenterOfMass(:)).';

        % payload inertia matrix
        Ic = I6ToMat_rst(I6);

        if strcmpi(about,'com')
            r = double(com(:)); r = r(:);
            I1 = Ic + m * ((r.'*r)*eye(3) - (r*r.'));
        else
            I1 = Ic;
        end

        switch mode
            case 'add'
                m_new = m0 + m;
                if m_new <= 0
                    c_new = [0 0 0];
                else
                    c_new = (m0*c0 + m*com) / m_new;
                end
                I_new = I0 + I1;
            otherwise % replace
                m_new = m;
                c_new = com;
                I_new = I1;
        end

        b.Mass = m_new;
        b.CenterOfMass = c_new(:).';
        b.Inertia = ImatTo6_rst(I_new);

    catch
        % if something fails, mark restore invalid
        restore.has = false;
    end
end

function restore_robot_last_body(robot, restore)
    if ~isstruct(restore) || ~isfield(restore,'has') || ~restore.has
        return;
    end
    try
        b = robot.Bodies{end};
        b.Mass = restore.Mass;
        b.CenterOfMass = restore.CenterOfMass;
        b.Inertia = restore.Inertia;
    catch
    end
end

function I = I6ToMat_rst(I6)
    % RST inertia vector: [Ixx Iyy Izz Iyz Ixz Ixy]
    Ixx = I6(1); Iyy = I6(2); Izz = I6(3);
    Iyz = I6(4); Ixz = I6(5); Ixy = I6(6);
    I = [ Ixx, Ixy, Ixz;
          Ixy, Iyy, Iyz;
          Ixz, Iyz, Izz ];
end

function I6 = ImatTo6_rst(I)
    I6 = [I(1,1), I(2,2), I(3,3), I(2,3), I(1,3), I(1,2)];
end

function v = getfield_def(S,fn,def)
    if isstruct(S) && isfield(S,fn) && ~isempty(S.(fn))
        v = S.(fn);
    else
        v = def;
    end
end
