function DS = spinn3d_make_dataset_alpha_gov_geom_seq(robot, TRAJ, caps, opts)
%SPINN3D_MAKE_DATASET_ALPHA_GOV_GEOM_SEQ
% 几何轨迹 → governor-aware look-ahead 序列数据集（用于 Transformer 训练）
%
% 该版本修正了两个关键口径问题（P0/P1）：
%   P0: 训练数据的序列演化与在线推理一致
%       - a_prev 按控制周期 Ts 滚动更新（不是按 dt）
%       - 几何进度 s 按 s <- s + Ts * a_post 推进（与在线一致）
%       - 数据量仍按 opts.dt（几何域步长）抽样，避免样本爆炸
%
%   P1: payload 必须真正进入动力学（teacher/特征）
%       - 在本函数内部把 payload 注入到末端 body（质量/COM/惯量）
%       - teacher(alpha_feasible) 与特征(spinn3d_features_alpha_plus_payload)均使用注入后的 robot
%
% 输入:
%   robot : rigidBodyTree
%   TRAJ  : struct, 需要字段:
%           - Tf
%           - eval(s) -> [q, dqg, ddqg]  (dqg,ddqg 为对几何参数 s 的导数)
%   caps  : struct, 需要字段:
%           - P_total_max
%   opts  : struct, 必需字段:
%           - dt        : 几何域采样步长（只用于抽样/记录样本数）
%           - Ts        : 控制周期（用于 rollout / governor 离散）
%           - alpha_min : 最小 alpha
%           - itmax     : alpha_feasible 迭代次数
%           - fric      : 摩擦参数结构体
%         可选字段:
%           - K, H, ds  : look-ahead 序列长度与间隔（ds=H/(K-1)）
%           - BETA, A_DOT_UP, A_DOT_DN : governor 参数（会作为 token 尾巴特征）
%           - payload   : struct 或 1x10 数值向量
%                        payload.inertia 采用 RST 口径 [Ixx Iyy Izz Iyz Ixz Ixy]
%                        payload.about='com' 表示惯量关于 COM（默认，会平行轴换算到 body 原点）
%                        payload.mode ='replace' 或 'add'（默认 replace）
%
% 输出:
%   DS.Xseq : N×K×D (double)
%   DS.y    : N×1 (a_des)
%   DS.alpha_min, DS.K, DS.ds, DS.dt, DS.Ts
%
% 重要惯量口径:
%   Robotics System Toolbox: rigidBody.Inertia 向量顺序为 [Ixx Iyy Izz Iyz Ixz Ixy]
%   且其值是“相对 body frame 原点”的惯量；URDF 常见的是关于 COM，需要做平行轴换算。
%
% 依赖函数（工程内已有）:
%   - alpha_feasible
%   - alpha_des_teacher
%   - governor_step
%   - spinn3d_features_alpha_plus_payload
%
% -------------------------------------------------------------------------

    assert(isfield(opts,'dt') && opts.dt>0, 'opts.dt 必须提供且 > 0');
    assert(isfield(opts,'alpha_min'), 'opts.alpha_min 必须提供');
    assert(isfield(opts,'itmax'), 'opts.itmax 必须提供');
    assert(isfield(opts,'fric'), 'opts.fric 必须提供');

    dt   = double(opts.dt);
    Ts   = double(getfield_def(opts,'Ts', dt));
    assert(isfinite(Ts) && Ts>0, 'opts.Ts 必须 > 0');

    beta = double(getfield_def(opts,'BETA',0.85));
    up   = double(getfield_def(opts,'A_DOT_UP',2.5));
    dn   = double(getfield_def(opts,'A_DOT_DN',5.0));
    fric = opts.fric;

    % look-ahead
    K  = double(getfield_def(opts,'K', 9));
    assert(isfinite(K) && K>=1 && round(K)==K, 'opts.K 必须为正整数');
    if K==1
        ds = 0;
    else
        H  = double(getfield_def(opts,'H', 0.8));
        ds = double(getfield_def(opts,'ds', H/(K-1)));
        assert(isfinite(ds) && ds>0, 'K>1 时 ds 必须 >0');
    end

    assert(isfield(caps,'P_total_max'), 'caps.P_total_max 缺失');
    Pmax = double(caps.P_total_max);

    % payload：兼容 struct 或 1x10
    if ~isfield(opts,'payload') || isempty(opts.payload)
        payload = struct('mass',0,'com',[0 0 0],'inertia',zeros(1,6), ...
                         'about','com','mode','replace');
    else
        if isstruct(opts.payload)
            payload = opts.payload;
            if ~isfield(payload,'mass'),    payload.mass = 0; end
            if ~isfield(payload,'com'),     payload.com = [0 0 0]; end
            if ~isfield(payload,'inertia'), payload.inertia = zeros(1,6); end
            if ~isfield(payload,'about') || isempty(payload.about), payload.about = 'com'; end
            if ~isfield(payload,'mode')  || isempty(payload.mode),  payload.mode  = 'replace'; end
        else
            v = double(opts.payload(:)).';
            if numel(v) < 10, v = [v, zeros(1,10-numel(v))]; end
            payload = struct('mass',v(1),'com',v(2:4),'inertia',v(5:10), ...
                             'about','com','mode','replace');
        end
    end

    % ===== 注入 payload 到 robot（teacher/特征同口径）=====
    [robot_use, restore] = inject_payload_last_body_rst(robot, payload);

    Tf = double(TRAJ.Tf);
    if ~isfinite(Tf) || Tf <= 0
        Tf = 1.0;
    end

    % 预分配：按几何域 dt 估计最大样本数（与旧版本数量级一致）
    Nmax = max(2, ceil(Tf/dt) + 2);

    % 先试算一次得到 Dtot
    [q0, dqg0, ddqg0] = TRAJ.eval(0);
    s_norm0 = 0;
    xb0 = spinn3d_features_alpha_plus_payload(row(q0), row(dqg0), row(ddqg0), robot_use, payload, Pmax, caps);
    x0  = double([xb0, double(opts.alpha_min), beta, up, dn, Ts, s_norm0]);
    Dtot = numel(x0);

    Xseq = zeros(Nmax, K, Dtot);
    Y    = zeros(Nmax, 1);

    % ===== Ts rollout：s 按 Ts*alpha 推进；每 dt 记录一次 =====
    s       = 0.0;
    s_next  = 0.0;                       % 下一个记录点（几何进度）
    a_prev  = double(opts.alpha_min);    % 上一步 a_post
    idx     = 0;

    % 防止极端参数导致死循环
    maxSteps = 5000000;
    stepCnt  = 0;

    while (s < Tf - 1e-12) && (idx < Nmax) && (stepCnt < maxSteps)
        stepCnt = stepCnt + 1;

        % 当前点 teacher
        [q, dqg, ddqg] = TRAJ.eval(s);
        q=row(q); dqg=row(dqg); ddqg=row(ddqg);

        a_star = alpha_feasible(q, dqg, ddqg, 1.0, robot_use, caps, Pmax, opts.alpha_min, opts.itmax, fric);
        a_des  = alpha_des_teacher(a_star, a_prev, beta, up, dn, Ts);

        % 记录样本（以几何域 dt 为主，保持数据量）
        if s >= s_next - 1e-12
            idx = idx + 1;

            for ti = 1:K
                s_i = min(Tf, s + (ti-1)*ds);
                [qi, dqgi, ddqgi] = TRAJ.eval(s_i);
                qi=row(qi); dqgi=row(dqgi); ddqgi=row(ddqgi);

                s_norm_i = min(1.0, s_i/max(Tf,1e-9));
                xb = spinn3d_features_alpha_plus_payload(qi, dqgi, ddqgi, robot_use, payload, Pmax, caps);

                Xseq(idx,ti,:) = double([xb, a_prev, beta, up, dn, Ts, s_norm_i]);
            end

            Y(idx,1) = a_des;

            s_next = s_next + dt;
        end

        % governor 更新 + 推进 s
        a_post = governor_step(a_des, a_prev, beta, up, dn, Ts);
        if ~isfinite(a_post)
            a_post = a_prev;
        end
        % 确保能推进
        if a_post <= 0
            a_post = max(a_post, double(opts.alpha_min));
        end

        a_prev = a_post;
        s      = min(Tf, s + Ts*a_post);

        % 如果 alpha_min=0 且 a_post 仍为 0，避免无限循环
        if Ts*a_post <= 0
            break;
        end
    end

    % 截断到实际样本数
    if idx < 1
        Xseq = zeros(1, K, Dtot);
        Y    = double(opts.alpha_min);
    else
        Xseq = Xseq(1:idx,:,:);
        Y    = Y(1:idx,:);
    end

    % 清洗
    Xseq(~isfinite(Xseq)) = 0;
    Y(~isfinite(Y)) = double(opts.alpha_min);
    Y = min(1.0, max(double(opts.alpha_min), Y));

    DS = struct();
    DS.Xseq = Xseq;
    DS.y    = Y;
    DS.alpha_min = double(opts.alpha_min);
    DS.K    = K;
    DS.ds   = ds;
    DS.dt   = dt;
    DS.Ts   = Ts;

    % 恢复 robot（避免对外部 caller 产生副作用）
    restore_robot_last_body(robot_use, restore);
end

% =====================================================================
% helper functions
% =====================================================================

function v = row(v)
    v = v(:).';
end

function x = getfield_def(S, fn, def)
    if isstruct(S) && isfield(S,fn) && ~isempty(S.(fn))
        x = S.(fn);
    else
        x = def;
    end
end

function [robot_out, restore] = inject_payload_last_body_rst(robot_in, payload)
% 将 payload 注入 robot 的末端 body。
% 口径：
%   payload.inertia 采用 RST 顺序 [Ixx Iyy Izz Iyz Ixz Ixy]
%   payload.about='com' 表示惯量关于 COM，会用平行轴定理换算到 body 原点
%   payload.mode='replace' 或 'add'
%
% 注意：rigidBodyTree/rigidBody 是 handle；此函数会就地修改 robot，但会返回 restore 供恢复。

    robot_out = robot_in;
    restore = struct('has',false,'Mass',[],'CenterOfMass',[],'Inertia',[]);

    if isempty(robot_in) || isempty(payload)
        return;
    end

    % 解析 payload
    if isnumeric(payload)
        v = double(payload(:)).';
        if numel(v)==10
            m = v(1); com = v(2:4); I6 = v(5:10);
        else
            return;
        end
        about = 'com';
        mode  = 'replace';
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
    if ~isfinite(m) || m <= 0
        return;
    end
    if numel(com)~=3 || any(~isfinite(com))
        com = [0 0 0];
    end

    try
        b = robot_out.Bodies{end};

        % cache for restore
        restore.has = true;
        restore.Mass = b.Mass;
        restore.CenterOfMass = b.CenterOfMass;
        restore.Inertia = b.Inertia;

        % 当前末端的原惯量（RST: about frame origin）
        I0 = I6ToMat_rst(double(b.Inertia(:)).');
        m0 = double(b.Mass);
        c0 = double(b.CenterOfMass(:)).';

        % payload inertia matrix（输入为 about=COM 或 about=origin）
        Ic = I6ToMat_rst(I6);

        if strcmpi(about,'com')
            r = double(com(:)); r = r(:);
            I1 = Ic + m * ((r.'*r)*eye(3) - (r*r.'));
        else
            I1 = Ic;
        end

        switch mode
            case 'add'
                % 两个分布都以“同一 body 原点”为参考，惯量关于原点可直接相加
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
        b.CenterOfMass = c_new;
        b.Inertia = ImatTo6_rst(I_new).';
    catch
        % 若注入失败，则不影响 dataset 的其它部分
    end
end

function restore_robot_last_body(robot_in, restore)
    if ~isstruct(restore) || ~isfield(restore,'has') || ~restore.has
        return;
    end
    try
        b = robot_in.Bodies{end};
        b.Mass = restore.Mass;
        b.CenterOfMass = restore.CenterOfMass;
        b.Inertia = restore.Inertia;
    catch
    end
end

function I = I6ToMat_rst(v6)
% Robotics System Toolbox: v6 = [Ixx Iyy Izz Iyz Ixz Ixy]
    v6 = double(v6(:)).';
    Ixx=v6(1); Iyy=v6(2); Izz=v6(3);
    Iyz=v6(4); Ixz=v6(5); Ixy=v6(6);
    I = [ Ixx Ixy Ixz;
          Ixy Iyy Iyz;
          Ixz Iyz Izz ];
end

function v6 = ImatTo6_rst(I)
% Robotics System Toolbox: v6 = [Ixx Iyy Izz Iyz Ixz Ixy]
    v6 = [ I(1,1), I(2,2), I(3,3), I(2,3), I(1,3), I(1,2) ];
end
