function TRAJ = make_traj_cart_bezier_dls_geom(robot, q0, qf, cfg)
% make_traj_cart_bezier_dls_geom
% 几何域 Bézier + DLS-IK（只控位置，带姿态保守项），用于训练数据生成
% 输出:
%   TRAJ.s        : 等间隔几何参数 s ∈ [0,1]
%   TRAJ.Q        : 关节路径（按 s）
%   TRAJ.QDg      : 几何一阶导 d q / d s
%   TRAJ.QDDg     : 几何二阶导 d^2 q / d s^2
%   TRAJ.eval(s)  : 返回 [q, qg_dot, qg_ddot]（线性插值）
%   TRAJ.Tf       : 1.0（几何域总长归一）
% 依赖 cfg 字段（均有默认）:
%   DT            : 几何采样间距（默认 0.01）
%   NS_MIN        : s 采样最少点数（默认 200）
%   DLS_LAMBDA    : DLS 阻尼（默认 5e-3）
%   DLS_W_ORI     : 姿态保守项权重 w_ori（默认 0.02）
%   BEZ_LIFT_Z    : 抬升高度（默认 0.15 m）
%   BEZ_GAMMA     : 控制点内推系数（默认 1/3）
%   EE_NAME       : 末端名（默认 robot.BodyNames{end}）
%   BASE_NAME     : 基座名（可空；兼容老接口）

    try, robot.DataFormat = 'row'; end
    nJ = numel(q0);

    % ---- 读取配置（带默认） ----
    DT       = getdef(cfg,'DT',0.01);
    NS_MIN   = getdef(cfg,'NS_MIN',200);
    lam      = getdef(cfg,'DLS_LAMBDA',5e-3);
    w_ori    = getdef(cfg,'DLS_W_ORI',0.02);
    lift_z   = getdef(cfg,'BEZ_LIFT_Z',0.15);
    gamma    = getdef(cfg,'BEZ_GAMMA',1/3);
    eeName   = getdef(cfg,'EE_NAME', robot.BodyNames{end});
    baseName = getdef(cfg,'BASE_NAME','');

    % ---- Bézier 控制点（抬升弓形）----
    P = make_bezier_ctrl_points(robot, q0, qf, eeName, baseName, lift_z, gamma);

    % 等间隔几何参数 s
    Ns  = max(round(1/DT)+1, NS_MIN);
    s   = linspace(0,1,Ns).';
    [p, ~, ~] = bezier3_eval(P, s);        % 3×Ns

    % ---- DLS-IK（只控位置 + 姿态保守项）→ Q(s) ----
    Q = zeros(Ns, nJ);  Q(1,:) = row(q0);
    for k=1:Ns-1
        qk = Q(k,:);
        % 目标增量（几何点列相邻差）
        dx = p(:,k+1) - p(:,k);

        % 雅可比（线/角）与增广
        J    = geometricJacobian(robot, qk, eeName);

        % Robotics System Toolbox: geometricJacobian 的 twist 顺序是 [omega; v]
        Jang = J(1:3,:);   % angular part
        Jlin = J(4:6,:);   % linear part

        Jaug = [Jlin; w_ori*Jang];  % 6×n

        A    = (Jaug*Jaug.' + (lam^2)*eye(6));
        dq   = (Jaug.' * (A \ [dx; zeros(3,1)])).';   % 1×n

        Q(k+1,:) = qk + dq;
    end

    % ---- 几何导数 QDg / QDDg（带姿态保守项的导数口径）----
    QDg  = zeros(Ns, nJ);
    QDDg = zeros(Ns, nJ);

    for k=2:Ns-1
        qkm = Q(k-1,:); qk = Q(k,:); qkp = Q(k+1,:);
        [~, dp, ddp] = bezier3_eval(P, s(k));    % 3×1, 3×1

        % J 与 dJ/ds
        Jm  = geometricJacobian(robot, qkm, eeName);
        J0  = geometricJacobian(robot, qk,  eeName);
        Jp  = geometricJacobian(robot, qkp, eeName);

        [Jlin_m, Jang_m] = splitJ(Jm);
        [Jlin_0, Jang_0] = splitJ(J0);
        [Jlin_p, Jang_p] = splitJ(Jp);

        ds   = s(k+1) - s(k-1);
        dJlin = (Jlin_p - Jlin_m) / ds;
        dJang = (Jang_p - Jang_m) / ds;

        Jaug0  = [Jlin_0; w_ori*Jang_0];
        A0     = (Jaug0*Jaug0.' + (lam^2)*eye(6));
        qdg    = (Jaug0.' * (A0 \ [dp; zeros(3,1)])).';     % 1×n

        dJaug  = [dJlin; w_ori*dJang];
        rhs    = [ddp; zeros(3,1)] - dJaug * qdg(:);        % 6×1
        qddg   = (Jaug0.' * (A0 \ rhs)).';

        QDg(k,:)  = qdg;
        QDDg(k,:) = qddg;
    end
    % 端点抄近邻
    QDg(1,:)  = QDg(2,:);     QDg(end,:)  = QDg(end-1,:);
    QDDg(1,:) = QDDg(2,:);    QDDg(end,:) = QDDg(end-1,:);

    % ---- 轨迹句柄（几何域）----
    TRAJ.s    = s;
    TRAJ.Q    = Q;
    TRAJ.QDg  = QDg;
    TRAJ.QDDg = QDDg;
    TRAJ.eval = @(t) interp_geom(TRAJ, clamp01(t));
    TRAJ.Tf   = 1.0;
    TRAJ.kind = 'cart-bezier-dls-geom';
end

% ====================== sub-functions ======================
function [Jlin, Jang] = splitJ(J)
    % Robotics System Toolbox: [omega; v]
    Jang = J(1:3,:);
    Jlin = J(4:6,:);
end

function t1 = clamp01(t), t1 = max(0,min(1,t)); end

function [q, qdg, qddg] = interp_geom(TR, t)
    q    = interp1(TR.s, TR.Q,    t, 'linear', 'extrap');
    qdg  = interp1(TR.s, TR.QDg,  t, 'linear', 'extrap');
    qddg = interp1(TR.s, TR.QDDg, t, 'linear', 'extrap');
end

function P = make_bezier_ctrl_points(robot, q0, qf, eeName, baseName, lift_z, gamma)
    q0_row = q0(:).';  qf_row = qf(:).';
    T0 = getTransform_compat(robot, q0_row, eeName, baseName);
    Tf = getTransform_compat(robot, qf_row, eeName, baseName);
    p0 = T0(1:3,4);  pf = Tf(1:3,4);
    v  = pf - p0;    eZ = [0;0;1];
    P0 = p0;
    P3 = pf;
    P1 = p0 + gamma*v + lift_z*eZ;
    P2 = pf - gamma*v + lift_z*eZ;
    P  = [P0, P1, P2, P3]; % 3×4
end

function T = getTransform_compat(robot, q_row, eeName, baseName)
    try
        if ~isempty(baseName), T = getTransform(robot, q_row, eeName, baseName);
        else,                   T = getTransform(robot, q_row, eeName);
        end
    catch
        T = getTransform(robot, q_row, eeName);  % 旧版本兼容
    end
end

function [p, dp, ddp] = bezier3_eval(P, s)
    s = s(:); one = 1 - s;

    % 三次 Bernstein 基
    B0 = (one.^3);
    B1 = 3 .* s .* (one.^2);
    B2 = 3 .* (s.^2) .* one;
    B3 = (s.^3);
    B  = [B0 B1 B2 B3].';

    % 位置
    p  = P * B;

    % 一阶导（对 s）
    B20 = (one.^2); B21 = 2 .* s .* one; B22 = (s.^2);
    B2m = [B20 B21 B22].';
    D1  = [P(:,2)-P(:,1), P(:,3)-P(:,2), P(:,4)-P(:,3)];
    dp  = 3 * (D1 * B2m);

    % 二阶导（对 s）
    B10 = one; B11 = s; B1m = [B10 B11].';
    D2  = [P(:,3)-2*P(:,2)+P(:,1), P(:,4)-2*P(:,3)+P(:,2)];
    ddp = 6 * (D2 * B1m);
end

function v = row(v), v=v(:)'; end
function val = getdef(S, f, def), if isfield(S,f) && ~isempty(S.(f)), val=S.(f); else, val=def; end, end
