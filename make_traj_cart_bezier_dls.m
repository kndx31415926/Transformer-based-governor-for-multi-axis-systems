function TRAJ = make_traj_cart_bezier_dls(robot, q0, qf, limits, cfg)
% Bézier + DLS-IK（只控位置），带姿态保守项与速度限幅；时间域轨迹（Ts 定步长）。
% 输出：
%   TRAJ.Tf    = (M-1)*Ts          （α=1 时的名义总时长）
%   TRAJ.eval(t)->[q,dq,ddq]       （线性插值，t∈[0, Tf]）
%
% 依赖字段：
%   cfg.TS                采样周期
%   limits.qd_max(1×n)    关节速度上限（用于 IK 步内缩放）
%
% 轨迹口径对应你 NN Demo 中的实现（w_ori=0.02, λ≈0.14, qstep≈0.6°，SG 平滑等）。

    ee  = robot.BodyNames{end};
    Ts  = cfg.TS;

    % ---- 起终位末端位姿 → 控制点（抬升 + 平滑）----
    T0 = getTransform(robot,q0,ee); T1 = getTransform(robot,qf,ee);
    p0 = T0(1:3,4); pf = T1(1:3,4);
    z_clear = 0.14; beta=0.25; gamma=0.75;
    Zlift = max([p0(3), pf(3)]) + z_clear;
    P0 = p0;
    P1 = [ (1-beta )*p0(1)+beta*pf(1) ;
           (1-beta )*p0(2)+beta*pf(2) ;
           Zlift ];
    P2 = [ (1-gamma)*p0(1)+gamma*pf(1) ;
           (1-gamma)*p0(2)+gamma*pf(2) ;
           Zlift ];
    P3 = pf;

    % ---- 弧长等距 + 缓起缓停采样（空间点列）----
    step_mm = 0.3;
    PP  = sample_bezier_cubic(P0,P1,P2,P3, step_mm); % 3×M
    M   = size(PP,2);

    % ---- DLS-IK（只控位置 + 姿态保守项）----
    prm.lambda    = 0.14;
    prm.qstep_deg = 0.6;
    prm.pos_tol   = 5e-4;
    prm.innerMax  = 60;
    prm.dropRatio = 0.95;

    Q = zeros(M, numel(q0)); Q(1,:) = q0(:).';
    for k=2:M
        q    = Q(k-1,:); ptgt = PP(:,k); gain = 1.0;
        for it=1:prm.innerMax
            Tnow = getTransform(robot,q,ee); pnow = Tnow(1:3,4);
            dp   = ptgt - pnow; if norm(dp) <= prm.pos_tol, break; end

            J    = geometricJacobian(robot,q,ee);
            Jlin = J(4:6,:);
            % 条件数自适应阻尼
            s = svd(Jlin); kappa = (s(1)/max(s(end),1e-9));
            lam_eff = prm.lambda * (1 + 0.5*max(0,kappa-10)/10);

            % —— 姿态保守项（目标角速度 0）——
            w_ori = 0.02;
            Jang  = J(1:3,:);
            Jaug  = [Jlin; w_ori*Jang];
            d_aug = [gain*dp; zeros(3,1)];
            A     = (Jaug*Jaug.' + (lam_eff^2)*eye(6));
            dqv   = Jaug.' * (A \ d_aug);

            % —— 速度上限缩放（按 Ts 折算）——
            qd_lim = vec_lim(limits,'qd_max', numel(q0), inf);
            scLim  = max(1, max(abs(dqv(:)/Ts) ./ max(qd_lim(:),1e-9)));
            dqv    = dqv / scLim;

            % —— 单步角度限制（~0.6°）+ 解缠 —— 
            scl = max(1, max(abs(dqv(:)))/deg2rad(0.6));
            qn  = unwrap_to_near(q, q + (dqv(:).'/scl));

            pnew = getTransform(robot,qn,ee); pnew = pnew(1:3,4);
            if norm(ptgt - pnew) < prm.dropRatio*norm(dp) || gain < 1e-3
                q = qn;
            else
                gain = gain * 0.5; continue;
            end
            if norm(ptgt - pnew) <= prm.pos_tol, break; end
        end
        Q(k,:) = q;
    end

    % ---- 轻度平滑 ----
    if M>=3, Q = movmean(Q,[1 1],1); end
    if exist('sgolayfilt','file') && M>=21
        for j=1:size(Q,2), Q(:,j) = sgolayfilt(Q(:,j),3,21); end
    end

    % ---- 固定 Ts 的有限差分 ----
    dQ  = zeros(M,size(Q,2)); ddQ = zeros(M,size(Q,2));
    if M>=3
        dQ(2:M-1,:)  = (Q(3:M,:) - Q(1:M-2,:)) / (2*Ts);
        dQ(1,:)      = (Q(2,:)   - Q(1,:)    ) / Ts;
        dQ(M,:)      = (Q(M,:)   - Q(M-1,:)  ) / Ts;
        ddQ(2:M-1,:) = (Q(3:M,:) - 2*Q(2:M-1,:) + Q(1:M-2,:)) / (Ts^2);
        ddQ(1,:)     = ddQ(2,:);  ddQ(M,:)   = ddQ(M-1,:);
    end

    TRAJ.Tf   = (M-1)*Ts;
    TRAJ.eval = @(t) eval_cart_joint_traj(t, Q, dQ, ddQ, Ts);
end

%% ====== 子函数（仅本文件内部使用） ======
function [q,dq,ddq] = eval_cart_joint_traj(t, Q, dQ, ddQ, Ts)
    M = size(Q,1);
    u = t / Ts;
    i = floor(u) + 1; i = max(1, min(M-1, i));
    lam = u - (i-1);
    q   = (1-lam)*Q(i,:)   + lam*Q(i+1,:);
    dq  = (1-lam)*dQ(i,:)  + lam*dQ(i+1,:);
    ddq = (1-lam)*ddQ(i,:) + lam*ddQ(i+1,:);
end

function P = sample_bezier_cubic(P0,P1,P2,P3, step_mm)
    step = max(step_mm,0.2)/1000;
    tt   = linspace(0,1,4000);
    B0=(1-tt).^3; B1=3*(1-tt).^2.*tt; B2=3*(1-tt).*tt.^2; B3=tt.^3;
    C   = P0*B0 + P1*B1 + P2*B2 + P3*B3;
    ds  = vecnorm(diff(C,1,2)); s = [0, cumsum(ds)];  L = s(end);
    N    = max(2, ceil(L/step));
    u    = linspace(0,1,N);
    u_e  = 0.5 - 0.5*cos(pi*u);                 % 缓起缓停再参数化
    t_eq = interp1(s./max(L,eps), tt, u_e, 'linear','extrap');
    B0=(1-t_eq).^3; B1=3*(1-t_eq).^2.*t_eq; B2=3*(1-t_eq).*t_eq.^2; B3=t_eq.^3;
    P  = P0*B0 + P1*B1 + P2*B2 + P3*B3;        % 3×N
end

function qn = unwrap_to_near(qprev, qcur)
    d  = row(qcur) - row(qprev);
    qn = row(qprev) + atan2(sin(d), cos(d));
end

function v = row(v), v = v(:)'; end
function v = vec_lim(S, field, n, def)
    if ~isfield(S,field) || isempty(S.(field)), v = def*ones(1,n); return; end
    v = S.(field); if isscalar(v), v = repmat(v,1,n); end
    v(~isfinite(v)) = def; v = v(:)'; 
end
