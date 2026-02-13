function [out, fig] = spinn3d_demo_ctcgov_online_nonn(cfg)
% CTC + Governor（feasible-α 基线，在线仿真）
% - α 来源：alpha_feasible（BUS 口径 + 摩擦）；
% - 命令级功率护栏与记录/绘图均改为 BUS 口径（含再生 + 制动电阻）；
% - 绘图：base→EE 链条骨架；强制可见；返回 fig 句柄。

%% -------- cfg / 必需字段 --------\
% 确保已经加载训练好的 NN 模型（结构体名字按你 nn 脚本来）

if nargin < 1 || isempty(cfg)
    blk = [];
    if exist('spinn3d_params_block','file')==2
        blk = spinn3d_params_block();
    elseif evalin('base',"exist(''spinn3d_params_block'',''var'')==1")
        blk = evalin('base','spinn3d_params_block');
    end
    assert(~isempty(blk), '[alpha-nonn] 未提供 cfg，且未找到 spinn3d_params_block。');
    cfg = spinn3d_params_to_cfg(blk, 'demo_alpha');
end

req(cfg, {'TS','DT','Q0_DEG','QF_DEG','P_TOTAL_MAX','ALPHA_MIN','ALPHA_ITMAX'});
assert(isfield(cfg,'ROBOT') && ~isempty(cfg.ROBOT), 'cfg.ROBOT 缺失。');
assert(isfield(cfg,'FRIC')  && all(isfield(cfg.FRIC,{'B','Fc','vel_eps'})), 'cfg.FRIC 不完整。');

% 限值 / 增益 / 停止设置
limits = pick_struct(cfg, {'CAPS','LIMITS'});
gains  = pick_struct(cfg, {'CTRL','GAINS'});
req(limits, {'tau_max','qd_max','qdd_max','P_axis_max'});
req(gains,  {'Kp','Kd'});
if isfield(limits,'P_total_max') && ~isempty(limits.P_total_max)
    assert(isscalar(limits.P_total_max) && isfinite(limits.P_total_max), 'P_total_max 必须为有限标量');
else
    limits.P_total_max = cfg.P_TOTAL_MAX;
end
if ~isfield(cfg,'STOP') || ~isstruct(cfg.STOP), cfg.STOP = struct(); end
if ~isfield(cfg.STOP,'NEAR_TOL') || isempty(cfg.STOP.NEAR_TOL), cfg.STOP.NEAR_TOL = 0.02; end   % m
if ~isfield(cfg.STOP,'VEL_TOL')  || isempty(cfg.STOP.VEL_TOL),  cfg.STOP.VEL_TOL  = 0.03; end  % rad/s

%% -------- 机器人 / 起终位形 --------
robot = cfg.ROBOT; try, robot.DataFormat='row'; end
nJ    = numel(homeConfiguration(robot));
q0    = deg2rad_local(row(cfg.Q0_DEG));
qf    = unwrap_to_near(q0, deg2rad_local(row(cfg.QF_DEG)));

% 末端名（可选）
if isfield(cfg,'EE_NAME') && ~isempty(cfg.EE_NAME)
    eeName = cfg.EE_NAME;
else
    eeName = robot.BodyNames{end};
end

% -------- 注入 payload（若 cfg.PAYLOAD.DEMO 提供） --------
pay_cfg = [];
if isfield(cfg,'PAYLOAD') && isstruct(cfg.PAYLOAD) && isfield(cfg.PAYLOAD,'DEMO')
    pay_cfg = cfg.PAYLOAD.DEMO;
end
% ★ 对比/复现实验时务必固定 payload 注入口径，避免 nn/nonn 机器人模型不一致
if isstruct(pay_cfg)
    if ~isfield(pay_cfg,'about') || isempty(pay_cfg.about), pay_cfg.about = 'com'; end
    if ~isfield(pay_cfg,'mode')  || isempty(pay_cfg.mode),  pay_cfg.mode  = 'replace'; end
end
[robot, ~] = inject_payload_last_body(robot, pay_cfg);

%% -------- 笛卡尔路径 → 关节参考（只控位置 IK） --------
traj  = make_traj_cart_bezier_dls(robot, q0, qf, limits, cfg);

%% -------- 控制器（feasible α 模式） --------
areg = struct('BETA',[], 'A_DOT_UP',[], 'A_DOT_DN',[]);
if isfield(cfg,'ALPHA_REG')
    fns = fieldnames(areg);
    for i=1:numel(fns)
        if isfield(cfg.ALPHA_REG,fns{i})
            areg.(fns{i}) = cfg.ALPHA_REG.(fns{i});
        end
    end
end

bus = get_bus_from_cfg(cfg, limits.P_total_max);  % BUS 结构（总线效率/制动电阻等），用于控制器与功率护栏统一口径

opts = struct();
opts.Ts      = cfg.TS;
opts.fric    = cfg.FRIC;
opts.bus     = bus;
opts.alphaReg= areg;     % ★ 非常重要：给 controller
opts.nnAlpha = struct( ...
    'enable',      true, ...
    'mode',        'feasible', ...
    'alpha_floor', cfg.ALPHA_MIN, ...
    'itmax',       cfg.ALPHA_ITMAX ...
);
ctl = spinn3d_controller_ctcgov(robot, traj, gains, limits, opts);

%% -------- 预算仿真步数 --------
Tcap = traj.Tf/max(cfg.ALPHA_MIN,1e-6) + 0.5;
Nt   = max(2, ceil(Tcap/cfg.TS) + 1);

%% -------- 常量 --------
B  = row(cfg.FRIC.B);   Fc = row(cfg.FRIC.Fc);   ve = cfg.FRIC.vel_eps;
tau_max   = vec_lim(limits,'tau_max',    nJ, inf);
P_axismax = vec_lim(limits,'P_axis_max', nJ, inf);

% -------- BUS 口径参数（可由 cfg.BUS 覆盖） -------- % BUS:

%% -------- 仿真主循环 --------
t=0; s=0; [q_ref0, dq_ref0, ddq_ref0] = traj.eval(0); %#ok<ASGLU>
q = row(q_ref0); dq = zeros(1,nJ);

log.t=zeros(Nt,1); log.s=zeros(Nt,1);
log.q=zeros(Nt,nJ); log.dq=zeros(Nt,nJ);
log.tau=zeros(Nt,nJ); log.alpha=zeros(Nt,1);
log.Pi_raw=zeros(Nt,nJ); log.Ptot_pos=zeros(Nt,1);  % 兼容旧字段
log.Pgrid=zeros(Nt,1); log.Pdump=zeros(Nt,1);       % BUS:
log.ee=nan(Nt,3);

ee_start= ee_pos(robot,q0,eeName);
ee_goal = ee_pos(robot,qf,eeName);

t_reach=NaN;

for k=1:Nt
    [q_r, dq_r, ddq_r] = traj.eval(s); q_r=row(q_r); dq_r=row(dq_r); ddq_r=row(ddq_r);

    % 控制器（内部求 α，返回 info.a_max）
    [tau_cmd, info] = ctl.step(t, q, dq, q_r, dq_r, ddq_r, robot, s);
    tau_cmd = row(tau_cmd);

    % ===== 命令级护栏：τ 限幅 + BUS 口径功率缩放 ===== % BUS:
    tau_cmd = clamp_tau(tau_cmd, tau_max);
    tau_cmd = clamp_power_axis_total_bus(tau_cmd, dq, P_axismax, bus);

    % ===== 执行级：仅扭矩限幅 =====
    tau_m = clamp_tau(tau_cmd, tau_max);

    % 被控对象动力学积分（含摩擦）
    Mq = massMatrix(robot, q); c = velocityProduct(robot, q, dq); g = gravityTorque(robot, q);
    tau_fric = B.*dq + Fc.*tanh(dq/max(ve,1e-9));
    ddq = (Mq \ (tau_m(:) - c(:) - g(:) - tau_fric(:))).';
    dq  = dq + cfg.TS*ddq;
    q   = q  + cfg.TS*dq;

    % —— 几何时间推进：只用控制器的 α —— 
    s = min(traj.Tf, s + cfg.TS * info.a_max);

    % 记录（含 BUS 口径功率） % BUS:
    Pi_raw = tau_m .* dq;
    [Ppos,Pneg,Pgrid,Pdump,~] = spinn3d_bus_power(tau_m, dq, bus);
    log.t(k)=t; log.s(k)=s; log.q(k,:)=q; log.dq(k,:)=dq; log.tau(k,:)=tau_m;
    log.alpha(k)=info.a_max; log.Pi_raw(k,:)=Pi_raw;
    log.Ptot_pos(k)=Ppos; log.Pgrid(k)=Pgrid; log.Pdump(k)=Pdump;
    Tee = getTransform(robot, q, eeName); log.ee(k,:) = tform2trvec_safe(Tee);

    % ===== 停止：几何时间到尾 + 末端近邻 + 低速 =====
    ee_err = norm(log.ee(k,:) - ee_goal);
    if (s >= traj.Tf) && (ee_err <= cfg.STOP.NEAR_TOL) && (norm(dq) <= cfg.STOP.VEL_TOL)
        t_reach = t;
        log = truncate_log(log, k);
        break;
    end

    t = t + cfg.TS;
end

%% -------- 绘图（总是绘；显式可见；robust 骨架） --------
fig = figure('Name','CTC+Gov (online nonn, BUS power)','Visible','on');
tlo = tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');

ax3 = nexttile(tlo,[2 1]); hold(ax3,'on'); grid(ax3,'on'); axis(ax3,'equal'); view(ax3,135,20);
[X0,Y0,Z0] = chain_skeleton_xyz(robot, q0, eeName);
if ~isempty(X0), plot3(ax3, X0, Y0, Z0, '-o','LineWidth',1.0,'DisplayName','Initial'); end
try
    show(robot, log.q(end,:), 'Parent', ax3, 'PreservePlot', true, 'Frames','off','Visuals','on');
catch
end
[Xf,Yf,Zf] = chain_skeleton_xyz(robot, log.q(end,:), eeName);
if ~isempty(Xf), plot3(ax3, Xf, Yf, Zf, 'm-o','MarkerFaceColor','m','LineWidth',1.0,'DisplayName','Final'); end
plot3(ax3, log.ee(:,1), log.ee(:,2), log.ee(:,3), 'c-','LineWidth',1.5,'DisplayName','EE traj');
plot3(ax3, ee_start(1),ee_start(2),ee_start(3),'go','MarkerFaceColor','g','DisplayName','Start');
plot3(ax3, ee_goal(1), ee_goal(2), ee_goal(3),'co','MarkerFaceColor','c','DisplayName','Target');
plot3(ax3, log.ee(end,1),log.ee(end,2),log.ee(end,3),'ro','MarkerFaceColor','r','DisplayName','Final EE');
xlabel(ax3,'X [m]'); ylabel(ax3,'Y [m]'); zlabel(ax3,'Z [m]');
title(ax3,'EE Trajectory + Robot Skeleton (Initial/Final; nonn)'); legend(ax3,'Location','best');

axU = nexttile(tlo); hold(axU,'on'); grid(axU,'on');
for j=1:nJ, plot(axU, log.t, log.Pi_raw(:,j), 'DisplayName', sprintf('J%d',j)); end
xlabel(axU,'t [s]'); ylabel(axU,'Joint Power P_j [W]');
title(axU,'Per-Joint Power'); legend(axU,'show','Location','best'); xlim(axU,[log.t(1),log.t(end)]);

axD = nexttile(tlo); hold(axD,'on'); grid(axD,'on');
plot(axD, log.t, log.Pgrid, 'DisplayName','P_{grid}');
win = max(1, round(0.10/cfg.TS)); plot(axD, log.t, movmean(log.Pgrid,win), 'LineWidth',1.6, 'DisplayName','P_{grid} avg');
plot(axD, log.t, log.Pdump, 'DisplayName','P_{dump}');
ymax=max([log.Pgrid; log.Pdump])*1.2; if ~isfinite(ymax)||ymax<=0, ymax=1; end
ylim(axD,[0,ymax]); xlim(axD,[log.t(1), log.t(end)]);
if limits.P_total_max <= ymax, yline(axD, limits.P_total_max, '--r','P_{grid,max}'); end
xlabel(axD,'t [s]'); ylabel(axD,'Power [W]'); title(axD,'Bus Power (grid & dump)'); legend(axD,'show','Location','best');

drawnow;

%% -------- 输出 --------
out = struct('log',log,'limits',limits,'gains',gains,'cfg',cfg,'t_reach',t_reach);
end

%% ==================== helpers ====================
function req(S, names), if ischar(names), names={names}; end, for i=1:numel(names), assert(isfield(S,names{i}) && ~isempty(S.(names{i})), 'cfg 缺少字段：%s', names{i}); end, end
function def = getfield_def(S, field, def), if isfield(S,field)&&~isempty(S.(field)), def=S.(field); end, end
function S = pick_struct(cfg, keys), S=[]; for i=1:numel(keys), k=keys{i}; if isfield(cfg,k)&&~isempty(cfg.(k))&&isstruct(cfg.(k)), S=cfg.(k); return; end, end, error('cfg 缺少结构体：%s', strjoin(keys,'/')); end
function v = row(v), v=v(:)'; end
function y = deg2rad_local(x), y=(pi/180).*x; end
function p = tform2trvec_safe(T), try, p=tform2trvec(T); catch, p=T(1:3,4).'; end, end
function log = truncate_log(log,k), fn=fieldnames(log); for i=1:numel(fn), v=log.(fn{i}); if size(v,1)>=k, log.(fn{i})=v(1:k,:); end, end, end
function p=ee_pos(robot,qrow,eeName), T=getTransform(robot,qrow,eeName); p=tform2trvec_safe(T); end

% —— base→EE 链条骨架（鲁棒） ——
function [X,Y,Z] = chain_skeleton_xyz(robot, qrow, eeName)
    try, robot.DataFormat='row'; end
    X=[];Y=[];Z=[];
    try
        names = {};
        b = getBody(robot, eeName);
        while ~strcmp(b.Name, robot.BaseName)
            names{end+1} = b.Name; %#ok<AGROW>
            b = getBody(robot, b.Parent);
        end
        names = fliplr(names);
    catch
        names = robot.BodyNames;
    end
    if isempty(names), return; end
    for i=1:numel(names)
        b = getBody(robot, names{i});
        if strcmp(b.Joint.Type,'fixed'), continue; end
        Tb = getTransform(robot, qrow, names{i});
        p  = tform2trvec_safe(Tb);
        X(end+1)=p(1); Y(end+1)=p(2); Z(end+1)=p(3); %#ok<AGROW>
    end
    Tee = getTransform(robot, qrow, eeName);
    pee = tform2trvec_safe(Tee);
    X(end+1)=pee(1); Y(end+1)=pee(2); Z(end+1)=pee(3);
end

function v = vec_lim(S, field, n, def)
    if ~isfield(S,field) || isempty(S.(field)), v = def*ones(1,n); return; end
    v = S.(field); if isscalar(v), v = repmat(v,1,n); end
    v(~isfinite(v)) = def; v = v(:)'; 
end
function tau = clamp_tau(tau, tau_max)
    for j=1:numel(tau)
        if isfinite(tau_max(j))
            tau(j) = max(-tau_max(j), min(tau_max(j), tau(j)));
        end
    end
end

% ---------- BUS 口径：总功率夹紧（含再生） ---------- % BUS:
function tau = clamp_power_axis_total_bus_local(tau, vel, P_axis_max, bus)
    for j=1:numel(tau)
        Pj = tau(j) * vel(j);
        if Pj > P_axis_max(j)
            tau(j) = P_axis_max(j) / max(vel(j), 1e-9);
        end
    end
    [Ppos,Pneg,Pgrid,~,~] = spinn3d_bus_power(tau, vel, bus);
    if Pgrid > bus.P_total_max + 1e-9
        denom = max(Ppos - bus.eta_share*Pneg, 1e-12);
        gamma = bus.P_total_max / denom;
        tau = tau * gamma;
    end
end
% ---------- BUS 结构体：从 cfg 读取并补默认 ---------- % BUS:
function bus = get_bus_from_cfg_local(cfg, P_total_max)
    bus = struct('eta_share',1.0,'P_brk_peak',inf,'P_total_max',P_total_max);
    if isfield(cfg,'BUS') && isstruct(cfg.BUS)
        if isfield(cfg.BUS,'eta_share') && ~isempty(cfg.BUS.eta_share), bus.eta_share = cfg.BUS.eta_share; end
        if isfield(cfg.BUS,'P_brk_peak') && ~isempty(cfg.BUS.P_brk_peak), bus.P_brk_peak = cfg.BUS.P_brk_peak; end
    end
end
function qn = unwrap_to_near(qprev, qcur)
    d  = row(qcur) - row(qprev);
    qn = row(qprev) + atan2(sin(d), cos(d));
end
function [robot_out, pay_vec] = inject_payload_last_body(robot_in, pay)
% 注入 payload（若提供），返回 1×10：
%   [m, com(1:3), Ixx Iyy Izz Iyz Ixz Ixy]   (Robotics System Toolbox 口径)
%
% pay 支持：
%   - 1×10 数值向量
%   - struct，字段 mass/com/inertia，可选 about/mode
%
% about:
%   - 'com'    : I6 相对 payload COM（默认；会平行轴换算到 body 原点）
%   - 'origin' : I6 相对 body 原点（不做平行轴换算）
% mode:
%   - 'replace': 覆盖末端刚体惯量（默认）
%   - 'add'    : 在末端刚体基础上累加

    robot_out = robot_in;
    pay_vec   = zeros(1,10);
    if isempty(pay), return; end

    if isnumeric(pay)
        v = double(pay(:)).';
        if numel(v) < 10, v = [v, zeros(1,10-numel(v))]; end
        m  = v(1); com = v(2:4); I6 = v(5:10);
        about = 'com';    % ★ 默认按 COM 惯量解释
        mode  = 'replace';
    elseif isstruct(pay)
        m   = getfield_def(pay,'mass',0);
        com = getfield_def(pay,'com',[0 0 0]);
        I6  = getfield_def(pay,'inertia',[0 0 0 0 0 0]);
        about = lower(getfield_def(pay,'about','com'));
        mode  = lower(getfield_def(pay,'mode','replace'));
    else
        return;
    end
    pay_vec = [double(m), double(com(:)).', double(I6(:)).'];

    try
        b  = robot_out.Bodies{end};
        m0 = double(b.Mass);
        c0 = double(b.CenterOfMass(:)).';
        I0 = I6ToMat(double(b.Inertia(:)).');

        Ic = I6ToMat(double(I6(:)).');
        if strcmpi(about,'com')
            r  = double(com(:)); r = r(:);
            I1 = Ic + double(m) * ((r.'*r)*eye(3) - (r*r.'));
        else
            I1 = Ic;
        end

        switch mode
            case 'add'
                m_new  = m0 + m;
                if m_new <= 0
                    c_new = [0 0 0];
                else
                    c_new = (m0*c0 + m*com(:).') / m_new;
                end
                I_new = I0 + I1;
            otherwise % 'replace'
                m_new  = m;
                c_new  = com(:).';
                I_new  = I1;
        end

        b.Mass         = m_new;
        b.CenterOfMass = c_new;
        b.Inertia      = ImatTo6(I_new).';
    catch
        % 静默跳过（仅影响动力学）
    end
end

function I = I6ToMat(v6)
% RST: v6 = [Ixx Iyy Izz Iyz Ixz Ixy]
    v6 = double(v6(:)).';
    Ixx=v6(1); Iyy=v6(2); Izz=v6(3);
    Iyz=v6(4); Ixz=v6(5); Ixy=v6(6);
    I = [ Ixx Ixy Ixz;
          Ixy Iyy Iyz;
          Ixz Iyz Izz ];
end

function v6 = ImatTo6(I)
% RST: v6 = [Ixx Iyy Izz Iyz Ixz Ixy]
    v6 = [ I(1,1), I(2,2), I(3,3), I(2,3), I(1,3), I(1,2) ];
end
