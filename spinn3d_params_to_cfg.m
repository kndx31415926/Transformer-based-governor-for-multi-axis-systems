function cfg = spinn3d_params_to_cfg(PB, mode)
% mode: 'demo_alpha' | 'offline_nonn' | 'offline_nn' | 'alpha_run'
if nargin<2, mode='demo_alpha'; end
cfg = struct();

%% Trajectory / Time / Alpha / Power
cfg.Q0_DEG = PB.traj.q0_deg;
cfg.QF_DEG = PB.traj.qf_deg;
cfg.VMAX   = PB.traj.vmax;
cfg.AMAX   = PB.traj.amax;

cfg.TS     = PB.time.Ts;
cfg.DT     = PB.sample.dt;

cfg.ALPHA_MIN   = PB.alpha.alpha_min;
cfg.ALPHA_ITMAX = PB.alpha.itmax;
cfg.P_TOTAL_MAX = PB.power.P_TOTAL_MAX;

% 模型路径（若在线 NN demo 用）
if isfield(PB,'paths') && isfield(PB.paths,'model_alpha')
    cfg.MODEL_PATH = PB.paths.model_alpha;
end

%% Robot
cfg.ROBOT = resolve_robot(PB);

%% Limits / Gains / Stop（直传，不改名）
cfg.CAPS = PB.caps;
cfg.CTRL = struct('Kp', PB.control.Kp, 'Kd', PB.control.Kd);
cfg.STOP = struct('EE_TOL', PB.stop.ee_tol, ...
                  'DQ_TOL_RAD', PB.stop.dq_tol_rad, ...
                  'STABLE_TIME', PB.stop.stable_time);

%% Friction & Noise
cfg.FRIC  = PB.fric;
cfg.NOISE = PB.noise;

%% Alpha Regulator (from PB.alpha_reg)
if isfield(PB,'alpha_reg') && ~isempty(PB.alpha_reg)
    cfg.ALPHA_REG = struct( ...
        'BETA',     PB.alpha_reg.beta, ...
        'A_DOT_UP', PB.alpha_reg.a_dot_up, ...
        'A_DOT_DN', PB.alpha_reg.a_dot_dn ...
    );
end

%% Payload 直传
cfg.PAYLOAD = struct();
if isfield(PB,'payload') && ~isempty(PB.payload)
    if isfield(PB.payload,'demo') && ~isempty(PB.payload.demo)
        cfg.PAYLOAD.DEMO = PB.payload.demo;
    end
    if isfield(PB.payload,'train') && ~isempty(PB.payload.train)
        cfg.PAYLOAD.TRAIN = PB.payload.train;
    end
    if isfield(PB.payload,'range') && ~isempty(PB.payload.range)
        cfg.PAYLOAD.RANGE = PB.payload.range;
    end
end
end

% ---- helpers ----
function robot = resolve_robot(PB)
robot = [];
if isfield(PB,'robot') && isfield(PB.robot,'object') && ~isempty(PB.robot.object)
    robot = PB.robot.object; return;
end
if isfield(PB,'robot') && isfield(PB.robot,'builder') && ~isempty(PB.robot.builder)
    robot = feval(PB.robot.builder); return;
end
try
    if evalin('base','exist(''robot'',''var'')'), robot = evalin('base','robot'); return; end
end
error('PB.robot 未提供 object 或 builder，且工作区亦无 robot。');
end
