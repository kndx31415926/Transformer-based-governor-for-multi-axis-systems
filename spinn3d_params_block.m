function PB = spinn3d_params_block()
% ===== Central Param Block =====
% 所有参数集中在此处；其它脚本不得再造默认或兜底。
% 角度单位：度

%% Robot
PB.robot.object  = [];                    % 直接提供 rigidBodyTree（可留空）
PB.robot.builder = @spinn3d_build_robot;  % 构造函数句柄

%% Trajectory（demo/online 基准位形与名义限速）
PB.traj.q0_deg = [ 80;  80;  80;  80];
PB.traj.qf_deg = [-35; -20; -30; -80];
PB.traj.vmax   = 60;                     % [rad/s]
PB.traj.amax   = 120;                     % [rad/s^2]

%% Sampling / Time
PB.sample.dt = 0.01;                      % 几何采样步长（offline/预扫 α）
PB.time.Ts   = 1e-3;                      % 控制/仿真步长

%% Power / Alpha
PB.power.P_TOTAL_MAX = 15;                % 总正功上限 [W]（只计正功）
PB.alpha.alpha_min   = 0.15;              % α 下界
PB.alpha.itmax       = 12;                % 可行化二分迭代

%% Limits（CAPS）
PB.caps.tau_max    = 60*ones(4,1);
PB.caps.qd_max     = 3.0*ones(4,1);
PB.caps.qdd_max    = 6.0*ones(4,1);
PB.caps.P_axis_max = inf(4,1);
PB.caps.countRegen = false;

%% Controller gains
PB.control.Kp = 40;
PB.control.Kd = 8;

%% Stop（demo/online）
PB.stop.ee_tol      = 0.02;               % [m]
PB.stop.dq_tol_rad  = 0.03;               % [rad/s]
PB.stop.stable_time = 0.10;               % [s]

%% Friction（训练/在线统一口径）
PB.fric.B       = [0.05 0.05 0.03 0.02];  % 粘滞 [N·m·s/rad]（1×nJ 行向量）
PB.fric.Fc      = [0.4  0.3  0.2  0.1 ];  % 库伦 [N·m]
PB.fric.vel_eps = 1e-3;                   % sign 平滑阈值

%% Measurement / torque noise（demo/online）
PB.noise.sigma_q  = 1e-4;                 % [rad]
PB.noise.sigma_dq = 1e-4;                 % [rad/s]
PB.noise.tau_std  = 1e-4;                 % [N·m]

%% Paths（使用 char + fullfile，避免 "…" 与 + ）
PB.paths.root  = 'data_alpha';

% Transformer look-ahead 模型（当前主用）
PB.paths.model_alpha = fullfile(PB.paths.root, 'model_alpha_gov_trf.mat');

% Transformer look-ahead 序列数据主表（由 spinn3d_run_alpha_gov_budget_trf 生成）
PB.paths.dataset_alpha_gov_seq_master = fullfile(PB.paths.root, 'spinn3d_alpha_gov_seq_dataset_master.mat');

%% ==== Train Settings: Transformer look-ahead ====
% 用途：
%   - 训练一个「看未来 K 个 token」的序列模型（Transformer encoder），输出当前步 a_des。
%   - 训练期可加入 horizon-PINN（物理残差）+ 轻度速度偏置 + 末段减速正则。
%   - 在线推理仍然不做硬限制，只靠 governor + CTC（与你要求一致）。

PB.train_trf.enable = true;

% look-ahead 窗口（几何域 s 轴）
PB.train_trf.K = 100;        % token 数
PB.train_trf.H = 0.8;      % look-ahead 覆盖的 s 范围（单位与 TRAJ.Tf 一致）

% 优化超参
PB.train_trf.epochs = 40;
PB.train_trf.batch  = 256;
PB.train_trf.lr     = 8e-4;

% 与 nonn teacher 的软对齐（同 MLP 口径）
PB.train_trf.lambda_over  = 160;
PB.train_trf.lambda_under = 140;

% horizon-PINN + “更快/更稳”
PB.train_trf.lambda_pinn   = 1.0;   % 0=关闭
PB.train_trf.pinn_token_decay = 0.35; % >0: 越往后 token 权重越小；0=不衰减

PB.train_trf.lambda_speed  = 1;  % 越大越倾向更大 alpha（更快）
PB.train_trf.lambda_end    = 0.5;  % 末段减速正则权重（更稳）
PB.train_trf.end_phase     = 0.85;  % s_norm 超过该值开始加权
PB.train_trf.end_width     = 0.05;  % sigmoid 过渡宽度

% Transformer 结构（轻量 encoder）
PB.train_trf.d_model  = 128;
PB.train_trf.n_heads  = 4;
PB.train_trf.ffn_dim  = 256;
PB.train_trf.n_layers = 2;
PB.train_trf.dropout  = 0.0;

%% Payload（demo 固定值 + 训练采样范围）
PB.payload.mode = 'overwrite_last';              % 'overwrite_last' | 'attach_tool'
PB.payload.demo.mass    = 0.40;                  % [kg]
PB.payload.demo.com     = [0 0 0.05];            % [m]

% Robotics System Toolbox 口径：rigidBody.Inertia = [Ixx Iyy Izz Iyz Ixz Ixy]
PB.payload.demo.inertia = [0.0008 0.0008 0.0006 0 0 0]; % [Ixx Iyy Izz Iyz Ixz Ixy]

% ★ 明确声明惯量是“关于 COM”的（常见 URDF/CAD 口径）
PB.payload.demo.about   = 'com';
PB.payload.demo.mode    = 'replace';
PB.payload.demo.mount.xyz     = [0 0 0];
PB.payload.demo.mount.rpy_deg = [0 0 0];

PB.payload.range.mass     = [0.20 0.80];
PB.payload.range.com      = [-0.03 0.03; -0.03 0.03; 0.00 0.10];
PB.payload.range.box_dims = [0.03 0.06; 0.03 0.06; 0.03 0.08];
% 如需固定训练端 payload 向量，可加：
% PB.payload.train.mass    = 0.40;
% PB.payload.train.com     = [0 0 0.05];
% PB.payload.train.inertia = [0.0008 0.0008 0.0006 0 0 0];

%% ===== Alpha 数据集预算（run 脚本仅认这一组；缺一即报错）=====
PB.alpha_budget.n_traj      = 50;               % 总轨迹数
PB.alpha_budget.chunk_traj  = 50;                % 每分片轨迹数
PB.alpha_budget.pmax_mode   = 'sample';           % 'fixed' | 'sample'
p_range=10;
PB.alpha_budget.pmax_range  = [PB.power.P_TOTAL_MAX-p_range PB.power.P_TOTAL_MAX+p_range]; % [Pmin Pmax]

% --- 用 demo 轨迹自动生成采样范围 ---
q0_deg = PB.traj.q0_deg(:);   % 初始位形（deg）
qf_deg = PB.traj.qf_deg(:);   % 终止位形（deg）

margin_q0   = 10;   % q0 附近 ±10°
margin_qref = 10;   % qf 附近 ±10°（你想别的值也行）

PB.alpha_budget.q0_range_deg   = [q0_deg - margin_q0,   q0_deg + margin_q0  ];  % nJ×2
PB.alpha_budget.qref_range_deg = [qf_deg - margin_qref, qf_deg + margin_qref];  % nJ×2

PB.alpha_budget.dls_lambda  = 5e-3;
PB.alpha_budget.dls_w_ori   = 0.02;
PB.alpha_budget.bez_lift_z  = 0.15;
PB.alpha_budget.bez_gamma   = 1/3;
PB.alpha_budget.ns_min      = 200;


%% Governor / alpha regulator（严格必填）
PB.alpha_reg.beta    = 0.001;   % 一阶滤波系数 BETA，取 [0,1)，越大越“粘”
PB.alpha_reg.a_dot_up = 2000;   % α 上升速率上限 [1/s]
PB.alpha_reg.a_dot_dn = 2000;   % α 下降速率上限 [1/s]
end
