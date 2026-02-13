function x = spinn3d_features_alpha_plus(q0, dq0, ddq0, varargin)
% Xα++：动力学 + spinn 风格 + 直接 Pmax（只计正功）
% 兼容两种签名：
%   A) (q0,dq0,ddq0,  Pmax, robot, kin,  caps, payload4)
%   B) (q0,dq0,ddq0,  robot, payload4, Pmax, caps)
% 并自适应 rigidBodyTree.DataFormat（'row' / 'column'）

% ---------- 解析入参（自动识别 A/B 口径） ----------
Pmax = []; robot = []; kin = []; caps = struct(); payload4 = [];
need = @(n) (numel(varargin)>=n && ~isempty(varargin{n}));

if ~isempty(varargin)
    v1 = varargin{1};
    if is_rbt(v1) || (isobject(v1) && ismethod(v1,'inverseDynamics'))
        % 口径 B：robot 在第一个额外参数
        robot    = v1;
        if need(2), payload4 = varargin{2}; end
        if need(3), Pmax     = varargin{3}; end
        if need(4), caps     = varargin{4}; end
    else
        % 口径 A：Pmax 在第一个额外参数
        Pmax = v1;
        if need(2), robot    = varargin{2}; end
        if need(3), kin      = varargin{3}; end %#ok<NASGU>
        if need(4), caps     = varargin{4}; end
        if need(5), payload4 = varargin{5}; end
    end
end
assert(~isempty(robot), 'spinn3d_features_alpha_plus: 缺少 robot 参数。');

% ---------- 统一到 robot 的 DataFormat 形状 ----------
fmt = getDataFormatSafe(robot);   % 'row' | 'column'（取不到则默认 row）
[qf,dqf,ddqf] = castToFmt(fmt, q0, dq0, ddq0);
asRow = @(v) reshape(v,1,[]);     % 拼特征时统一为行向量
asCol = @(v) reshape(v,[],1);

% ---------- 动力学：τ_ff、正功、重力项（只计正功） ----------
tau_ff0 = inverseDynamics(robot, qf, dqf, ddqf);    % 与 fmt 一致
tau_row = asRow(tau_ff0);
dq_row  = asRow(dqf);
p_axis0 = max(tau_row .* dq_row, 0);
Pff0    = sum(p_axis0);

n = numel(asRow(qf));
z = zeros(1,n); if strcmp(fmt,'column'), z = asCol(z); end
gq0 = inverseDynamics(robot, qf, z, z);
g_row = asRow(gq0);

% ---------- 负载 / 限值 ----------
if isempty(payload4), payload4 = getPayload4OrZero(robot); end
limvec = double(Pmax);   % 允许为空；为空时这里就是 []
if isstruct(caps)
    if isfield(caps,'tau_max'),    limvec = [limvec, caps.tau_max(:)'];    end
    if isfield(caps,'P_axis_max'), limvec = [limvec, caps.P_axis_max(:)']; end
end

% ---------- spinn 风格摘要：末端线速/范数/三角/符号/限位裕度 ----------
vx = zeros(1,3);
try
    ee = getEndEffectorNameSafe(robot);
    J  = geometricJacobian(robot, qf, ee);   % 6×n
    Jv = J(4:6,:);
    vx = asRow(Jv*asCol(asRow(dqf)));
catch
    % 取不到雅可比时，用 dq 的范数代替尺度
    vx = [norm(asRow(dqf)), 0, 0];
end

q_row  = asRow(qf);
sinq   = sin(q_row);  cosq = cos(q_row);  sgn_dq = sign(dq_row);
norms  = [norm(dq_row), norm(vx), norm(tau_row)];

dist_to_jlim = zeros(1, 2*numel(q_row));
try
    [lo,hi] = getJointLimitsRST(robot);
    if ~isempty(lo)
        dist_to_jlim = [ (q_row-lo)./max(hi-lo,1e-9), (hi-q_row)./max(hi-lo,1e-9) ];
    end
catch
end

% ---------- 拼接（固定顺序，便于离线标准化） ----------
x = double([ ...
     q_row, dq_row, asRow(ddqf), ...
     tau_row, p_axis0, Pff0, ...
     g_row, ...
     payload4, ...
     limvec, ...
     vx, norms, ...
     sinq, cosq, sgn_dq, ...
     dist_to_jlim ...
]);
end

% ====================== 工具函数 ======================
function tf = is_rbt(x)
tf = isa(x,'rigidBodyTree') || (isobject(x) && ismethod(x,'inverseDynamics'));
end
function fmt = getDataFormatSafe(robot)
try, fmt = robot.DataFormat; if ~ischar(fmt), fmt='row'; end
catch, fmt = 'row'; end
end
function [qf,dqf,ddqf] = castToFmt(fmt,q,dq,ddq)
if strcmp(fmt,'column')
    qf = q(:);  dqf = dq(:);  ddqf = ddq(:);
else
    qf = q(:)'; dqf = dq(:)'; ddqf = ddq(:)';
end
end
function payload4 = getPayload4OrZero(robot)
payload4 = [0 0 0 0];
try
    if isfield(robot,'tool')
        if isfield(robot.tool,'mass'), payload4(1)=robot.tool.mass; end
        if isfield(robot.tool,'com') && numel(robot.tool.com)>=3
            payload4(2:4) = robot.tool.com(1:3);
        end
    end
catch, end
end
function ee = getEndEffectorNameSafe(robot)
try, ee = robot.BodyNames{end}; catch, ee = robot.BodyNames{end}; end
end
function [lo,hi] = getJointLimitsRST(robot)
try
    names = robot.BodyNames;
    lo=[]; hi=[];
    for i=1:numel(names)
        b = robot.getBody(names{i});
        if ~strcmp(b.Joint.Type,'fixed')
            lim = b.Joint.PositionLimits; if isempty(lim), lim=[-inf,inf]; end
            lo(end+1)=lim(1); hi(end+1)=lim(2); %#ok<AGROW>
        end
    end
catch
    lo=[]; hi=[];
end
end
