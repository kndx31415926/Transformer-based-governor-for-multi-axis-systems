function [robot, kin] = spinn3d_robot(params)
% SPINN3D_ROBOT  4-DoF manipulator: base yaw (Z) + 3 pitch (Y).
% 惯量口径：rigidBody.Inertia 一律为“相对 body 原点”的惯量张量。
% 做法：先在 COM 处计算惯量 -> 再用平行轴定理（张量版）移到原点。
% 本版本在 tool 段支持从 params.payload 注入末端负载。

% ===== 参数整理 =====
if nargin < 1 || isempty(params), params = struct(); end
if ~isfield(params,'L'),           params.L = [0.24 0.214 0.324]; end
if ~isfield(params,'gravity'),     params.gravity = [0 0 -9.81];   end
if ~isfield(params,'base_mass'),   params.base_mass = 0.24;          end
if ~isfield(params,'base_radius'), params.base_radius = 0.15;        end
if ~isfield(params,'base_height'), params.base_height = 0.04;        end
if ~isfield(params,'base_z'),      params.base_z = params.base_height; end
if ~isfield(params,'mass'),        params.mass = [2.8 2.2 1.6];     end
if ~isfield(params,'com_ratio'),   params.com_ratio = [0.5 0.5 0.5]; end
if ~isfield(params,'link_radius'), params.link_radius = 0.015;       end
if ~isfield(params,'jposlim'),     params.jposlim = repmat([-inf inf], 4, 1); end

L  = params.L(:).';   m  = params.mass(:).';  c  = params.com_ratio(:).';
R  = params.link_radius;  bz = params.base_z;

% ===== 机器人骨架 =====
robot = rigidBodyTree('DataFormat','row','MaxNumBodies',6);
robot.Gravity = params.gravity(:).';

% ---- Base yaw ----
b1 = rigidBody('link_baseYaw');
j1 = rigidBodyJoint('joint_baseYaw','revolute'); j1.JointAxis = [0 0 1];
j1.PositionLimits = params.jposlim(1,:); setFixedTransform(j1, eye(4)); b1.Joint = j1;

m_base = max(params.base_mass, 1e-3);
rb = params.base_radius; hb = params.base_height;
[Ixx_b_com, Iyy_b_com, Izz_b_com] = cylInertiaCOM_Z(m_base, rb, hb);
b1.Mass = m_base; b1.CenterOfMass = [0 0 0];
b1.Inertia = [Ixx_b_com, Iyy_b_com, Izz_b_com, 0, 0, 0];
addBody(robot, b1, robot.BaseName);
try, addVisual(b1,'Cylinder',[rb hb], eye(4)); end

% ---- Shoulder link (Y) ----
b2 = rigidBody('link_shoulder');
j2 = rigidBodyJoint('joint_shoulder','revolute'); j2.JointAxis = [0 1 0];
j2.PositionLimits = params.jposlim(2,:); setFixedTransform(j2, trvec2tform([0 0 bz])); b2.Joint = j2;
m2 = max(m(1),1e-3); L2 = L(1); r2 = [c(1)*L2, 0, 0];
[Ixx2c, Iyy2c, Izz2c] = rodInertiaCOM_X(m2, R, L2);
I2o = shiftInertiaCOMtoOrigin([Ixx2c Iyy2c Izz2c 0 0 0], m2, r2);
b2.Mass = m2; b2.CenterOfMass = r2; b2.Inertia = I2o;
addBody(robot, b2, 'link_baseYaw');
try, addVisual(b2,'Cylinder',[R L2], troty(-pi/2)*trvec2tform([L2/2 0 0])); end

% ---- Elbow link (Y) ----
b3 = rigidBody('link_elbow');
j3 = rigidBodyJoint('joint_elbow','revolute'); j3.JointAxis = [0 1 0];
j3.PositionLimits = params.jposlim(3,:); setFixedTransform(j3, trvec2tform([L2 0 0])); b3.Joint = j3;
m3 = max(m(2),1e-3); L3 = L(2); r3 = [c(2)*L3, 0, 0];
[Ixx3c, Iyy3c, Izz3c] = rodInertiaCOM_X(m3, R, L3);
I3o = shiftInertiaCOMtoOrigin([Ixx3c Iyy3c Izz3c 0 0 0], m3, r3);
b3.Mass = m3; b3.CenterOfMass = r3; b3.Inertia = I3o;
addBody(robot, b3, 'link_shoulder');
try, addVisual(b3,'Cylinder',[R L3], troty(-pi/2)*trvec2tform([L3/2 0 0])); end

% ---- Wrist link (Y) ----
b4 = rigidBody('link_wrist');
j4 = rigidBodyJoint('joint_wrist','revolute'); j4.JointAxis = [0 1 0];
j4.PositionLimits = params.jposlim(4,:); setFixedTransform(j4, trvec2tform([L3 0 0])); b4.Joint = j4;
m4 = max(m(3),1e-3); L4 = L(3); r4 = [c(3)*L4, 0, 0];
[Ixx4c, Iyy4c, Izz4c] = rodInertiaCOM_X(m4, R, L4);
I4o = shiftInertiaCOMtoOrigin([Ixx4c Iyy4c Izz4c 0 0 0], m4, r4);
b4.Mass = m4; b4.CenterOfMass = r4; b4.Inertia = I4o;
addBody(robot, b4, 'link_elbow');
try, addVisual(b4,'Cylinder',[R L4], troty(-pi/2)*trvec2tform([L4/2 0 0])); end

% ---- Tool (fixed) ----
tool = rigidBody('tool');
jt   = rigidBodyJoint('fix_tool','fixed'); setFixedTransform(jt, trvec2tform([L4 0 0]));
tool.Joint = jt; addBody(robot, tool, 'link_wrist');
try, addVisual(tool,'Sphere',0.02,eye(4)); end

% === 末端负载注入（从 params.payload 读取；默认无负载，mass>0 时注入） ===
tool.Mass         = 0;
tool.CenterOfMass = [0 0 0];
tool.Inertia      = [0 0 0 0 0 0];
if isfield(params,'payload') && isstruct(params.payload) ...
        && isfield(params.payload,'mass') && params.payload.mass > 0
    M    = double(params.payload.mass);
    rcom = [0 0 0];
    if isfield(params.payload,'com') && ~isempty(params.payload.com)
        rcom = double(params.payload.com(:)).';
    end
    if isfield(params.payload,'I6_com') && ~isempty(params.payload.I6_com)
        I6_com = double(params.payload.I6_com(:)).';
        if numel(I6_com) ~= 6, error('params.payload.I6_com must be 1x6'); end
    else
        shape = 'box';
        if isfield(params.payload,'shape') && ~isempty(params.payload.shape)
            shape = lower(string(params.payload.shape));
        end
        dims = struct();
        if isfield(params.payload,'dims') && ~isempty(params.payload.dims)
            dims = params.payload.dims;
        end
        default.dims_box = struct('a',0.06,'b',0.04,'c',0.03);
        default.dims_cyl = struct('r',0.015,'h',0.06);
        default.dims_sph = struct('r',0.03);
        switch char(shape)
            case 'box'
                if ~isfield(dims,'a'), dims.a = default.dims_box.a; end
                if ~isfield(dims,'b'), dims.b = default.dims_box.b; end
                if ~isfield(dims,'c'), dims.c = default.dims_box.c; end
                Ixxc = (1/12)*M*(dims.b^2 + dims.c^2);
                Iyyc = (1/12)*M*(dims.a^2 + dims.c^2);
                Izzc = (1/12)*M*(dims.a^2 + dims.b^2);
                I6_com = [Ixxc Iyyc Izzc 0 0 0];
            case 'cyl_x'
                if ~isfield(dims,'r'), dims.r = default.dims_cyl.r; end
                if ~isfield(dims,'h'), dims.h = default.dims_cyl.h; end
                Ixxc = 0.5*M*dims.r^2;
                Iyyc = (1/12)*M*(3*dims.r^2 + dims.h^2);
                Izzc = Iyyc; I6_com = [Ixxc Iyyc Izzc 0 0 0];
            case 'cyl_z'
                if ~isfield(dims,'r'), dims.r = default.dims_cyl.r; end
                if ~isfield(dims,'h'), dims.h = default.dims_cyl.h; end
                Izzc = 0.5*M*dims.r^2;
                Ixxc = (1/12)*M*(3*dims.r^2 + dims.h^2);
                Iyyc = Ixxc; I6_com = [Ixxc Iyyc Izzc 0 0 0];
            case 'sphere'
                if ~isfield(dims,'r'), dims.r = default.dims_sph.r; end
                I = (2/5)*M*dims.r^2; I6_com = [I I I 0 0 0];
            otherwise
                error('Unknown params.payload.shape = %s', shape);
        end
    end
    % 平行轴定理：把 COM 处惯量移到 tool 原点
    tool.Mass         = M;
    tool.CenterOfMass = rcom;
    tool.Inertia      = shiftInertiaCOMtoOrigin(I6_com, M, rcom);
end

% ===== 兜底体检（不改口径，只做最小值夹紧/非数修复） =====
robot = local_sanitize_inertial(robot);

% ===== kin 输出（保持既有口径）=====
kin = struct('L', L, 'base_z', params.base_z);

end

% ====================== 子函数：惯量与工具 ======================
function I6_O = shiftInertiaCOMtoOrigin(I6_C, m, r)
rx=r(1); ry=r(2); rz=r(3);
Ixx=I6_C(1); Iyy=I6_C(2); Izz=I6_C(3);
Iyz=I6_C(4); Ixz=I6_C(5); Ixy=I6_C(6);
IC = [ Ixx, Ixy, Ixz; Ixy, Iyy, Iyz; Ixz, Iyz, Izz ];
S  = (rx*rx + ry*ry + rz*rz)*eye(3) - [rx;ry;rz]*[rx ry rz];
IO = IC + m*S;
I6_O = [ IO(1,1), IO(2,2), IO(3,3), IO(2,3), IO(1,3), IO(1,2) ];
end

function [Ixx, Iyy, Izz] = rodInertiaCOM_X(m, r, L)
Ixx = 0.5*m*r^2;
Iyy = (1/12)*m*(3*r^2 + L^2);
Izz = Iyy;
end

function [Ixx, Iyy, Izz] = cylInertiaCOM_Z(m, r, h)
Izz = 0.5*m*r^2;
Ixx = (1/12)*m*(3*r^2 + h^2);
Iyy = Ixx;
end

function robot = local_sanitize_inertial(robot)
minM = 1e-3; minI = 1e-5;
for i = 1:numel(robot.Bodies)
    b = robot.Bodies{i};
    if strcmpi(b.Joint.Type,'fixed'), continue; end
    if ~isfinite(b.Mass) || b.Mass<=0, b.Mass = minM; end
    I = b.Inertia;
    if numel(I)~=6 || any(~isfinite(I)), I = [minI minI minI 0 0 0];
    else, I(1:3) = max(I(1:3), minI); for k = 4:6, if ~isfinite(I(k)), I(k)=0; end, end
    end
    b.Inertia = I;
    if any(~isfinite(b.CenterOfMass)), b.CenterOfMass = [0 0 0]; end
end
end

% --- 简易 SE(3) 变换构造 ---
function T = troty(a)
c=cos(a); s=sin(a); T=eye(4); T(1:3,1:3)=[ c 0 s; 0 1 0; -s 0 c ];
end
function T = trotz(a)
c=cos(a); s=sin(a); T=eye(4); T(1:3,1:3)=[ c -s 0; s c 0; 0 0 1 ];
end
