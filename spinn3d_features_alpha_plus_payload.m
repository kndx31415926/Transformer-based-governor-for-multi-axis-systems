function x = spinn3d_features_alpha_plus_payload(q,dq,ddq, robot, payload, Pmax, caps)
% x = [ features_from_spinn3d_features_alpha_plus , payload_vec(10) ]
% payload 可为 struct(mass/com/inertia) 或 1×10 数值向量

q=row(q); dq=row(dq); ddq=row(ddq);

% 基础特征
if exist('spinn3d_features_alpha_plus','file')==2
    try
        x0 = spinn3d_features_alpha_plus(q,dq,ddq, robot, [], Pmax, caps);
    catch
        x0 = feat_fallback(q,dq,ddq,Pmax,robot,caps);
    end
else
    x0 = feat_fallback(q,dq,ddq,Pmax,robot,caps);
end

% payload → 10 维
pv = zeros(1,10);
if nargin>=5 && ~isempty(payload)
    if isstruct(payload)
        if isfield(payload,'mass')    && ~isempty(payload.mass),     pv(1)=payload.mass; end
        if isfield(payload,'com')     && numel(payload.com)>=3,      pv(2:4)=row(payload.com(1:3)); end
        if isfield(payload,'inertia') && numel(payload.inertia)>=6,  pv(5:10)=row(payload.inertia(1:6)); end
    elseif isnumeric(payload)
        v = row(double(payload)); if numel(v) < 10, v=[v, zeros(1,10-numel(v))]; end
        pv = v(1:10);
    end
end

x = double([x0, pv]);
end

%% ---- 兜底特征，与数据生成兜底一致 ----
function x = feat_fallback(q0,dq0,ddq0,Pmax,robot,caps)
    n=numel(q0);
    tau_ff0 = row(inverseDynamics(robot,q0,dq0,ddq0));
    p_axis0 = max(tau_ff0.*dq0,0);
    Pff0=sum(p_axis0);
    gq0 = row(inverseDynamics(robot,q0,zeros(1,n),zeros(1,n)));
    x = double([q0, dq0, ddq0, tau_ff0, p_axis0, Pff0, gq0, Pmax]);
end
function r=row(v), r=v(:)'; end
