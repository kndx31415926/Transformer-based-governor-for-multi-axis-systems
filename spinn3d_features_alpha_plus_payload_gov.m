function x = spinn3d_features_alpha_plus_payload_gov(q,dq,ddq, robot, payload10_or_struct, Pmax, caps, a_prev, BETA, A_DOT_UP, A_DOT_DN, Ts, s_norm)
% α++(含payload) + governor 状态参数（尾 6 维）：[a_prev, BETA, A_UP, A_DN, Ts, s_norm]

    if nargin < 7, error('至少需要前 7 个参数'); end
    if nargin < 8 || isempty(a_prev),    a_prev   = 0;     end
    if nargin < 9 || isempty(BETA),      BETA     = 0.85;  end
    if nargin < 10 || isempty(A_DOT_UP), A_DOT_UP = 2.5;   end
    if nargin < 11 || isempty(A_DOT_DN), A_DOT_DN = 5.0;   end
    if nargin < 12 || isempty(Ts),       Ts       = 0.002; end
    if nargin < 13 || isempty(s_norm),   s_norm   = NaN;   end

    % —— 统一 payload 口径：支持 1×10 数值或 struct
    if isnumeric(payload10_or_struct)
        v = double(payload10_or_struct(:)).'; v = [v, zeros(1,10-numel(v))];
        pay = struct('mass',v(1), 'com',v(2:4), 'inertia',v(5:10));
    elseif isstruct(payload10_or_struct)
        pay = payload10_or_struct;
    else
        pay = struct('mass',0, 'com',[0 0 0], 'inertia',zeros(1,6));
    end

    % —— 基础特征 + governor 尾巴 6 维
    x_base = spinn3d_features_alpha_plus_payload(q,dq,ddq, robot, pay, Pmax, caps);
    x = double([x_base, a_prev, BETA, A_DOT_UP, A_DOT_DN, Ts, s_norm]);
end
