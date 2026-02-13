function a_star = alpha_feasible(q0, dq0, ddq0, a_init, robot, caps, Pmax, amin, itmax, fric) %#ok<INUSD>
% ALPHA_FEASIBLE (BUS-aware overwrite)
% - 与旧签名完全一致，但“功率口径”改为：共直流母线 + 制动电阻
%   * 逐轴正功：P_axis_j = max(tau_j * dq_j, 0)
%   * 总取电功率（母线从电源吸收）：P_grid = max(sum(P^+) - eta_share*sum(P^-), 0)  ≤  Pmax
%   * 制动电阻瞬时功率：      P_dump = max(sum(P^-) - min(eta_share*sum(P^-), sum(P^+)), 0) ≤ caps.P_brk_peak
% - 其余约束不变：|tau|/|a·dq0|/|a^2·ddq0|/逐轴正功 ≤ caps.*
% - 兼容：若 caps.eta_share / caps.P_brk_peak 缺省，则分别取 1.0 / Inf（退化回原 P^+ 口径）。
%
% NOTE: a_init 仅为保留入参，避免改动调用端；本严格版不使用它来收缩上界。

    % ==== 输入检查（严格） ====
    q0  = row(q0);  dq0  = row(dq0);  ddq0 = row(ddq0);
    nJ  = numel(q0);
    assert(isstruct(caps) && all(isfield(caps,{'tau_max','qd_max','qdd_max','P_axis_max'})), ...
        'alpha_feasible: caps 必须含 tau_max/qd_max/qdd_max/P_axis_max');
    assert(isstruct(fric) && all(isfield(fric,{'B','Fc','vel_eps'})), ...
        'alpha_feasible: fric 必须含 B/Fc/vel_eps');

    tau_lim = ensure_row_len(caps.tau_max,    nJ);
    qd_lim  = ensure_row_len(caps.qd_max,     nJ);
    qdd_lim = ensure_row_len(caps.qdd_max,    nJ);
    Pax_lim = ensure_row_len(caps.P_axis_max, nJ);

    if ~exist('Pmax','var') || isempty(Pmax), Pmax = inf; end
    assert(isscalar(Pmax) && (isfinite(Pmax) || isinf(Pmax)), 'alpha_feasible: Pmax 必须为标量');

    % BUS 口径参数（可缺省）
    eta_share  = getfield_def(caps,'eta_share',1.0);          % 多轴回灌共享效率/比例
    P_brk_peak = getfield_def(caps,'P_brk_peak',inf);         % 制动电阻瞬时功率上限
    assert(isfinite(eta_share) && eta_share>=0 && eta_share<=1.0, 'caps.eta_share ∈ [0,1]');

    % ==== 固定括区：lo = max(0,amin), hi = 1.0 ====
    if ~exist('amin','var') || isempty(amin), amin = 0; end
    if ~exist('itmax','var') || isempty(itmax), itmax = 20; end
    lo = max(0, amin);
    hi = 1.0;

    % ==== 二分 ====
    for it = 1:max(1,itmax)
        a  = 0.5*(lo + hi);
        dq = a * dq0;
        dd = (a*a) * ddq0;

        % 动力学 + 摩擦
        tau = row(inverseDynamics(robot, q0, dq, dd));
        tau = tau + row(fric.B).*dq + row(fric.Fc).*tanh(dq./max(fric.vel_eps,1e-9));

        % 逐轴正功与 BUS 功率
        [Pax, Pgrid, Pdump] = bus_powers(tau, dq, eta_share);

        ok =  all(abs(tau) <= tau_lim + 1e-9) && ...
              all(abs(dq)  <= qd_lim  + 1e-9) && ...
              all(abs(dd)  <= qdd_lim + 1e-9) && ...
              all(Pax      <= Pax_lim + 1e-9) && ...
              (Pgrid <= Pmax + 1e-9) && ...
              (Pdump <= P_brk_peak + 1e-9);

        if ok, lo = a; else, hi = a; end
        if (hi - lo) < 1e-6, break; end
    end

    a_star = lo;
end

% ===== helpers =====
function v = row(v), v = v(:)'; end
function r = ensure_row_len(v, n)
    r = v(:)';  assert(numel(r)==n || isscalar(r), '尺寸不符');
    if isscalar(r), r = repmat(r,1,n); end
end
function x = getfield_def(S,fn,def), if isstruct(S)&&isfield(S,fn)&&~isempty(S.(fn)), x=S.(fn); else, x=def; end, end
function [Pax, Pgrid, Pdump] = bus_powers(tau, dq, eta_share)
    Pj = tau .* dq;
    Ppos = sum(max(Pj,0));
    Pneg = sum(max(-Pj,0));
    Pax  = max(Pj,0);
    Pgrid = max(Ppos - eta_share*Pneg, 0);              % 从电网/电源吸收
    Pdump = max(Pneg - min(eta_share*Pneg, Ppos), 0);    % 制动电阻消耗
end
