function tau_out = clamp_power_axis_total_bus(tau, dq, P_axis_max, bus)
% 命令级扭矩护栏（BUS 口径，含再生 share）：
%  1) 逐轴正功限幅：P_i = max(tau_i*dq_i, 0) <= P_axis_max(i)
%  2) DC BUS 取电功率限幅：P_grid = max(Ppos - eta_share*Pneg, 0) <= bus.P_total_max
%  3) (可选) 制动电阻瞬时功率限幅：P_dump <= bus.P_brk_peak（或 bus.P_dump_max）
%
% 口径与 spinn3d_bus_power / alpha_feasible 保持一致：
%   Ppos  = sum(max(tau.*dq, 0));
%   Pneg  = sum(max(-tau.*dq, 0));
%   Pgrid = max(Ppos - eta_share*Pneg, 0);
%   Pdump = max(Pneg - min(eta_share*Pneg, Ppos), 0);

    tau = row(tau);
    dq  = row(dq);
    n   = numel(tau);

    if nargin < 3 || isempty(P_axis_max), P_axis_max = inf(1,n); end
    P_axis_max = row(P_axis_max);

    if nargin < 4 || isempty(bus), bus = struct(); end
    try
        bus = spinn3d_bus_defaults(bus);
    catch
        % 若没有该函数，也不影响：直接走 getfield_def 默认
    end

    % 兼容字段名：P_dump_max <-> P_brk_peak
    if ~isfield(bus,'P_brk_peak') && isfield(bus,'P_dump_max'), bus.P_brk_peak = bus.P_dump_max; end
    if ~isfield(bus,'P_dump_max') && isfield(bus,'P_brk_peak'), bus.P_dump_max = bus.P_brk_peak; end

    Pcap      = getfield_def(bus, 'P_total_max', inf);
    eta_share = getfield_def(bus, 'eta_share', 1.0);
    Pdump_cap = getfield_def(bus, 'P_brk_peak', getfield_def(bus,'P_dump_max', inf));

    % ---------- 1) 逐轴正功限幅 ----------
    Pj = tau .* dq;
    for i = 1:n
        if isfinite(P_axis_max(i)) && (Pj(i) > P_axis_max(i))
            s = P_axis_max(i) / max(Pj(i), 1e-12);   % 只会在 Pj>0 时触发
            tau(i) = tau(i) * s;
        end
    end

    % ---------- 2) BUS 取电功率限幅 ----------
    tau = clamp_bus_grid_power(tau, dq, Pcap, eta_share);

    % ---------- 3) (可选) dump 功率限幅 ----------
    if isfinite(Pdump_cap)
        tau = clamp_bus_dump_power(tau, dq, Pdump_cap, eta_share);

        % dump 限幅可能会减少再生，从而抬高 Pgrid；再做一次 Pgrid 限幅
        tau = clamp_bus_grid_power(tau, dq, Pcap, eta_share);
    end

    tau_out = tau;
end

% ===== helpers =====
function tau = clamp_bus_grid_power(tau, dq, Pcap, eta_share)
    if ~isfinite(Pcap), return; end

    Pj   = tau .* dq;
    Ppos = sum(max(Pj,0));
    Pneg = sum(max(-Pj,0));
    Pgrid = max(Ppos - eta_share*Pneg, 0);

    if Pgrid > Pcap + 1e-12
        % 只缩放“正功通道”，把 Ppos 压到 Pcap + eta_share*Pneg
        Ppos_cap = max(Pcap + eta_share*Pneg, 0);
        if Ppos > 0
            s = Ppos_cap / max(Ppos, 1e-12);
            pos = (Pj > 0);
            tau(pos) = tau(pos) * s;
        end
    end
end

function tau = clamp_bus_dump_power(tau, dq, Pdump_cap, eta_share)
    Pj   = tau .* dq;
    Ppos = sum(max(Pj,0));
    Pneg = sum(max(-Pj,0));

    share = min(eta_share * Pneg, Ppos);
    Pdump = max(Pneg - share, 0);

    if Pdump > Pdump_cap + 1e-12
        neg = (Pj < 0);
        if any(neg) && Pneg > 0
            % 保守目标：让 Pneg 降到同时满足两种 regime 的上界
            % 1) 若 share=Ppos：Pdump=Pneg-Ppos <= cap  -> Pneg <= cap + Ppos
            % 2) 若 share=eta_share*Pneg：Pdump=(1-eta_share)*Pneg <= cap  -> Pneg <= cap/(1-eta_share)
            Pneg_cap1 = Pdump_cap + Ppos;
            if eta_share < 1.0
                Pneg_cap2 = Pdump_cap / max(1.0 - eta_share, 1e-12);
            else
                Pneg_cap2 = inf;
            end
            Pneg_target = min([Pneg, Pneg_cap1, Pneg_cap2]);
            s = Pneg_target / max(Pneg, 1e-12);
            tau(neg) = tau(neg) * s;
        end
    end
end

function y = row(x), y = x(:)'; end
function v = getfield_def(S,fn,def), if isstruct(S)&&isfield(S,fn)&&~isempty(S.(fn)), v=S.(fn); else, v=def; end, end
