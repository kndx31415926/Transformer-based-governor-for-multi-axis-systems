function [Ppos, Pneg, Pgrid, Pdump, Paxis] = spinn3d_bus_power(tau, dq, bus)

    tau   = row(tau); 
    dq    = row(dq);
    Paxis = tau .* dq;

    % 逐轴功率
    pw = Paxis;

    % 正功率 & 再生功率（正数）
    Ppos = sum(max(pw,0));
    Pneg = sum(max(-pw,0));

    % BUS 参数
    eta_share = getfield_def(bus,'eta_share',1.0);
    Pcap      = getfield_def(bus,'P_total_max',inf);

    % 利用 share 公式（与 nonn 一致）
    share = min(eta_share * Pneg, Ppos);

    % 电网吸收的功率
    Pgrid = max(Ppos - eta_share * Pneg, 0);

    % 进入 dump 电阻的功率
    Pdump = max(Pneg - share, 0);

end


% ===== local helpers =====
function y = row(x), y = x(:)'; end
function v = getfield_def(S,fn,def), if isstruct(S)&&isfield(S,fn)&&~isempty(S.(fn)), v=S.(fn); else, v=def; end, end
