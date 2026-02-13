function bus = get_bus_from_cfg(cfg, P_total_max_fallback)
% 从 cfg.BUS 组装 BUS 口径结构体；没有则给默认。
%
% 统一字段（建议使用）：
%   P_total_max  —— 总线“取电”功率上限 [W]
%   eta_share    —— 再生共享比例/效率（0~1）
%   P_brk_peak   —— 制动电阻瞬时可消耗功率上限 [W]（可为 Inf）
%
% 兼容字段（历史别名）：
%   P_dump_max   —— 等价于 P_brk_peak
%   eta_mot / eta_regen —— 不在当前 Pgrid/Pdump 口径里使用，但保留透传

    if nargin < 2 || isempty(P_total_max_fallback)
        P_total_max_fallback = getfield_def(cfg,'P_TOTAL_MAX', inf);
    end

    % 先给默认：eta_share=1.0, P_brk_peak=Inf
    bus = spinn3d_bus_defaults(struct('P_total_max', double(P_total_max_fallback)));

    if isfield(cfg,'BUS') && isstruct(cfg.BUS)
        B = cfg.BUS;

        % --- 上限 ---
        if isfield(B,'P_total_max') && ~isempty(B.P_total_max), bus.P_total_max = double(B.P_total_max); end

        % --- share（统一字段） ---
        if isfield(B,'eta_share') && ~isempty(B.eta_share), bus.eta_share = double(B.eta_share); end

        % --- 制动电阻（统一字段 + 兼容别名） ---
        if isfield(B,'P_brk_peak') && ~isempty(B.P_brk_peak)
            bus.P_brk_peak = double(B.P_brk_peak);
        elseif isfield(B,'P_dump_max') && ~isempty(B.P_dump_max)
            bus.P_brk_peak = double(B.P_dump_max);   % 兼容旧名
        end
        % 同步别名，方便旧代码读取
        bus.P_dump_max = bus.P_brk_peak;

        % --- 其它字段：保留透传（当前 clamp/bus_power 不使用） ---
        if isfield(B,'eta_mot')     && ~isempty(B.eta_mot),     bus.eta_mot     = double(B.eta_mot); end
        if isfield(B,'eta_regen')   && ~isempty(B.eta_regen),   bus.eta_regen   = double(B.eta_regen); end
    end
end

% ===== local helper =====
function v = getfield_def(S,fn,def), if isstruct(S)&&isfield(S,fn)&&~isempty(S.(fn)), v=S.(fn); else, v=def; end, end
