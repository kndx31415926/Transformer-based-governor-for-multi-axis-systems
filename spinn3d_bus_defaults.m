function bus = spinn3d_bus_defaults(bus)
%SPINN3D_BUS_DEFAULTS  Default params for common DC bus + braking resistor model.
% bus.eta_share  : fraction of instantaneous regen that can be reused by other axes (0~1, default 1.0)
% bus.P_brk_peak : peak power that braking chopper+resistor can dissipate instantaneously [W] (default Inf)
    if nargin < 1 || isempty(bus), bus = struct(); end
    if ~isfield(bus,'eta_share') || isempty(bus.eta_share),  bus.eta_share  = 1.0; end
    if ~isfield(bus,'P_brk_peak') || isempty(bus.P_brk_peak), bus.P_brk_peak = Inf; end
end
