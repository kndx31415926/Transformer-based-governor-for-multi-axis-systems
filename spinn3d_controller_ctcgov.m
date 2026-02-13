function ctl = spinn3d_controller_ctcgov(robot, traj, gains, limits, opts)
% CTC + Governor（严格版；母线功率口径）
% - NN-α：dlnetwork.forward + governor 特征；标准化前按 keepBaseIdx 压基础特征；
% - 尾巴 6 维固定顺序：[a_prev,BETA,A_DOT_UP,A_DOT_DN,Ts,s_norm]（不压列）。

try, robot.DataFormat='row'; end
nJ = local_num_joints(robot);

% ===== 基本参数（无默认）=====
assert(isfield(opts,'Ts') && opts.Ts>0, 'opts.Ts 必须提供且 >0');
Ts  = double(opts.Ts);
Kp  = vec_row(reqf(gains,'Kp'), nJ);
Kd  = vec_row(reqf(gains,'Kd'), nJ);

% ===== NN α / feasible α =====
assert(isfield(opts,'nnAlpha') && isstruct(opts.nnAlpha), '缺少 opts.nnAlpha');
nn = opts.nnAlpha;
assert(isfield(nn,'enable') && nn.enable==true, 'nnAlpha 未启用');

% ★ 默认模式：不写就当 nn
if ~isfield(nn,'mode') || isempty(nn.mode)
    nn.mode = 'nn';
end
assert(ismember(nn.mode, {'nn','feasible'}), 'nnAlpha.mode 必须为 ''nn'' 或 ''feasible''');

if strcmp(nn.mode,'nn')
    % —— 原始严格 NN 检查，保持不动 ——
    assert(isfield(nn,'model') && isfield(nn.model,'net') && isfield(nn.model,'preproc'), 'nnAlpha.model 不完整');
    assert(isa(nn.model.net,'dlnetwork'), 'nn.model.net 必须为 dlnetwork');
    assert(isfield(nn,'featureFcn') && isa(nn.featureFcn,'function_handle'), '必须提供 governor 版特征函数');
    assert(isfield(nn,'payload') && isstruct(nn.payload), 'nnAlpha.payload 必须为 struct(mass/com/inertia)');
else
    % feasible 模式：只需要 α 相关参数，其余 NN 结构全不管
    if ~isfield(nn,'alpha_floor') || isempty(nn.alpha_floor)
        nn.alpha_floor = 0;
    end
    % itmax 在这里其实没用到，但保留口径
    if ~isfield(nn,'itmax') || isempty(nn.itmax)
        nn.itmax = 20;
    end
end

% α 正则（必须显式提供 A_DOT_UP / A_DOT_DN / BETA）
req_fields = {'A_DOT_UP','A_DOT_DN','BETA'};
areg = struct();
for i = 1:numel(req_fields)
    k = req_fields{i};
    assert(isfield(opts.alphaReg,k) && ~isempty(opts.alphaReg.(k)), ...
           ['缺少 opts.alphaReg.' k]);
    areg.(k) = double(opts.alphaReg.(k));
end

% 可选字段：如果你以后想用，可以传；不传就没有
if isfield(opts.alphaReg,'N_BISECT')
    areg.N_BISECT = double(opts.alphaReg.N_BISECT);
end
if isfield(opts.alphaReg,'GAMMA')
    areg.GAMMA = double(opts.alphaReg.GAMMA);
end


% 母线/摩擦/限值
bus  = spinn3d_bus_defaults(getfield_def(opts,'bus',struct()));
fric = getfield_def(opts,'fric', struct('B',0,'Fc',0,'vel_eps',1e-3));
caps = normalize_caps(limits, nJ);
Pcap = getfield_def(limits,'P_total_max', inf);

% 状态
% 状态
state = struct('a_prev',      getfield_def(nn,'alpha_floor',0), ...
               'Pgrid_lp',    0.0, ...          % 低通母线功率估计（用于 gating/诊断）
               'gate_active', false);           % gating 状态（滞回）

% ===== Slack-gating（可选，用于“预算不紧时不乱动”）=====
% 目的：避免 NN 在本来无需触发功率约束的段落引入不必要的调速/抖动（减少 outlier）。
% 触发依据：上一拍估计的低通母线功率 Pgrid_lp / Pcap（带滞回）。
gate = getfield_def(nn, 'gate', struct());
if ~isfield(gate,'enable') || isempty(gate.enable), gate.enable = false; end
if ~isfield(gate,'on_ratio') || isempty(gate.on_ratio), gate.on_ratio = 0.60; end   % r < on -> 开启 gating
if ~isfield(gate,'off_ratio')|| isempty(gate.off_ratio), gate.off_ratio= 0.80; end  % r > off -> 关闭 gating
if gate.off_ratio < gate.on_ratio, gate.off_ratio = min(1.0, gate.on_ratio + 0.05); end
if ~isfield(gate,'lp_tau_s') || isempty(gate.lp_tau_s), gate.lp_tau_s = 0.50; end   % LP 时间常数（秒）
if ~isfield(gate,'a_nom')    || isempty(gate.a_nom),    gate.a_nom = 1.0; end       % gating 时的名义 a_des
if ~isfield(gate,'skip_forward') || isempty(gate.skip_forward), gate.skip_forward = true; end
if ~isfield(gate,'fallback') || isempty(gate.fallback), gate.fallback = 'a_nom'; end
assert(ismember(gate.fallback, {'a_nom','feasible_cap'}), ...
       'nnAlpha.gate.fallback 必须为 ''a_nom'' 或 ''feasible_cap''');

% ===== 主循环（注意：8 参，包含几何进度 s） =====
    function [tau_cmd, info] = step(t, q_meas, dq_meas, q_ref, dq_ref, ddq_ref, robot_in, s) %#ok<INUSD>

        % ★ 先初始化所有在 if/else 中用到的变量（嵌套函数要求）
        a0      = 0;
        a       = 0;
        a_slew  = 0;
        a_filt  = 0;
        adot    = 0;
        guard   = struct();
        mode_str = '';

gate_hit   = false;
gate_ratio = NaN;
a_star_gate = NaN;
Pgrid_est  = NaN;

        q_meas = row(q_meas); dq_meas = row(dq_meas);
        q_ref  = row(q_ref);  dq_ref  = row(dq_ref);  ddq_ref = row(ddq_ref);

        % ===================== 1) 计算 α =====================
        if strcmp(nn.mode,'feasible')
            % ===== feasible 模式：alpha_feasible 基线 =====
            % 1) 计算参考(q_ref,dq_ref,ddq_ref)在约束下的可行上界 a_star（含 τ/qd/qdd/逐轴正功/BUS + 摩擦）
            % 2) 反解 governor 得到 a0(=a_des)，使 governor 后尽量落在 a_star（与训练 teacher 口径一致）
            % 3) 最终再做一次 hard cap：a <= a_star（避免滤波/速率限制导致短暂超界）
            caps_feas = caps;
            caps_feas.eta_share  = getfield_def(bus,'eta_share', getfield_def(caps,'eta_share',1.0));
            caps_feas.P_brk_peak = getfield_def(bus,'P_brk_peak', ...
                                      getfield_def(bus,'P_dump_max', getfield_def(caps,'P_brk_peak',inf)));

            a_star = alpha_feasible(q_ref, dq_ref, ddq_ref, 1.0, robot, caps_feas, Pcap, ...
                                    getfield_def(nn,'alpha_floor',0), getfield_def(nn,'itmax',20), fric);
            a_star = clamp(double(a_star), getfield_def(nn,'alpha_floor',0), 1.0);

            a0 = alpha_des_teacher(a_star, state.a_prev, areg.BETA, areg.A_DOT_UP, areg.A_DOT_DN, Ts);

            [a, a_slew, a_filt, adot, guard] = alpha_guard_backtrack( ...
                a0, state.a_prev, Ts, q_ref, dq_ref, ddq_ref, robot, caps_feas, Pcap, fric, areg, bus);

            a = min(a, a_star);
            adot = (a - state.a_prev)/max(Ts,1e-12);
            guard.a_star = a_star;

            mode_str = 'feasible';
        else
            % ===== NN 模式：保持原版完整逻辑 =====
% --- 0) Slack-gating：预算很松时，避免 NN 产生不必要的调速/抖动（可选）---
if gate.enable && isfinite(Pcap) && Pcap>0
    gate_ratio = state.Pgrid_lp / max(Pcap, 1e-12);
    if state.gate_active
        if gate_ratio > gate.off_ratio
            state.gate_active = false;
        end
    else
        if gate_ratio < gate.on_ratio
            state.gate_active = true;
        end
    end
    gate_hit = state.gate_active;
else
    state.gate_active = false;
end

% gate 命中：可选择跳过 NN 前向（省算力/降低 outlier），直接走名义 a 或可行 cap
if gate_hit && gate.skip_forward
    if strcmp(gate.fallback,'feasible_cap')
        caps_feas = caps;
        caps_feas.eta_share  = getfield_def(bus,'eta_share', getfield_def(caps,'eta_share',1.0));
        caps_feas.P_brk_peak = getfield_def(bus,'P_brk_peak', ...
                                  getfield_def(bus,'P_dump_max', getfield_def(caps,'P_brk_peak',inf)));

        a_star_gate = alpha_feasible(q_ref, dq_ref, ddq_ref, 1.0, robot, caps_feas, Pcap, ...
                                getfield_def(nn,'alpha_floor',0), getfield_def(nn,'itmax',20), fric);
        a_star_gate = clamp(double(a_star_gate), getfield_def(nn,'alpha_floor',0), 1.0);

        a0 = alpha_des_teacher(a_star_gate, state.a_prev, areg.BETA, areg.A_DOT_UP, areg.A_DOT_DN, Ts);
        mode_str = 'nn_gate_feasible';
    else
        a0 = clamp(double(gate.a_nom), getfield_def(nn,'alpha_floor',0), 1.0);
        mode_str = 'nn_gate_nom';
    end
else

            % --- 1) 特征：基础 + payload10 + 尾巴6（严格顺序）---
            % 支持两种 NN：
            %   (a) 单帧 MLP（历史版本）：preproc.K 不存在或 K==1
            %   (b) look-ahead Transformer：preproc.K>1，输入为 K 个 token 的序列

            assert(isfield(nn.model.preproc,'keepBaseIdx') && ~isempty(nn.model.preproc.keepBaseIdx), ...
                   '模型缺 preproc.keepBaseIdx（训练端未保存列掩码）');
            kb    = logical(nn.model.preproc.keepBaseIdx(:).');

            Klook = 1;
            if isfield(nn.model.preproc,'K') && ~isempty(nn.model.preproc.K)
                Klook = double(nn.model.preproc.K);
            end
            if ~isfinite(Klook) || Klook<1, Klook = 1; end
            Klook = round(Klook);

            if Klook > 1
                % ===== (b) look-ahead 序列输入 =====
                ds = getfield_def(nn.model.preproc,'ds', Ts);  % s 轴上的 look-ahead 步长（默认 Ts）
                if ~isfinite(ds) || ds<=0, ds = Ts; end

                % token#1 用本步参考，后续 token 用 traj.eval(s_i)
                s_norm0 = min(1.0, s / max(traj.Tf, 1e-9));
                x1 = nn.featureFcn(q_ref, dq_ref, ddq_ref, robot, nn.payload, Pcap, caps, ...
                                   state.a_prev, areg.BETA, areg.A_DOT_UP, areg.A_DOT_DN, Ts, s_norm0);
                x1 = row(double(x1));

                Dtot  = numel(x1);
                Dtail = 6;
                Dbase = Dtot - Dtail;
                assert(numel(kb)==Dbase, 'keepBaseIdx 长度(%d)≠基础特征维(%d)', numel(kb), Dbase);

                Xfull = zeros(Klook, Dtot);
                Xfull(1,:) = x1;
                for ti = 2:Klook
                    s_i = min(traj.Tf, s + (ti-1)*ds);
                    [qi, dqgi, ddqgi] = traj.eval(s_i);
                    s_norm_i = min(1.0, s_i / max(traj.Tf, 1e-9));
                    xi = nn.featureFcn(row(qi), row(dqgi), row(ddqgi), robot, nn.payload, Pcap, caps, ...
                                       state.a_prev, areg.BETA, areg.A_DOT_UP, areg.A_DOT_DN, Ts, s_norm_i);
                    Xfull(ti,:) = row(double(xi));
                end

                % 压列：只压基础特征，尾巴6保留
                Xbase = Xfull(:,1:Dbase);
                Xtail = Xfull(:,Dbase+1:end);
                Xc    = [Xbase(:,kb), Xtail];
                Din   = size(Xc,2);

                % z-score（逐 token）
                mu  = row(double(nn.model.preproc.mu));
                sig = row(double(nn.model.preproc.sig));
                assert(numel(mu)==Din && numel(sig)==Din, ...
                       '特征维(%d)与 mu/sig 维(%d)不一致', Din, numel(mu));
                Z = (Xc - mu) ./ max(sig, 1e-12);   % K×Din

                % 输入格式：C×B×T = Din×1×Klook
                dlX = dlarray(single(reshape(Z.', [Din, 1, Klook])), 'CBT');
                a0  = clamp(double(extractdata(forward(nn.model.net, dlX))), getfield_def(nn,'alpha_floor',0), 1.0);
            else
                % ===== (a) 单帧输入（原版） =====
                s_norm = min(1.0, s / max(traj.Tf, 1e-9));   % ★ 用几何进度，和训练对齐
                x_full = nn.featureFcn(q_ref, dq_ref, ddq_ref, robot, nn.payload, Pcap, caps, ...
                                       state.a_prev, areg.BETA, areg.A_DOT_UP, areg.A_DOT_DN, Ts, s_norm);
                x_full = row(double(x_full));

                Dtot  = numel(x_full);
                Dtail = 6;
                Dbase = Dtot - Dtail;
                assert(numel(kb)==Dbase, 'keepBaseIdx 长度(%d)≠基础特征维(%d)', numel(kb), Dbase);

                x_base = x_full(1:Dbase);
                x_tail = x_full(Dbase+1:end);
                x      = [x_base(kb), x_tail];

                mu  = row(double(nn.model.preproc.mu));
                sig = row(double(nn.model.preproc.sig));
                assert(numel(mu)==numel(x) && numel(sig)==numel(x), ...
                       '特征维(%d)与 mu/sig 维(%d)不一致', numel(x), numel(mu));
                z   = (x - mu) ./ max(sig, 1e-12);
                dlX = dlarray(single(z'), 'CB');
                a0  = clamp(double(extractdata(forward(nn.model.net, dlX))), getfield_def(nn,'alpha_floor',0), 1.0);
            end

end % gate.skip_forward

            % --- 4) Governor 护栏（BUS 口径）---
            [a, a_slew, a_filt, adot, guard] = alpha_guard_backtrack( ...
                a0, state.a_prev, Ts, q_ref, dq_ref, ddq_ref, robot, caps, Pcap, fric, areg, bus);
% 若 gate 使用 feasible_cap，则再做一次 hard cap：a <= a_star_gate（避免滤波/速率导致短暂超界）
if gate_hit && strcmp(gate.fallback,'feasible_cap') && isfinite(a_star_gate)
    a = min(a, a_star_gate);
    adot = (a - state.a_prev)/max(Ts,1e-12);
    guard.a_star_gate = a_star_gate;
end

% 记录 gate 诊断信息
guard.gate_active = gate_hit;
guard.gate_ratio  = gate_ratio;

        end

        % ===================== 2) CTC（含 α̇） =====================
        dq_cmd  = a    * dq_ref;
        ddq_cmd = a*a  * ddq_ref + adot * dq_ref;

        e  = q_ref  - q_meas;
        ed = dq_cmd - dq_meas;
        v  = ddq_cmd + Kd.*ed + Kp.*e;

        Mq = massMatrix(robot, q_meas);
        c  = velocityProduct(robot, q_meas, dq_meas);
        g  = gravityTorque(robot, q_meas);
        tau_cmd = (Mq * v(:) + c(:) + g(:)).';

        if any(fric.B) || any(fric.Fc)
            tau_cmd = tau_cmd + fric.B.*dq_meas + fric.Fc.*tanh(dq_meas./max(fric.vel_eps,1e-9));
        end

        % --- 5) 估计/低通母线功率（仅用于 gating/诊断，不参与控制律） ---
        Pgrid_est = bus_power_grid(tau_cmd, dq_meas, bus);
        if gate.enable && isfinite(Pcap) && Pcap>0
            tau_lp = max(gate.lp_tau_s, 1e-6);
            betaP  = exp(-Ts/tau_lp);
            state.Pgrid_lp = betaP*state.Pgrid_lp + (1-betaP)*Pgrid_est;
        else
            % gate 未启用时也更新一下（便于日志观察）
            state.Pgrid_lp = Pgrid_est;
        end

        state.a_prev = a;
        info = struct('a_max',a,'a0',a0,'guard',guard, ...
              'a_slew',a_slew,'a_filt',a_filt,'adot',adot,'mode',mode_str, ...
              'Pgrid_est',Pgrid_est,'Pgrid_lp',state.Pgrid_lp, ...
              'gate_active',gate_hit,'gate_ratio',gate_ratio);
    end

ctl = struct('step',@step, 'Kp',Kp,'Kd',Kd,'limits',caps,'nnAlpha',nn,'Ts',Ts,'alphaReg',areg,'bus',bus);
end

% ================= 辅助函数 =================
function n = local_num_joints(robot)
    n = numel(homeConfiguration(robot));
end
function caps = normalize_caps(lim, nJ)
    caps = struct();
    fn = {'tau_max','qd_max','qdd_max','P_axis_max'};
    for i=1:numel(fn)
        if isfield(lim,fn{i}) && ~isempty(lim.(fn{i}))
            v = lim.(fn{i})(:).'; if isscalar(v), v = repmat(v,1,nJ); end
            caps.(fn{i}) = double(v(1:nJ));
        else
            caps.(fn{i}) = inf(1,nJ);
        end
    end
    if isfield(lim,'countRegen'), caps.countRegen = logical(lim.countRegen); end
end
function v = vec_row(x, n), x=double(x); if isscalar(x), v=repmat(x,1,n); else, v=x(:).'; end, v=v(1:n); end
function v = reqf(S, k), assert(isfield(S,k) && ~isempty(S.(k)), ['缺少字段: ',k]); v = S.(k); end
function x = getfield_def(S, k, d), if isstruct(S)&&isfield(S,k)&&~isempty(S.(k)), x=S.(k); else, x=d; end, end
function r = row(v), r = v(:)'; end
function y = clamp(x, lo, hi), y = min(hi, max(lo, x)); end
function Pgrid = bus_power_grid(tau, dq, bus)
    % 机械功率 -> DC-bus 口径（与 spinn3d_bus_power 对齐的简化实现）
    tau = row(tau); dq = row(dq);
    Paxis = tau .* dq;
    Ppos  = sum(max(Paxis,0));
    Pneg  = sum(max(-Paxis,0));
    eta   = getfield_def(bus,'eta_share',1.0);
    Pgrid = max(Ppos - eta*Pneg, 0);
end

function m = min_safe(v), m = min(v(:)); if isempty(m), m = 1.0; end, end

function [a, a_slew, a_filt, adot, G] = alpha_guard_backtrack(a0, a_prev, Ts, q, dq_ref, ddq_ref, robot, caps, Pcap, fric, areg, bus) %#ok<INUSD>
    a0     = clamp(a0, 0, 1);
    a_slew = min(a0, a_prev + areg.A_DOT_UP*Ts);
    a_slew = max(a_slew, a_prev - areg.A_DOT_DN*Ts);
    a_filt = areg.BETA*a_prev + (1-areg.BETA)*a_slew;

    sf_qd  = min_safe(caps.qd_max  ./ max(abs(dq_ref),  1e-12));
    sf_qdd = min_safe(sqrt(caps.qdd_max ./ max(abs(ddq_ref),1e-12)));
    a_kin  = min([1.0, sf_qd, sf_qdd]);

    a    = min(a_filt, a_kin);
    adot = (a - a_prev)/max(Ts,1e-12);
    G = struct('a0',a0,'a_slew',a_slew,'a_filt',a_filt,'a_kin',a_kin);
end
