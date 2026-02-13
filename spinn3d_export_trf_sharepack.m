function rep = spinn3d_export_trf_sharepack(master_mat, model_mat, out_dir, opts)
% 导出 Transformer(look-ahead) 评估诊断包（可上传给我分析）
%
% 输出：
%   out_dir/trf_sharepack_report.mat
%   out_dir/trf_sharepack_val_samples.csv
%   out_dir/trf_sharepack_bins.csv
%   out_dir/trf_sharepack_summary.txt
%
% 默认路径会读取 PB.paths.* 的习惯：
%   master = data_alpha/spinn3d_alpha_gov_seq_dataset_master.mat
%   model  = data_alpha/model_alpha_gov_trf.mat

    if nargin < 1 || isempty(master_mat)
        master_mat = fullfile('data_alpha','spinn3d_alpha_gov_seq_dataset_master.mat');
    end
    if nargin < 2 || isempty(model_mat)
        model_mat  = fullfile('data_alpha','model_alpha_gov_trf.mat');
    end
    if nargin < 3 || isempty(out_dir)
        out_dir = fileparts(master_mat);
        if isempty(out_dir), out_dir = '.'; end
    end
    if nargin < 4 || isempty(opts), opts = struct(); end

    Batch   = getfield_def(opts,'Batch', 1024);
    SensN   = getfield_def(opts,'SensN', 1024);   % look-ahead 敏感性评估样本数（val 中抽）
    TolOver = getfield_def(opts,'TolOver', 1e-3); % 判断 a_post 超 teacher 的容差
    Seed    = getfield_def(opts,'Seed', 0);
    ComputePhysics = getfield_def(opts,'ComputePhysics', true);

    assert(exist(master_mat,'file')==2, '找不到 master_mat: %s', master_mat);
    assert(exist(model_mat,'file')==2,  '找不到 model_mat: %s', model_mat);
    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    S = load(master_mat);
    assert(isfield(S,'Xseq') && isfield(S,'y'), 'master_mat 必须包含 Xseq/y');
    Xseq = double(S.Xseq);
    y    = double(S.y(:));

    PB    = [];
    caps0 = [];
    if isfield(S,'PB'),    PB = S.PB; end
    if isfield(S,'caps0'), caps0 = S.caps0; end

    [N,K,Dtot] = size(Xseq);

    M = load(model_mat,'model');
    assert(isfield(M,'model'), 'model_mat 内必须保存变量 model');
    model = M.model;
    assert(isfield(model,'net') && isfield(model,'preproc'), 'model 缺 net/preproc');

    % --- 维度：末尾 tail6 固定 ---
    Dtail   = 6;
    baseDim = Dtot - Dtail;
    assert(baseDim>0, 'Dtot 太小');
    keepBaseIdx = logical(model.preproc.keepBaseIdx(:));
    mu  = double(model.preproc.mu(:)).';
    sig = double(model.preproc.sig(:)).';

    % --- 复现训练时的 val split（rng(0), 90/10） ---
    rng(0);
    idx = randperm(N);
    nTr = max(1, round(0.90*N));
    Itr = idx(1:nTr);
    Iva = idx(nTr+1:end);
    Nv  = numel(Iva);

    % --- 统计标签分布（全体） ---
    y_stats = basic_stats(y);

    % --- 预测 val：yhat（a_des） ---
    y_va = y(Iva);
    yhat_va = zeros(Nv,1);
    for p = 1:ceil(Nv/Batch)
        a = (p-1)*Batch+1;
        b = min(p*Batch, Nv);
        I = Iva(a:b);

        yhat_va(a:b) = predict_yhat_batch(Xseq(I,:,:), model, keepBaseIdx, mu, sig, baseDim);
    end

    yhat_stats = basic_stats(yhat_va);

    % --- governor 后的 a_post（线上真正用） ---
    tail1 = squeeze(Xseq(Iva,1,baseDim+1:end)); % Nv×6: [a_prev,beta,up,dn,Ts,s_norm]
    a_prev = tail1(:,1);
    beta   = tail1(:,2);
    up     = tail1(:,3);
    dn     = tail1(:,4);
    Ts     = tail1(:,5);
    s_norm = tail1(:,6);

    a_post_true = governor_apply(y_va,    a_prev, beta, up, dn, Ts);
    a_post_hat  = governor_apply(yhat_va, a_prev, beta, up, dn, Ts);

    post_stats_true = basic_stats(a_post_true);
    post_stats_hat  = basic_stats(a_post_hat);

    over_mask  = (a_post_hat > a_post_true + TolOver);
    under_mask = (a_post_hat < a_post_true - TolOver);

    % --- 回归指标（val） ---
    met_des  = regression_metrics(y_va, yhat_va);
    met_post = regression_metrics(a_post_true, a_post_hat);

    % --- look-ahead 是否真的被用到：替换 token2..K 的 base 特征 ---
    rng(Seed);
    Ns = min(SensN, Nv);
    perm = randperm(Nv, Ns);
    pos_self  = perm(:);
    pos_donor = randi(Nv, Ns, 1);

    Iself  = Iva(pos_self);
    Idonor = Iva(pos_donor);

    Xmix = Xseq(Iself,:,:);
    if K >= 2
        % 只替换 base（1:baseDim），保留 tail（governor 参数 + s_norm）
        Xmix(:,2:K,1:baseDim) = Xseq(Idonor,2:K,1:baseDim);
    end

    yhat_self = yhat_va(pos_self);
    yhat_mix  = predict_yhat_batch(Xmix, model, keepBaseIdx, mu, sig, baseDim);
    delta_look = abs(yhat_mix - yhat_self);
    look_stats = basic_stats(delta_look);

    % --- 物理/功率（可选）：同 PINN 口径计算 horizon Pgrid 超限 ---
    phys = struct();
    if ComputePhysics && ~isempty(caps0) && isstruct(caps0)
        phys = physics_horizon_report(Xseq(Iva,:,:), a_post_true, a_post_hat, caps0, PB);
    end

    % --- 导出 val 样本表（CSV） ---
    nJ = infer_nJ_from_caps(caps0);
    D0 = 13*nJ + 12;
    idxPmax = 6*nJ + 6;
    idxPayM = D0 + 1;
    Pmax = squeeze(Xseq(Iva,1,idxPmax));
    payM = squeeze(Xseq(Iva,1,idxPayM));

    T = table();
    T.sample_idx = Iva(:);
    T.y_des_true = y_va(:);
    T.y_des_hat  = yhat_va(:);
    T.a_prev     = a_prev(:);
    T.a_post_true= a_post_true(:);
    T.a_post_hat = a_post_hat(:);
    T.s_norm     = s_norm(:);
    T.Pmax       = Pmax(:);
    T.payload_m  = payM(:);
    T.over_post  = double(over_mask(:));
    T.under_post = double(under_mask(:));

    if isfield(phys,'maxPgrid_hat')
        T.maxPgrid_hat = phys.maxPgrid_hat(:);
        T.maxPgrid_true= phys.maxPgrid_true(:);
        T.anyPgrid_over_hat  = double(phys.anyOver_hat(:));
        T.anyPgrid_over_true = double(phys.anyOver_true(:));
        T.maxRatio_hat = phys.maxRatio_hat(:);
        T.maxRatio_true= phys.maxRatio_true(:);
    end

    fn_samples = fullfile(out_dir,'trf_sharepack_val_samples.csv');
    writetable(T, fn_samples);

    % --- bins 表（按 Pmax 五分位） ---
    Tb = make_bins_table(Pmax, y_va, yhat_va, a_post_true, a_post_hat, phys);
    fn_bins = fullfile(out_dir,'trf_sharepack_bins.csv');
    writetable(Tb, fn_bins);

    % --- 组装 rep 并保存 mat + txt ---
    rep = struct();
    rep.meta = struct('master_mat',master_mat,'model_mat',model_mat,'out_dir',out_dir, ...
                      'N',N,'K',K,'Dtot',Dtot,'baseDim',baseDim, ...
                      'Din', sum(keepBaseIdx)+6, ...
                      'Ntrain',numel(Itr),'Nval',Nv);
    rep.y_stats = y_stats;
    rep.yhat_stats = yhat_stats;
    rep.a_post_true_stats = post_stats_true;
    rep.a_post_hat_stats  = post_stats_hat;
    rep.metrics_des  = met_des;
    rep.metrics_post = met_post;
    rep.over_rate  = mean(over_mask);
    rep.under_rate = mean(under_mask);
    rep.lookahead_delta_stats = look_stats;
    rep.physics = phys;

    fn_mat = fullfile(out_dir,'trf_sharepack_report.mat');
    save(fn_mat,'rep','-v7.3');

    fn_txt = fullfile(out_dir,'trf_sharepack_summary.txt');
    write_summary_txt(fn_txt, rep);

    % 同时在命令行打印一份关键摘要（方便你直接复制给我）
    disp_summary(rep);

    fprintf('[sharepack] saved:\n  %s\n  %s\n  %s\n  %s\n', fn_mat, fn_samples, fn_bins, fn_txt);
end

% ================= helpers =================
function yhat = predict_yhat_batch(XseqB, model, keepBaseIdx, mu, sig, baseDim)
    % XseqB: B×K×Dtot
    Xbase = XseqB(:,:,1:baseDim);
    Xtail = XseqB(:,:,baseDim+1:end);
    Xbk   = Xbase(:,:,keepBaseIdx);
    Xc    = cat(3, Xbk, Xtail);      % B×K×Din

    Din = size(Xc,3);
    mu3  = reshape(mu,  1,1,Din);
    sig3 = reshape(sig, 1,1,Din);
    Z = (Xc - mu3) ./ sig3;

    dlX = dlarray(single(permute(Z,[3 1 2])),'CBT'); % Din×B×K
    dlY = forward(model.net, dlX);                   % 1×B
    yhat = double(gather(extractdata(dlY)));
    yhat = yhat(:);                                  % B×1
end

function a_post = governor_apply(a_des, a_prev, beta, up, dn, Ts)
    lo = a_prev - dn.*Ts;
    hi = a_prev + up.*Ts;
    a_slew = min(hi, max(lo, a_des));
    a_post = beta.*a_prev + (1-beta).*a_slew;
end

function st = basic_stats(x)
    x = double(x(:));
    st = struct();
    st.n    = numel(x);
    st.mean = mean(x);
    st.std  = std(x);
    st.min  = min(x);
    st.p01  = pct(x,1);
    st.p05  = pct(x,5);
    st.p50  = pct(x,50);
    st.p95  = pct(x,95);
    st.p99  = pct(x,99);
    st.max  = max(x);
    st.frac_gt_099 = mean(x>0.99);
    st.frac_lt_020 = mean(x<0.20);
    st.frac_lt_050 = mean(x<0.50);
end

function m = regression_metrics(y, yhat)
    y = double(y(:)); yhat = double(yhat(:));
    e = yhat - y;
    m = struct();
    m.mse  = mean(e.^2);
    m.mae  = mean(abs(e));
    m.rmse = sqrt(m.mse);
    if std(y) < 1e-12
        m.r2 = NaN;
        m.corr = NaN;
    else
        ss_res = sum((y-yhat).^2);
        ss_tot = sum((y-mean(y)).^2);
        m.r2 = 1 - ss_res/max(ss_tot,1e-12);
        C = corrcoef(y,yhat);
        if numel(C)>=4, m.corr = C(1,2); else, m.corr = NaN; end
    end
    m.bias = mean(e);
end

function q = pct(x, p)
    x = sort(double(x(:)));
    n = numel(x);
    if n==0, q=NaN; return; end
    k = 1 + (n-1)*p/100;
    f = floor(k); c = ceil(k);
    if f<1, f=1; end
    if c<1, c=1; end
    if f>n, f=n; end
    if c>n, c=n; end
    if f==c
        q = x(f);
    else
        q = x(f)*(c-k) + x(c)*(k-f);
    end
end

function v = getfield_def(S, fn, def)
    if isstruct(S) && isfield(S,fn) && ~isempty(S.(fn))
        v = S.(fn);
    else
        v = def;
    end
end

function nJ = infer_nJ_from_caps(caps)
    nJ = 4;
    if isstruct(caps)
        if isfield(caps,'tau_max') && ~isempty(caps.tau_max), nJ = numel(caps.tau_max); return; end
        if isfield(caps,'qd_max')  && ~isempty(caps.qd_max),  nJ = numel(caps.qd_max);  return; end
    end
end

function phys = physics_horizon_report(XseqV, a_true, a_hat, caps, PB)
    % 用训练 PINN 同口径的近似动力学算 horizon 上的 Pgrid 超限情况（val）
    [Nv,K,Dtot] = size(XseqV);
    nJ = infer_nJ_from_caps(caps);

    idx_dq    = (nJ+1):(2*nJ);
    idx_ddq   = (2*nJ+1):(3*nJ);
    idx_tau1  = (3*nJ+1):(4*nJ);
    idx_g     = (5*nJ+2):(6*nJ+1);
    idx_Pmax  = (6*nJ+6);

    dqg  = XseqV(:,:,idx_dq);    % Nv×K×nJ
    ddqg = XseqV(:,:,idx_ddq);
    tau1 = XseqV(:,:,idx_tau1);
    gq   = XseqV(:,:,idx_g);
    Pmax = squeeze(XseqV(:,1,idx_Pmax)); % Nv×1

    % fric
    fricB = 0; fricFc = 0; vel_eps = 1e-3;
    if isstruct(PB) && isfield(PB,'fric')
        if isfield(PB.fric,'B'), fricB = PB.fric.B; end
        if isfield(PB.fric,'Fc'), fricFc = PB.fric.Fc; end
        if isfield(PB.fric,'vel_eps'), vel_eps = PB.fric.vel_eps; end
    end
    fricB  = reshape(double(fricB), 1,1,[]);
    fricFc = reshape(double(fricFc),1,1,[]);
    if numel(fricB)==1,  fricB  = repmat(fricB,  1,1,nJ); end
    if numel(fricFc)==1, fricFc = repmat(fricFc, 1,1,nJ); end

    eta_share = 1.0;
    if isfield(caps,'eta_share') && ~isempty(caps.eta_share), eta_share = double(caps.eta_share); end

    Pax_lim = inf(1,nJ);
    if isfield(caps,'P_axis_max') && ~isempty(caps.P_axis_max), Pax_lim = double(caps.P_axis_max(:)).'; end
    if numel(Pax_lim)==1, Pax_lim = repmat(Pax_lim,1,nJ); end
    Pax_lim3 = reshape(Pax_lim, 1,1,nJ);

    % compute for both alpha sets
    [maxPgrid_true, maxRatio_true, anyOver_true] = pgrid_stats(dqg,ddqg,tau1,gq,Pmax,a_true,fricB,fricFc,vel_eps,eta_share);
    [maxPgrid_hat,  maxRatio_hat,  anyOver_hat ] = pgrid_stats(dqg,ddqg,tau1,gq,Pmax,a_hat, fricB,fricFc,vel_eps,eta_share);

    phys = struct();
    phys.maxPgrid_true = maxPgrid_true;
    phys.maxPgrid_hat  = maxPgrid_hat;
    phys.maxRatio_true = maxRatio_true;
    phys.maxRatio_hat  = maxRatio_hat;
    phys.anyOver_true  = anyOver_true;
    phys.anyOver_hat   = anyOver_hat;

    phys.frac_anyOver_true = mean(anyOver_true);
    phys.frac_anyOver_hat  = mean(anyOver_hat);
end

function [maxPgrid, maxRatio, anyOver] = pgrid_stats(dqg,ddqg,tau1,gq,Pmax,alpha,fricB,fricFc,vel_eps,eta_share)
    Nv = size(dqg,1);
    K  = size(dqg,2);
    nJ = size(dqg,3);

    a   = reshape(double(alpha(:)), Nv,1,1);
    a2  = a.^2;

    dq  = dqg  .* a;
    % ddq = ddqg .* a2; %#ok<NASGU> % 目前 Pgrid 里不直接用 ddq
    tau = gq + a2 .* (tau1 - gq);

    % friction
    tau = tau + fricB.*dq + fricFc.*tanh(dq/max(double(vel_eps),1e-9));

    Pj   = tau .* dq;
    Ppos = sum(max(Pj,0), 3);         % Nv×K
    Pneg = sum(max(-Pj,0),3);
    Pgrid = max(Ppos - eta_share.*Pneg, 0);

    Pmax2 = reshape(max(double(Pmax(:)),1e-9), Nv,1);
    ratio = Pgrid ./ Pmax2;

    maxPgrid = max(Pgrid,[],2);
    maxRatio = max(ratio,[],2);
    anyOver  = any(ratio > 1.0 + 1e-12, 2);
end

function Tb = make_bins_table(Pmax, y, yhat, apost_t, apost_h, phys)
    Pmax = double(Pmax(:));
    edges = [min(Pmax), pct(Pmax,20), pct(Pmax,40), pct(Pmax,60), pct(Pmax,80), max(Pmax)];
    edges = unique(edges);
    if numel(edges) < 3
        edges = linspace(min(Pmax), max(Pmax)+eps, 3);
    end
    bin = discretize(Pmax, edges);

    bins = unique(bin(~isnan(bin)));
    nb = numel(bins);

    Tb = table();
    Tb.bin = bins(:);
    Tb.Pmax_lo = zeros(nb,1);
    Tb.Pmax_hi = zeros(nb,1);
    Tb.n = zeros(nb,1);

    Tb.y_mean    = zeros(nb,1);
    Tb.yhat_mean = zeros(nb,1);
    Tb.apost_mean_true = zeros(nb,1);
    Tb.apost_mean_hat  = zeros(nb,1);

    if isfield(phys,'frac_anyOver_hat')
        Tb.anyOver_hat = zeros(nb,1);
        Tb.anyOver_true= zeros(nb,1);
        Tb.maxRatio_hat_mean  = zeros(nb,1);
        Tb.maxRatio_true_mean = zeros(nb,1);
    end

    for i=1:nb
        b = bins(i);
        I = (bin==b);
        Tb.n(i) = sum(I);
        Tb.Pmax_lo(i) = edges(find(edges<=min(Pmax(I))+eps,1,'last'));
        Tb.Pmax_hi(i) = edges(find(edges>=max(Pmax(I))-eps,1,'first'));

        Tb.y_mean(i)    = mean(y(I));
        Tb.yhat_mean(i) = mean(yhat(I));
        Tb.apost_mean_true(i) = mean(apost_t(I));
        Tb.apost_mean_hat(i)  = mean(apost_h(I));

        if isfield(phys,'maxRatio_hat')
            Tb.anyOver_hat(i)  = mean(phys.anyOver_hat(I));
            Tb.anyOver_true(i) = mean(phys.anyOver_true(I));
            Tb.maxRatio_hat_mean(i)  = mean(phys.maxRatio_hat(I));
            Tb.maxRatio_true_mean(i) = mean(phys.maxRatio_true(I));
        end
    end
end

function write_summary_txt(fn, rep)
    fid = fopen(fn,'w');
    if fid<0, warning('无法写入 %s', fn); return; end

    fprintf(fid,'=== TRF SharePack Summary ===\n');
    fprintf(fid,'N=%d, K=%d, Dtot=%d, Din=%d | train=%d, val=%d\n', ...
        rep.meta.N, rep.meta.K, rep.meta.Dtot, rep.meta.Din, rep.meta.Ntrain, rep.meta.Nval);

    fprintf(fid,'\n[y stats]\n');   fprintf(fid,'%s\n', stats_line(rep.y_stats));
    fprintf(fid,'\n[yhat stats (val)]\n'); fprintf(fid,'%s\n', stats_line(rep.yhat_stats));

    fprintf(fid,'\n[metrics (a_des, val)]\n');  fprintf(fid,'%s\n', met_line(rep.metrics_des));
    fprintf(fid,'\n[metrics (a_post, val)]\n'); fprintf(fid,'%s\n', met_line(rep.metrics_post));

    fprintf(fid,'\n[post over/under]\n');
    fprintf(fid,'over_rate=%.4f (tol=%.3g), under_rate=%.4f\n', rep.over_rate, 1e-3, rep.under_rate);

    fprintf(fid,'\n[look-ahead sensitivity |Δyhat|]\n'); fprintf(fid,'%s\n', stats_line(rep.lookahead_delta_stats));

    if isfield(rep,'physics') && isfield(rep.physics,'frac_anyOver_hat')
        fprintf(fid,'\n[physics horizon Pgrid (PINN口径)]\n');
        fprintf(fid,'anyOver_true=%.4f, anyOver_hat=%.4f\n', rep.physics.frac_anyOver_true, rep.physics.frac_anyOver_hat);
    end

    fclose(fid);
end

function disp_summary(rep)
    fprintf('\n=== TRF SharePack (console summary) ===\n');
    fprintf('N=%d, K=%d, Dtot=%d, Din=%d | train=%d, val=%d\n', ...
        rep.meta.N, rep.meta.K, rep.meta.Dtot, rep.meta.Din, rep.meta.Ntrain, rep.meta.Nval);

    fprintf('\n[y stats]\n%s\n', stats_line(rep.y_stats));
    fprintf('\n[yhat stats (val)]\n%s\n', stats_line(rep.yhat_stats));
    fprintf('\n[metrics a_des (val)]\n%s\n', met_line(rep.metrics_des));
    fprintf('\n[metrics a_post (val)]\n%s\n', met_line(rep.metrics_post));
    fprintf('\n[post over/under]\n over_rate=%.4f  under_rate=%.4f\n', rep.over_rate, rep.under_rate);
    fprintf('\n[look-ahead |Δyhat|]\n%s\n', stats_line(rep.lookahead_delta_stats));

    if isfield(rep,'physics') && isfield(rep.physics,'frac_anyOver_hat')
        fprintf('\n[physics horizon Pgrid]\n anyOver_true=%.4f  anyOver_hat=%.4f\n', ...
            rep.physics.frac_anyOver_true, rep.physics.frac_anyOver_hat);
    end
    fprintf('======================================\n\n');
end

function s = stats_line(st)
    s = sprintf('mean=%.4f std=%.4f min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f | frac>0.99=%.3f frac<0.50=%.3f', ...
        st.mean, st.std, st.min, st.p05, st.p50, st.p95, st.max, st.frac_gt_099, st.frac_lt_050);
end

function s = met_line(m)
    s = sprintf('mse=%.6g rmse=%.6g mae=%.6g corr=%.4f r2=%.4f bias=%.6g', ...
        m.mse, m.rmse, m.mae, m.corr, m.r2, m.bias);
end
