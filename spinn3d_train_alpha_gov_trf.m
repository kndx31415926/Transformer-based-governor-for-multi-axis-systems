function model = spinn3d_train_alpha_gov_trf(DS, out_path, opts)
% 训练 look-ahead Transformer：预测 a_des（governor 输入）
%
% 输入 DS（来自 spinn3d_make_dataset_alpha_gov_geom_seq）：
%   DS.Xseq : N×K×Dtot  (double)
%   DS.y    : N×1
%   DS.alpha_min
%   DS.K, DS.ds
%
% opts（建议从 spinn3d_params_block().train_trf 组装）：
%   MaxEpochs, MiniBatchSize, InitialLearnRate
%   LambdaOver, LambdaUnder
%   LambdaPINN (>=0)  —— horizon-PINN 物理残差正则（训练期）
%   LambdaSpeed (>=0) —— 轻度“更快”倾向（鼓励更大的 alpha）
%   LambdaEnd   (>=0) —— 末段减速正则（鼓励收敛稳定）
%
%   Transformer 架构（可选）：
%     DModel (default 128)
%     NumHeads (default 4)
%     FFNDim (default 256)
%     NumLayers (default 2)
%     Dropout (default 0)
%
%   PINN 常量：
%     opts.CAPS  : caps0（需要 qd_max/qdd_max/tau_max/P_axis_max；eta_share/P_brk_peak 可选）
%     opts.FRIC  : fric（B/Fc/vel_eps）
%
% 输出：
%   model.net   : dlnetwork
%   model.preproc.mu/sig/keepBaseIdx : 与在线推理严格对齐
%   model.preproc.K/ds/kind          : look-ahead 口径

	% NOTE: 这里不使用 arguments name-value 形式，是为了允许调用方直接传入 opts 结构体：
	%   spinn3d_train_alpha_gov_trf(DS, out_path, optsStruct)
	% 否则 MATLAB 会认为第 3 个位置参数无效（只允许 1~2 个位置参数）。
	if nargin < 2 || isempty(out_path)
		out_path = "model_alpha_gov_trf.mat";
	end
	if nargin < 3 || isempty(opts)
		opts = struct();
	end

	% ---- defaults (允许 opts 缺字段) ----
	opts.MaxEpochs        = getfield_def(opts,'MaxEpochs',40);
	opts.MiniBatchSize    = getfield_def(opts,'MiniBatchSize',256);
	opts.InitialLearnRate = getfield_def(opts,'InitialLearnRate',8e-4);
	opts.LambdaOver       = getfield_def(opts,'LambdaOver',1);
	opts.LambdaUnder      = getfield_def(opts,'LambdaUnder',8);
	opts.LambdaPINN       = getfield_def(opts,'LambdaPINN',0);
	opts.LambdaSpeed      = getfield_def(opts,'LambdaSpeed',0);
	opts.LambdaEnd        = getfield_def(opts,'LambdaEnd',0);
	opts.EndPhase         = getfield_def(opts,'EndPhase',0.85);
	opts.EndWidth         = getfield_def(opts,'EndWidth',0.05);
	opts.PinnTokenDecay   = getfield_def(opts,'PinnTokenDecay',0);

	% transformer arch
	opts.DModel    = getfield_def(opts,'DModel',128);
	opts.NumHeads  = getfield_def(opts,'NumHeads',4);
	opts.FFNDim    = getfield_def(opts,'FFNDim',256);
	opts.NumLayers = getfield_def(opts,'NumLayers',2);
	opts.Dropout   = getfield_def(opts,'Dropout',0);

	% physics constants
	opts.CAPS = getfield_def(opts,'CAPS',struct());
	if ~isfield(opts,'FRIC') || isempty(opts.FRIC)
		opts.FRIC = struct('B',0,'Fc',0,'vel_eps',1e-3);
	end
	opts.Verbose = logical(getfield_def(opts,'Verbose',true));

	% ---- basic validation ----
	assert(isfinite(opts.MaxEpochs) && opts.MaxEpochs>=1, 'opts.MaxEpochs 无效');
	assert(isfinite(opts.MiniBatchSize) && opts.MiniBatchSize>=1, 'opts.MiniBatchSize 无效');
	assert(isfinite(opts.InitialLearnRate) && opts.InitialLearnRate>0, 'opts.InitialLearnRate 无效');

    assert(isfield(DS,'Xseq') && ~isempty(DS.Xseq), 'DS.Xseq 缺失');
    assert(isfield(DS,'y')    && ~isempty(DS.y),    'DS.y 缺失');
    assert(isfield(DS,'alpha_min') && ~isempty(DS.alpha_min), 'DS.alpha_min 缺失');

    Xseq = double(DS.Xseq);
    y    = double(DS.y(:));

    [N, K, Dtot] = size(Xseq);
    assert(numel(y)==N, 'DS.y 长度与 Xseq 样本数不一致');

    % 固定尾巴 6 维：a_prev, BETA, AUP, ADN, Ts, s_norm
    Dtail = 6;
    baseDim = Dtot - Dtail;
    assert(baseDim > 0, 'Dtot 太小');

    % ========= keepBaseIdx：只在基础特征上做删列（沿 N*K 展平统计方差） =========
    Xbase_flat = reshape(Xseq(:,:,1:baseDim), [], baseDim);
    v = var(Xbase_flat, 0, 1);
    keepBaseIdx = (v > 1e-12) & isfinite(v);
    keepBaseIdx = logical(keepBaseIdx(:));

    Xbase_keep = Xseq(:,:,keepBaseIdx);           % N×K×Db
    Xtail      = Xseq(:,:,baseDim+1:end);         % N×K×6
    Xc         = cat(3, Xbase_keep, Xtail);       % N×K×Din
    Din        = size(Xc,3);

    % ========= train/val split =========
    rng(0);
    idx = randperm(N);
    nTr = max(1, round(0.90*N));
    Itr = idx(1:nTr);
    Iva = idx(nTr+1:end);

    % ========= z-score（mu/sig 在训练集 token 展平上估计） =========
    Xc_tr_flat = reshape(Xc(Itr,:,:), [], Din);
    mu  = mean(Xc_tr_flat, 1);
    sig = std(Xc_tr_flat, 0, 1);
    sig(~isfinite(sig) | sig<1e-12) = 1;

    Z = (Xc - reshape(mu,1,1,Din)) ./ reshape(sig,1,1,Din);   % N×K×Din

    % ========= Transformer 网络 =========
    net = build_trf_encoder(Din, opts.DModel, opts.NumHeads, opts.FFNDim, opts.NumLayers, opts.Dropout);

    % ========= PINN / 物理残差配置 =========
    pinn = build_pinn_cfg(opts, K);

    % ========= 训练循环 =========
    maxEpochs = opts.MaxEpochs;
    mbs       = opts.MiniBatchSize;
    lr        = opts.InitialLearnRate;
    lamO      = opts.LambdaOver;
    lamU      = opts.LambdaUnder;

    lamPINN   = opts.LambdaPINN;
    lamSPD    = opts.LambdaSpeed;
    lamEND    = opts.LambdaEnd;

    trailingAvg = [];
    trailingAvgSq = [];

    if opts.Verbose
        fprintf('[trf] N=%d, K=%d, Dtot=%d, Din=%d | train=%d, val=%d\n', N, K, Dtot, Din, numel(Itr), numel(Iva));
        fprintf('[trf] epochs=%d, batch=%d, lr=%.3g | lamPINN=%.3g, lamSPD=%.3g, lamEND=%.3g\n', ...
            maxEpochs, mbs, lr, lamPINN, lamSPD, lamEND);
    end

    % 预取：物理索引（相对于 DS.Xseq 的第 3 维）
    nJ = infer_nJ_from_caps(opts.CAPS);
    idxF = decode_feat_indices(nJ);

    for epoch = 1:maxEpochs
        % shuffle
        Itr = Itr(randperm(numel(Itr)));

        nIter = ceil(numel(Itr)/mbs);
        loss_epoch = 0;

        for it = 1:nIter
            I = Itr((it-1)*mbs+1 : min(it*mbs, numel(Itr)));
            B = numel(I);

            % --- inputs (CBT) ---
            Zb = Z(I,:,:);                    % B×K×Din
            dlX = dlarray(single(permute(Zb, [3 1 2])), 'CBT');   % Din×B×K

            % teacher
            dly = dlarray(single(y(I).'), 'CB');

            % governor tail（取 token#1）
            Xc1 = squeeze(Xc(I,1,:));  % B×Din
            dlAp = dlarray(single(Xc1(:,Din-5).'), 'CB');
            dlBe = dlarray(single(Xc1(:,Din-4).'), 'CB');
            dlUp = dlarray(single(Xc1(:,Din-3).'), 'CB');
            dlDn = dlarray(single(Xc1(:,Din-2).'), 'CB');
            dlTs = dlarray(single(Xc1(:,Din-1).'), 'CB');
            dlS0 = dlarray(single(Xc1(:,Din).'),   'CB');

            % 物理特征（raw，来自 DS.Xseq）
            DQG   = permute(Xseq(I,:,idxF.dq),    [3 1 2]);   % nJ×B×K
            DDQG  = permute(Xseq(I,:,idxF.ddq),   [3 1 2]);
            TAUFF = permute(Xseq(I,:,idxF.tau_ff),[3 1 2]);
            GQ    = permute(Xseq(I,:,idxF.g),     [3 1 2]);
            PmaxB = squeeze(Xseq(I,1,idxF.Pmax));             % B×1
            dlDQG   = dlarray(single(DQG),   'CBT');
            dlDDQG  = dlarray(single(DDQG),  'CBT');
            dlTAUFF = dlarray(single(TAUFF), 'CBT');
            dlG     = dlarray(single(GQ),    'CBT');
            dlPmax  = dlarray(single(PmaxB.'), 'CB');

            % --- forward + loss ---
            [L, grad] = dlfeval(@loss_batch, net, dlX, dly, dlAp, dlBe, dlUp, dlDn, dlTs, ...
                                lamO, lamU, pinn, dlDQG, dlDDQG, dlTAUFF, dlG, dlPmax, ...
                                lamPINN, lamSPD, lamEND, dlS0, opts.EndPhase, opts.EndWidth);

            % --- Adam ---
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, grad, trailingAvg, trailingAvgSq, epoch, lr);

            loss_epoch = loss_epoch + double(extractdata(L));
        end

        loss_epoch = loss_epoch / max(nIter,1);

        % --- val ---
        if ~isempty(Iva)
            Lva = validate_epoch(net, Z, y, Xc, Xseq, Iva, mbs, idxF, Din, lamO, lamU, pinn, lamPINN, lamSPD, lamEND, opts.EndPhase, opts.EndWidth);
        else
            Lva = NaN;
        end

        if opts.Verbose
            fprintf('[trf] epoch %3d/%3d | L=%.4g | Lval=%.4g\n', epoch, maxEpochs, loss_epoch, Lva);
        end
    end

    % ========= save =========
    model = struct();
    model.net = net;
    model.preproc = struct();
    model.preproc.mu = mu;
    model.preproc.sig = sig;
    model.preproc.keepBaseIdx = keepBaseIdx;
    if isfield(DS,'K'),  model.preproc.K  = DS.K;  else, model.preproc.K  = K; end
    if isfield(DS,'ds'), model.preproc.ds = DS.ds; else, model.preproc.ds = 0; end
    model.preproc.kind = 'trf_lookahead';

    if ~isempty(out_path)
        model_path = char(out_path);
        save(model_path, 'model', '-v7.3');
        if opts.Verbose
            fprintf('[trf] saved model: %s\n', model_path);
        end
    end
end

% ===================== loss =====================
function [L, grad] = loss_batch(net, dlX, dly_t, dlAp, dlBe, dlUp, dlDn, dlTs, ...
                                lamO, lamU, pinn, dlDQG, dlDDQG, dlTAUFF, dlG, dlPmax, ...
                                lamPINN, lamSPD, lamEND, dlS0, endPhase, endWidth)

    % NN 输出：a0_hat
    dly_hat = forward(net, dlX);   % 1×B

    % teacher governor（监督项）
    a_slew_t = min(dly_t, dlAp + dlUp.*dlTs);
    a_slew_t = max(a_slew_t, dlAp - dlDn.*dlTs);
    a_star   = dlBe.*dlAp + (1-dlBe).*a_slew_t;

    % NN governor（soft constraint + PINN）
    a_slew_h = min(dly_hat, dlAp + dlUp.*dlTs);
    a_slew_h = max(a_slew_h, dlAp - dlDn.*dlTs);
    a_post_h = dlBe.*dlAp + (1-dlBe).*a_slew_h;

    % ===== loss 1：MSE（NN 输出逼近 teacher） =====
    mse_term = mean((dly_hat - dly_t).^2, 'all');

    % ===== loss 2：governor soft constraint（防止“乐观超界/保守欠界”） =====
    over  = max(a_post_h - a_star, 0);
    under = max(a_star - a_post_h, 0);
    soft_term = lamO * mean(over.^2,'all') + lamU * mean(under.^2,'all');

    L = mse_term + soft_term;

    % ===== loss 3：horizon-PINN 物理残差（训练期正则；在线不做硬限制） =====
    if lamPINN > 0 && pinn.enabled
        Lphys = physics_residual_horizon_loss(a_post_h, dlDQG, dlDDQG, dlTAUFF, dlG, dlPmax, pinn);
        L = L + lamPINN * Lphys;
    end

    % ===== loss 4：轻度“更快”倾向（鼓励更大的 alpha） =====
    if lamSPD > 0
        Lspd = -mean(a_post_h,'all');
        L = L + lamSPD * Lspd;
    end

    % ===== loss 5：末段减速（鼓励靠近末端时更稳收敛） =====
    if lamEND > 0 && isfield(pinn,'qd_lim') && isfield(pinn,'nJ')
        Lend = end_slowdown_loss(a_post_h, dlDQG, dlS0, pinn, endPhase, endWidth);
        L = L + lamEND * Lend;
    end

    grad = dlgradient(L, net.Learnables);
end

function Lva = validate_epoch(net, Z, y, Xc, Xseq, Iva, mbs, idxF, Din, lamO, lamU, pinn, lamPINN, lamSPD, lamEND, endPhase, endWidth)
    nIter = ceil(numel(Iva)/mbs);
    acc = 0;
    for it=1:nIter
        I = Iva((it-1)*mbs+1 : min(it*mbs, numel(Iva)));

        Zb = Z(I,:,:);
        dlX = dlarray(single(permute(Zb, [3 1 2])), 'CBT');
        dly = dlarray(single(y(I).'), 'CB');

        Xc1 = squeeze(Xc(I,1,:));
        dlAp = dlarray(single(Xc1(:,Din-5).'), 'CB');
        dlBe = dlarray(single(Xc1(:,Din-4).'), 'CB');
        dlUp = dlarray(single(Xc1(:,Din-3).'), 'CB');
        dlDn = dlarray(single(Xc1(:,Din-2).'), 'CB');
        dlTs = dlarray(single(Xc1(:,Din-1).'), 'CB');
        dlS0 = dlarray(single(Xc1(:,Din).'),   'CB');

        DQG   = permute(Xseq(I,:,idxF.dq),    [3 1 2]);
        DDQG  = permute(Xseq(I,:,idxF.ddq),   [3 1 2]);
        TAUFF = permute(Xseq(I,:,idxF.tau_ff),[3 1 2]);
        GQ    = permute(Xseq(I,:,idxF.g),     [3 1 2]);
        PmaxB = squeeze(Xseq(I,1,idxF.Pmax));
        dlDQG   = dlarray(single(DQG),   'CBT');
        dlDDQG  = dlarray(single(DDQG),  'CBT');
        dlTAUFF = dlarray(single(TAUFF), 'CBT');
        dlG     = dlarray(single(GQ),    'CBT');
        dlPmax  = dlarray(single(PmaxB.'), 'CB');

        % forward
        dly_hat = forward(net, dlX);

        a_slew_t = min(dly, dlAp + dlUp.*dlTs);
        a_slew_t = max(a_slew_t, dlAp - dlDn.*dlTs);
        a_star   = dlBe.*dlAp + (1-dlBe).*a_slew_t;

        a_slew_h = min(dly_hat, dlAp + dlUp.*dlTs);
        a_slew_h = max(a_slew_h, dlAp - dlDn.*dlTs);
        a_post_h = dlBe.*dlAp + (1-dlBe).*a_slew_h;

        mse_term = mean((dly_hat - dly).^2, 'all');
        over  = max(a_post_h - a_star, 0);
        under = max(a_star - a_post_h, 0);
        soft_term = lamO*mean(over.^2,'all') + lamU*mean(under.^2,'all');
        L = mse_term + soft_term;

        if lamPINN>0 && pinn.enabled
            L = L + lamPINN * physics_residual_horizon_loss(a_post_h, dlDQG, dlDDQG, dlTAUFF, dlG, dlPmax, pinn);
        end
        if lamSPD>0
            L = L + lamSPD * (-mean(a_post_h,'all'));
        end
        if lamEND>0 && isfield(pinn,'qd_lim') && isfield(pinn,'nJ')
            L = L + lamEND * end_slowdown_loss(a_post_h, dlDQG, dlS0, pinn, endPhase, endWidth);
        end

        acc = acc + double(extractdata(L));
    end
    Lva = acc / max(nIter,1);
end

% ===================== network =====================
function net = build_trf_encoder(Din, dModel, numHeads, ffnDim, numLayers, dropP)

    assert(dModel>0 && numHeads>0 && numLayers>0);
    numKeyChannels = dModel;  % 通常取 dModel（每头 dModel/numHeads）

    lg = layerGraph();

    % 输入投影（先显式连线，避免 layerGraph 出现悬空层）
    lg = addLayers(lg, sequenceInputLayer(Din, Normalization="none", Name="in"));
    lg = addLayers(lg, fullyConnectedLayer(dModel, Name="embed"));
    lg = addLayers(lg, reluLayer(Name="embed_relu"));
    lg = connectLayers(lg, "in", "embed");
    lg = connectLayers(lg, "embed", "embed_relu");

    % 主干
    prev = "embed_relu";

    for l = 1:numLayers
        saName    = sprintf("sa_%d", l);
        addSaName = sprintf("add_sa_%d", l);
        lnSaName  = sprintf("ln_sa_%d", l);

        fc1Name   = sprintf("ffn1_%d", l);
        actName   = sprintf("ffn_act_%d", l);
        fc2Name   = sprintf("ffn2_%d", l);
        addFName  = sprintf("add_ffn_%d", l);
        lnFName   = sprintf("ln_ffn_%d", l);

        sa = selfAttentionLayer(numHeads, numKeyChannels, Name=saName, DropoutProbability=dropP);
        lg = addLayers(lg, sa);
        lg = addLayers(lg, additionLayer(2, Name=addSaName));
        lg = addLayers(lg, layerNormalizationLayer(Name=lnSaName));

        lg = addLayers(lg, fullyConnectedLayer(ffnDim, Name=fc1Name));
        % geluLayer 可能版本差异，优先 gelu，否则 relu
        if exist('geluLayer','file')==2
            lg = addLayers(lg, geluLayer(Name=actName));
        else
            lg = addLayers(lg, reluLayer(Name=actName));
        end
        lg = addLayers(lg, dropoutLayer(dropP, Name=sprintf("drop_%d",l)));
        lg = addLayers(lg, fullyConnectedLayer(dModel, Name=fc2Name));
        lg = addLayers(lg, additionLayer(2, Name=addFName));
        lg = addLayers(lg, layerNormalizationLayer(Name=lnFName));

        % connections: prev -> sa -> addSa/in1 ; prev -> addSa/in2
        lg = connectLayers(lg, prev, saName);
        lg = connectLayers(lg, saName, addSaName + "/in1");
        lg = connectLayers(lg, prev, addSaName + "/in2");
        lg = connectLayers(lg, addSaName, lnSaName);

        % ffn: lnSa -> fc1 -> act -> dropout -> fc2 -> addF/in1 ; lnSa -> addF/in2
        lg = connectLayers(lg, lnSaName, fc1Name);
        lg = connectLayers(lg, fc1Name, actName);
        lg = connectLayers(lg, actName, sprintf("drop_%d",l));
        lg = connectLayers(lg, sprintf("drop_%d",l), fc2Name);
        lg = connectLayers(lg, fc2Name, addFName + "/in1");
        lg = connectLayers(lg, lnSaName, addFName + "/in2");
        lg = connectLayers(lg, addFName, lnFName);

        prev = lnFName;
    end

    % 取第一个 token（当前步），其 embedding 已通过注意力汇聚未来信息
    if exist('indexing1dLayer','file')==2
        pick = indexing1dLayer("first", Name="pick_first");
        lg = addLayers(lg, pick);
        lg = connectLayers(lg, prev, "pick_first");
        prev2 = "pick_first";
    else
        pick = functionLayer(@(X) X(:,:,1), Name="pick_first", Formattable=true);
        lg = addLayers(lg, pick);
        lg = connectLayers(lg, prev, "pick_first");
        prev2 = "pick_first";
    end

    head = [
        fullyConnectedLayer(64, Name="head_fc1")
        reluLayer(Name="head_relu")
        fullyConnectedLayer(1, Name="head_out")
        sigmoidLayer(Name="sig")
    ];

    lg = addLayers(lg, head);
    lg = connectLayers(lg, prev2, "head_fc1");

    net = dlnetwork(lg);
end

% ===================== PINN =====================
function pinn = build_pinn_cfg(opts, K)
    pinn = struct();
    pinn.enabled = false;

    if ~isstruct(opts.CAPS) || isempty(fieldnames(opts.CAPS))
        return;
    end

    caps = opts.CAPS;
    nJ = infer_nJ_from_caps(caps);
    pinn.nJ = nJ;

    % friction（与 alpha_feasible 口径一致）
    fric = opts.FRIC;
    pinn.fricB    = reshape(double(getfield_def(fric,'B',0)), [], 1);
    pinn.fricFc   = reshape(double(getfield_def(fric,'Fc',0)), [], 1);
    pinn.vel_eps  = double(getfield_def(fric,'vel_eps',1e-3));

    pinn.tau_lim  = reshape(double(getfield_def(caps,'tau_max',inf(nJ,1))), [], 1);
    pinn.qd_lim   = reshape(double(getfield_def(caps,'qd_max',inf(nJ,1))),  [], 1);
    pinn.qdd_lim  = reshape(double(getfield_def(caps,'qdd_max',inf(nJ,1))), [], 1);
    pinn.Pax_lim  = reshape(double(getfield_def(caps,'P_axis_max',inf(nJ,1))), [], 1);

    pinn.eta_share = double(getfield_def(caps,'eta_share',1.0));
    pinn.P_brk_peak = double(getfield_def(caps,'P_brk_peak',getfield_def(caps,'P_dump_max',inf)));

    % token weights（用于 horizon-PINN）
    if isfield(opts,'PinnTokenDecay') && opts.PinnTokenDecay>0
        w = exp(-double(opts.PinnTokenDecay) * (0:K-1));
    else
        w = ones(1,K);
    end
    w = w / max(mean(w), 1e-12);   % 归一化为 mean=1，便于与单步损失同量纲
    pinn.wtoken = reshape(double(w), 1, 1, []);   % 1×1×K（对应 CBT 的 T 维）

    % 各残差权重（可按需调整）
    pinn.w = struct('tau',1.0,'qd',0.1,'qdd',0.1,'Pax',0.5,'Pgrid',1.0,'Pdump',0.2);

    pinn.enabled = true;
end

function Lphys = physics_residual_horizon_loss(alpha, dlDQG, dlDDQG, dlTAUFF, dlG, dlPmax, pinn)
% alpha: 1×B (CB)
% dlDQG/dlDDQG/dlTAUFF/dlG: nJ×B×K (CBT)
% dlPmax: 1×B (CB)

    eps0 = 1e-12;
    nJ = pinn.nJ;

    a2 = alpha.^2;                  % 1×B
    a  = reshape(alpha,  1, [], 1); % 1×B×1
    a2 = reshape(a2,     1, [], 1); % 1×B×1

    dqg  = dlDQG;
    ddqg = dlDDQG;
    tau1 = dlTAUFF;
    gq   = dlG;

    dq  = dqg  .* a;
    ddq = ddqg .* a2;

    tau = gq + a2 .* (tau1 - gq);

    % friction
    Bv = reshape(single(pinn.fricB(1:nJ)), [], 1, 1);
    Fc = reshape(single(pinn.fricFc(1:nJ)),[], 1, 1);
    tau = tau + Bv.*dq + Fc.*tanh(dq./max(pinn.vel_eps,1e-9));

    % token weight
    wT = dlarray(single(pinn.wtoken), 'CBT');   % 1×1×K
    sw = sqrt(wT);

    % tau
    tau_lim = reshape(single(pinn.tau_lim(1:nJ)), [], 1, 1);
    r_tau = max(abs(tau)./max(tau_lim,eps0) - 1, 0) .* sw;
    L_tau = mean(r_tau.^2,'all');

    % qd/qdd
    qd_lim = reshape(single(pinn.qd_lim(1:nJ)), [], 1, 1);
    qdd_lim= reshape(single(pinn.qdd_lim(1:nJ)),[], 1, 1);

    if all(isfinite(pinn.qd_lim))
        r_qd = max(abs(dq)./max(qd_lim,eps0) - 1, 0) .* sw;
        L_qd = mean(r_qd.^2,'all');
    else
        L_qd = 0;
    end
    if all(isfinite(pinn.qdd_lim))
        r_qdd = max(abs(ddq)./max(qdd_lim,eps0) - 1, 0) .* sw;
        L_qdd = mean(r_qdd.^2,'all');
    else
        L_qdd = 0;
    end

    % power
    Pax_lim = reshape(single(pinn.Pax_lim(1:nJ)), [], 1, 1);
    Pj = tau .* dq;
    Pax = max(Pj,0);
    r_Pax = max(Pax./max(Pax_lim,eps0) - 1, 0) .* sw;
    L_Pax = mean(r_Pax.^2,'all');

    Ppos = sum(max(Pj,0),1);   % 1×B×K
    Pneg = sum(max(-Pj,0),1);
    Pgrid = max(Ppos - pinn.eta_share.*Pneg, 0);

    Pmax_eff = reshape(max(dlPmax, 1e-6), 1, [], 1);  % 1×B×1
    r_Pgrid = max(Pgrid./Pmax_eff - 1, 0) .* sw;
    L_Pgrid = mean(r_Pgrid.^2,'all');

    if isfinite(pinn.P_brk_peak)
        share = min(pinn.eta_share.*Pneg, Ppos);
        Pdump = max(Pneg - share, 0);
        r_Pdump = max(Pdump./max(pinn.P_brk_peak,1e-6) - 1, 0) .* sw;
        L_Pdump = mean(r_Pdump.^2,'all');
    else
        L_Pdump = 0;
    end

    Lphys = pinn.w.tau*L_tau + pinn.w.qd*L_qd + pinn.w.qdd*L_qdd + ...
            pinn.w.Pax*L_Pax + pinn.w.Pgrid*L_Pgrid + pinn.w.Pdump*L_Pdump;
end

function Lend = end_slowdown_loss(alpha, dlDQG, dlS0, pinn, endPhase, endWidth)
% 在接近末端（s_norm -> 1）时，抑制 dq，提升稳定到达。
% alpha: 1×B
% dlDQG: nJ×B×K
% dlS0 : 1×B（token#1 的 s_norm）

    nJ = pinn.nJ;
    dqg0 = dlDQG(:,:,1);           % nJ×B
    dq0  = dqg0 .* alpha;          % nJ×B

    qd_lim = reshape(single(pinn.qd_lim(1:nJ)), [], 1);
    v = mean((dq0./max(qd_lim,1e-6)).^2, 1);    % 1×B

    % w_end(s) = sigmoid((s - endPhase)/endWidth)
    z = (dlS0 - endPhase) ./ max(endWidth,1e-6);
    w = 1 ./ (1 + exp(-z));

    Lend = mean(w .* v, 'all');
end

% ===================== indices / utils =====================
function nJ = infer_nJ_from_caps(caps)
    if isstruct(caps) && isfield(caps,'tau_max') && ~isempty(caps.tau_max)
        nJ = numel(caps.tau_max);
    elseif isstruct(caps) && isfield(caps,'qd_max') && ~isempty(caps.qd_max)
        nJ = numel(caps.qd_max);
    else
        nJ = 4; % fallback
    end
end

function idx = decode_feat_indices(nJ)
% 对齐 spinn3d_features_alpha_plus 的固定拼接顺序：
% DS.Xseq token 特征 = [alpha_plus(D0=13*nJ+12), payload10(10), tail6(6)]
    idx = struct();
    idx.dq     = (nJ+1):(2*nJ);
    idx.ddq    = (2*nJ+1):(3*nJ);
    idx.tau_ff = (3*nJ+1):(4*nJ);
    idx.g      = (5*nJ+2):(6*nJ+1);
    idx.Pmax   = (6*nJ+6);
end

function x = getfield_def(S,fn,def)
    if isstruct(S) && isfield(S,fn) && ~isempty(S.(fn))
        x = S.(fn);
    else
        x = def;
    end
end
