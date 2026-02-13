function M = load_model_alpha_strict(model_path)
% 严格载入 alpha 模型：
% - .mat 内必须有变量 `model`
% - model.net   : dlnetwork
% - model.preproc.mu/sig : 数值向量（维度一致）
% - model.preproc.keepBaseIdx : 可选（逻辑/0-1向量）
% - model.preproc 的其余字段：原样透传（例如 Transformer look-ahead 的 K/ds 等）
%
% 返回值 M：struct，包含字段：
%   M.net               (dlnetwork)
%   M.preproc.mu        (row double)
%   M.preproc.sig       (row double)
%   M.preproc.keepBaseIdx (可选，row logical)

    % --- 基本检查 ---
    assert(ischar(model_path) || isstring(model_path), '模型路径必须为字符串。');
    model_path = char(model_path);
    assert(exist(model_path,'file')==2, '模型文件不存在：%s', model_path);

    S = load(model_path);

    % --- 严格格式：必须有变量 `model` ---
    assert(isfield(S,'model') && isstruct(S.model), ...
        '模型文件必须包含变量 "model"（struct）。');

    model = S.model;

    % --- net: dlnetwork ---
    assert(isfield(model,'net'), '缺少字段 model.net。');
    assert(isa(model.net,'dlnetwork'), 'model.net 必须是 dlnetwork。');

    % --- preproc: mu/sig（必须）；keepBaseIdx（可选）；其余字段透传 ---
    assert(isfield(model,'preproc') && isstruct(model.preproc), ...
        '缺少字段 model.preproc。');
    P = model.preproc;
    assert(isfield(P,'mu') && isfield(P,'sig'), ...
        'model.preproc 必须包含 mu 与 sig。');

    mu  = double(P.mu(:)).';   % row
    sig = double(P.sig(:)).';  % row
    assert(isvector(mu) && isvector(sig) && numel(mu)==numel(sig) && numel(mu)>0, ...
        'preproc.mu/sig 维度非法或不一致。');

    M = struct();
    M.net = model.net;

    % 先透传全部 preproc 字段（例如 K/ds/kind 等），再强制修正 mu/sig/keepBaseIdx 的类型。
    M.preproc = P;
    M.preproc.mu  = mu;
    M.preproc.sig = sig;

    if isfield(P,'keepBaseIdx') && ~isempty(P.keepBaseIdx)
        kb = logical(P.keepBaseIdx(:)).';
        M.preproc.keepBaseIdx = kb;
    else
        % 如果未提供 keepBaseIdx，保持字段缺省（由调用侧决定如何处理）
        if isfield(M.preproc,'keepBaseIdx')
            M.preproc = rmfield(M.preproc,'keepBaseIdx');
        end
    end
end
