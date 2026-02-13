function S = pick_struct(cfg, keys)
S=[]; for i=1:numel(keys)
    k=keys{i};
    if isfield(cfg,k)&&~isempty(cfg.(k))&&isstruct(cfg.(k)), S=cfg.(k); return; end
end
error('cfg 缺少结构体：%s', strjoin(keys,'/'));
end
