function req(S, names)
if ischar(names), names={names}; end
for i=1:numel(names)
    assert(isfield(S,names{i}) && ~isempty(S.(names{i})), 'cfg 缺少字段：%s', names{i});
end
end
