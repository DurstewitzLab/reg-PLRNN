function ids = get_mask(configs, varargin)
% Get ids of experiments given by parameters of varargin.
% See examples in pick_experiments.m
mask = zeros(length(varargin)/3, length(configs));
idx = 1;
for i=1:3:length(varargin)
    if ischar(varargin{i+2}) && (strcmp('==', varargin{i+1}) || strcmp('=', varargin{i+1}))
        mask(idx,:) = strcmp(varargin{i+2}, {configs(:).(varargin{i})});
    elseif ischar(varargin{i+2}) && (strcmp('~=', varargin{i+1}) || strcmp('!=', varargin{i+1}))
        mask(idx,:) = ~strcmp(varargin{i+2}, {configs(:).(varargin{i})});
    elseif ischar(varargin{i+2}) && isempty(varargin{i+1})
        mask(idx,:) = 1;
    elseif isnumeric(varargin{i+2})
        if strcmp('==', varargin{i+1})
            mask(idx,:) = [configs(:).(varargin{i})] == varargin{i+2};
        elseif strcmp('>=', varargin{i+1})
            mask(idx,:) = [configs(:).(varargin{i})] >= varargin{i+2};
        elseif strcmp('<=', varargin{i+1})
            mask(idx,:) = [configs(:).(varargin{i})] <= varargin{i+2};
        elseif strcmp('>', varargin{i+1})
            mask(idx,:) = [configs(:).(varargin{i})] > varargin{i+2};
        elseif strcmp('<', varargin{i+1})
            mask(idx,:) = [configs(:).(varargin{i})] < varargin{i+2};
        elseif strcmp('~=', varargin{i+1})
            mask(idx,:) = [configs(:).(varargin{i})] ~= varargin{i+2};
        elseif isempty(varargin{i+1})
            mask(idx,:) = 1;
        end
    end
    idx = idx+1;
end
ids = find(all(mask,1));
fprintf('selected %i experiments\n', length(ids));
end

% function ids = get_mask(configs, varargin)
% mask = zeros(length(varargin)/2, length(configs));
% idx = 1;
% for i=1:2:length(varargin)
%     mask(idx,:) = eval(strcat('all([configs(:).',varargin{i},']', varargin{i+1},')'));
%     idx = idx+1;
% end
% ids = all(mask,1);
% end
% 
