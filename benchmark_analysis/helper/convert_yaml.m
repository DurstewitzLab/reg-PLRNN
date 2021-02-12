function out_struct = convert_yaml(fname)
% convert yaml file containing fit config to a MATLAB struct
    string_params = {'W_regularization_scheme', 'initialize_regulated',...
                'label', 'network_structure', 'optimizer', 'problem'};
    fid = fopen(fname, 'r');
    str = fscanf(fid, '%c');
    
    out_struct = struct();
    if isempty(find(str == ',', 1)) % new yaml-style
        enter_str = str(end);
        str = erase(str(1:end-1), ' ');
        str_split1 = strsplit(str, enter_str);
    else
        enter_str = str(end);
        str = erase(str(1:end-1), ' ');
        str = erase(str, '\n');
        str = erase(str, '{');
        str = erase(str, '}');
        str = erase(str, enter_str);
        str = erase(str, "'");
        str_split1 = strsplit(str, ',');
    end
    for i=1:length(str_split1)
        str_split2 = strsplit(str_split1{i}, ':');
        field = str_split2{1};
        val = str_split2{2};
        if ~any(contains(string_params, field))
            val = str2num(val);
        end
        
        out_struct.(field) = val;
    end
    if isfield(out_struct, 'problem')
        out_struct.problemID = strcmp(out_struct.problem, 'MNIST') + 1;
    end
end

