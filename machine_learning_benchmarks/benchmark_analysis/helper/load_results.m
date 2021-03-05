%%
verbose = 0;
labels_code = {'uninitialized', 'l2RNN', 'IRNN', 'np_RNN', 'PLRNN', 'PLRNN', 'PLRNN',  'LSTM', 'oRNN', 'L2PLRNN2','L2PLRNN' };

if ~exist('raw_per_corrs', "var")
    fits = dir(fullfile(rootpath, 'Fits'));
    fits = fits(3:end);
    for i = 1:length(fits)
        if ~mod(i, 100)
            fprintf('%d / %d\n', i, length(fits))
        end
        if ~strcmp(fits(i).name(end-3), '_')
            continue
        end
        label = fits(i).name(1:end-4);
        configs(i) = convert_yaml(fullfile(rootpath, 'config_files/', [label, '.yaml']));
        fitpath = fullfile(rootpath, 'Fits', fits(i).name);
        results(i).finished = exist(fullfile(fitpath,'finished.txt'), 'file')>0;
        results(i).N = str2num(fits(i).name(end-2:end));
        try
            if exist(fullfile(fitpath, 'error.txt'))
                results(i).error = textscan(fopen(fullfile(fitpath, 'error.txt')),'%q');
                results(i).error = results(i).error{1};
            end
            raw_per_corrs{i} = csvread(fullfile(fitpath, 'per_corr.txt'));
            raw_MSEs{i} = csvread(fullfile(fitpath, 'mses.txt'));
        catch err
            results(i).folder = fits(i).name;
            results(i).per_corrs = nan;
            results(i).max_per_corr = nan;
            results(i).last_per_corr = nan;
            results(i).MSEs = nan;
            results(i).min_MSE = nan;
            results(i).last_MSE = nan;
            results(i).epoch = 0;
            results(i).load_error = err.message;
            continue
            if verbose
                fprintf('couldnt load %s,%d: %s\n', configs(i).network_structure,max_epoch,err.message)
            end
        end
    end
end
fclose('all');
for i=1:length(fits)
    n = find(strcmp(labels_code, configs(i).network_structure), 1);
    max_epoch = max_epochs(n);
    min_epoch = min_epochs(n);
    label = fits(i).name(1:end-4);
    try
        per_corrs = raw_per_corrs{i};
        MSEs = raw_MSEs{i};
        results(i).per_corrs = per_corrs(per_corrs>0);
        results(i).MSEs = MSEs(MSEs>0);
        if use_unfinished
            results(i).MSEs = results(i).MSEs(1:min(max_epoch, length(results(i).MSEs)));
        else
            results(i).MSEs = results(i).MSEs(1:max_epoch);
        end
        if use_unfinished
            results(i).per_corrs = results(i).per_corrs(1:min(max_epoch, length(results(i).per_corrs)));
        else
            results(i).per_corrs = results(i).per_corrs(1:max_epoch);
        end
%         results(i).max_per_corr = max(per_corrs);
        results(i).max_per_corr = max(results(i).per_corrs);
        results(i).last_per_corr = per_corrs(find(results(i).per_corrs>0, 1, 'last'));
        results(i).min_MSE = min(results(i).MSEs);
        if isempty(results(i).min_MSE)
            results(i).min_MSE = nan; end
        results(i).last_MSE = results(i).MSEs(end);
        if isempty(results(i).last_MSE)
            results(i).last_MSE = nan; end
        results(i).epoch = find(MSEs>0, 1, 'last');       
        results(i).folder = fits(i).name;
%         results(i).has_nan = length(per_corrs)~=length(results(i).per_corrs)...
%                            | length(MSEs)~=length(results(i).MSEs);
        results(i).has_nan = any(isnan(MSEs)) | any(isnan(per_corrs));
        
        assert(length(results(i).MSEs)>min_epoch, ...
               sprintf('%i: Not enough epochs', length(results(i).MSEs)))
    catch err
        results(i).folder = fits(i).name;
        results(i).per_corrs = nan;
        results(i).max_per_corr = nan;
        results(i).last_per_corr = nan;
        results(i).MSEs = nan;
        results(i).min_MSE = nan;
        results(i).last_MSE = nan;
        results(i).epoch = 0;
        results(i).load_error = err.message;
        if verbose
            fprintf('couldnt load %s,%d: %s\n', configs(i).network_structure, max_epoch, err.message)
        end
    end
end
all_results = results;
all_configs = configs;