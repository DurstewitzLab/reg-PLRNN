%% Plot figure 4 from paper:
% Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher, 
% Daniel Durstewitz, Identifying nonlinear dynamical systems with multiple 
% time scales and long-range dependencies, ICLR (2021)
% Copyright: Dept. of Theoretical Neuroscience, CIMH, Heidelberg University

addpath('helper')
load('benchmark_configs.mat')
load('benchmark_results.mat')
%%
problem = 'Addition'; pick_experiments;
%%
model_ids = [4,6,9,10]; % PLRNN, rPLRNN, L2fPLRNN, L2pPLRNN
all_evs = cell(1,length(model_ids)); jj=1;
for model_id = model_ids 
    clear evs
    ids_working = ids{model_id};
    
    for j=1:length(ids_working)
        i = ids_working(j);
        evs{j} = results(i).eigenvalues;
    end
    all_evs{jj} = evs; jj=jj+1;
end
%%
clf
subplot(1,2,1)
colors=colormap('lines'); colors = colors([1,4,5,3],:); icol = 1;
ss=.11;
edges=(-eps:ss:2)-ss/2;
model_ids = [4,6,9,10];
cla;hold on; box on; lw=2;
for i=1:length(all_evs)
    model_id = model_ids(i);
    tmp = [all_evs{i}{:}];
    eigs_array = abs([tmp{:}]);
    max_eigs = max(eigs_array, [], 1);
    hh = histogram(max_eigs, 'BinEdges', edges, 'Normalization', 'probability', "FaceColor", colors(icol,:), "EdgeColor", "none",'HandleVisibility','off');
    icol = icol+1;
    stairs(hh.BinEdges(1:end-1), hh.Values, "LineWidth", lw, "Color", hh.FaceColor);
end
ylim([0,1.1])
xlim([-.2,2])
ylabel("relative frequency")
xlabel("$|\lambda|_{max}$ around FPs")
set(gcf,'Position',[2200 400 600 200])
legend({'PLRNN', "rPLRNN", "L2fPLRNN", "L2pPLRNN"}, "Position", [0.22,0.78,0,0])
legend('boxoff')

subplot(1,2,2)
lw2 = 2;
cla;hold on
eps = 0.1;
means = zeros(1,length(model_ids));
stds = zeros(1,length(model_ids));
for i=1:length(model_ids)
    model_id = model_ids(i);
    tmp = [all_evs{i}{:}];
    eigs_array = abs([tmp{:}]);
    min_diffs{i} = min(abs(eigs_array-1));
    means(i) = mean(min_diffs{i});
    stds(i) = std(min_diffs{i});
end

ordering = [3,4,1,2];
for i=1:length(ordering)
    bar(i, means(ordering(i)), 'FaceColor', 'none', 'EdgeColor', colors(ordering(i),:), 'LineWidth', lw2)
    errorbar(i, means(ordering(i)), stds(ordering(i)), 'LineStyle', 'none', 'Color', colors(ordering(i),:), 'LineWidth', lw2)
end

plot([0,4], [0,0], 'k-')
ylim([-0.2,1])
set(gca, 'YTick', [1e-4 1e-3 1e-2 1e-1 1])
ylabel("$\mathrm{min}(||\lambda|_{max}-1|$)")
set(gca, 'XTick', 1:length(model_ids), 'XTickLabels', {'L2f', 'L2p', 'PL','rPL' })
box on;


annotation('textbox', [0.075, 0.99, 0, 0], 'string', '\textbf{A}', ...
    'Interpreter', 'latex', 'FontSize',14)
annotation('textbox', [0.49, 0.99, 0, 0], 'string', '\textbf{B}', ...
    'Interpreter', 'latex', 'FontSize',14)
