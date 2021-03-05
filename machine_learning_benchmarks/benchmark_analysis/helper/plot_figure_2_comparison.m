% Run this only from within plot_figure_2
% Otherwise, parameters will not be set!
figure(1); clf;
Nmax = 10;
add_annotations = true;
save_figure_to_latex = false;
show_errorbar = true;
show_figure = true;

% colors1 = {[228,26,28]/255, [51,160,44]/255*0.7, [77,175,74]/255,...
%            [152,78,163]/255, [255,127,0]/255, [5,55,151]/255,...
%            [166,86,40]/255,  [1,1,1]*0};
colors2 = reshape(cell2mat(colors1), [3,length(colors1)])';
% additional_suffixes = '_colorbar';

lw = 1; % linewidth
ms = 3; % markersize

f = gcf;

set(f, 'Visible', figure_visibility{show_figure+1})
set(f, 'Units', 'inches', 'Position', [plotxoffset,6, 10, 5], ...
            'PaperType', 'usletter'); 
pos = get(f, 'Position'); set(f, 'PaperSize', pos(3:4))
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');


problem = 'Addition'; pick_experiments;

labels = labels_long(ordering);
XTickLabels = {20,30,'',50,'',70,'','',150,'',250,'','',400, '', '', '','',900};
subplot(2,3,1)
problem = 'Addition'; pick_experiments;
yplot = 'min_MSE';
plot_addition_multiplication
plot([10,700], [0.166,0.166], '--', 'Color', [1,1,1]*0.3, 'LineWidth', lw)
ylim([1e-7,0.4])
xlim([0,800])
set(gca, 'YScale', 'log', 'LineWidth', 1, 'FontSize', 10)
set(gca, 'YMinorTick', 'off', 'YMinorGrid', 'off')
set(gca, 'XMinorTick', 'off', 'XMinorGrid', 'off')
set(gca, 'YTick', [1e-7,1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
set(gca, 'XTick', [20:10:80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700])
set(gca, 'XTickLabel', XTickLabels)
ylabel('MSE')
title('Addition problem')
ids_addition = ids;

subplot(2,3,4)
yplot = 'max_per_corr';
plot_addition_multiplication
plot([10,700], [0.0783,0.0783], '--', 'Color', [1,1,1]*0.3, 'LineWidth', lw)
ylim([0,1.05])
xlim([0,800])
% xlabel('T')
ylabel('$P_{\mathrm{correct}}$')
set(gca, 'LineWidth', 1, 'FontSize', 10)
set(gca, 'YMinorTick', 'off', 'YMinorGrid', 'off')
set(gca, 'XMinorTick', 'off', 'XMinorGrid', 'off')
set(gca, 'YTick', 0:0.2:1)
set(gca, 'XTick', [20:10:80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700])
set(gca, 'XTickLabel', XTickLabels)
Ns_addition = Ns;
legend(labels_long(ordering), 'Orientation', 'horizontal', ...
            'Position', [0.40,0.515,0,0], 'NumColumns', 3)
legend('boxoff')
%%
problem = 'Multiplication'; pick_experiments;

subplot(2,3,2)
yplot = 'min_MSE';
plot_addition_multiplication
plot([10,900], [0.049,0.049], '--', 'Color', [1,1,1]*0.3, 'LineWidth', lw)
ylim([1e-7,0.4])
xlim([15,1100])
set(gca,    'YScale', 'log', 'LineWidth', 1, 'FontSize', 10, ...
            'YMinorTick', 'off', 'YMinorGrid', 'off', ...
            'XMinorTick', 'off', 'XMinorGrid', 'off', ...
            'YTick', [1e-7,1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], ...
            'XTick', [20:10:80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700,800,900], ...
            'XTickLabel', XTickLabels)
yplot = 'max_per_corr';
title('Multiplication problem')

subplot(2,3,5)
plot_addition_multiplication
plot([10,900], [0.11,0.11], '--', 'Color', [1,1,1]*0.3, 'LineWidth', lw)
xlim([15,1100])
ylim([0,1.05])
xlabel('T')
set(gca,    'LineWidth', 1, 'FontSize', 10, ...
            'YMinorTick', 'off', 'YMinorGrid', 'off', ...
            'XMinorTick', 'off', 'XMinorGrid', 'off', ...
            'YTick', 0:0.2:1, ...
            'XTick', [20:10:80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700,800,900], ...
            'XTickLabel', XTickLabels)

%%
Ns_multiplication = Ns;

ordering = ordering_MNIST;
problem = 'MNIST'; pick_experiments

labels = labels_short(ordering);
subplot(2,3,3)
title('Sequential MNIST')
yplot = 'min_MSE';
plot_MNIST
plot([0,length(ordering)+1],[2.3,2.3], '--', 'Color', [1,1,1]*0.3, 'LineWidth', lw)
ylim([0,2.5])
xlim([0,length(ordering)+1])
ylabel('cross entropy loss')
xtickangle(45)
set(gca,'LineWidth', 1, 'FontSize', 10)

subplot(2,3,6)
yplot = 'max_per_corr';
plot_MNIST
plot([0,length(ordering)+1],[0.1,0.1], '--', 'Color', [1,1,1]*0.3, 'LineWidth', lw)
set(gca, 'YTick', 0:0.2:1)
ylabel('$P_{\mathrm{correct}}$')
ylim([0,1])
xlim([0,length(ordering)+1])
xtickangle(45)
set(gca,'LineWidth', 1, 'FontSize', 10)

Ns_MNIST = Ns;

if add_annotations
    annotation('textbox', [0.075, 0.99, 0, 0], 'string', '\textbf{A}', ...
        'Interpreter', 'latex', 'FontSize',14)
    annotation('textbox', [0.375, 0.99, 0, 0], 'string', '\textbf{B}', ...
        'Interpreter', 'latex', 'FontSize',14)
    annotation('textbox', [0.655, 0.99, 0, 0], 'string', '\textbf{C}', ...
        'Interpreter', 'latex', 'FontSize',14)
%     annotation('textbox', [0.075, 0.5, 0, 0], 'string', '\textbf{D}', ...
%         'Interpreter', 'latex', 'FontSize',14)
%     annotation('textbox', [0.355, 0.5, 0, 0], 'string', '\textbf{E}', ...
%         'Interpreter', 'latex', 'FontSize',14)
%     annotation('textbox', [0.655, 0.5, 0, 0], 'string', '\textbf{F}', ...
%         'Interpreter', 'latex', 'FontSize',14)
%     additional_suffixes = [additional_suffixes, '_annotations'];
end
