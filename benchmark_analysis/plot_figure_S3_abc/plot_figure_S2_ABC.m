%% Plot figure S2 A,B,C from:
%   Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher,`
%   Daniel Durstewitz, Identifying nonlinear dynamical systems with multiple
%   time scales and long-range dependencies, ICLR (2021)
% Copyright: Dept. of Theoretical Neuroscience, CIMH, Heidelberg University
clear all
addpath("helper")
load("results_S2_ABC.mat")
%% 1) Plot the curve to get the best d_hidden with iRNN
add_annotations = true;
show_errorbar = true;

lw = 1; % linewidth
ms = 3; % markersize
col = [5,55,151]/255;
additional_suffixes = '';
f = gcf;
clf
set(f, 'Visible', 'on')
set(f, 'Units', 'inches', 'Position', [21,3, 10, 2.5], ...
            'PaperType', 'usletter'); 
pos = get(f, 'Position'); set(f, 'PaperSize', pos(3:4))
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

% Find best d_hidden
problem = 'Addition';
Nmax = 10;
ids =  get_mask(configs, 'problem', '==', problem,...
                           'network_structure', '==', 'PLRNN', ...
                           'tau_A', '==', 5, 'tau_W', '==', 5, ...
                           'T','==',300, 'd_hidden', '>',2, 'per_reg', '==', 0.5);
y = [results(ids).min_MSE];
x = [configs(ids).d_hidden];
x_u = unique(x);
xplot = [];
yplot = [];
mean_y = zeros(1, length(x_u));
std_y = zeros(1,length(x_u));
Ns = zeros(2,length(x_u));
for i=1:length(x_u)
    j = find(x == x_u(i)); j = j(1:min(length(j),Nmax));
    mean_y(i) = nanmean(y(j));
    std_y(i) = nanstd(y(j))/(length(y(j))-1);
    xplot = [xplot, x(j)]; yplot = [yplot, y(j)];
    Ns(:,i) = [x_u(i);length(j)];
end

subplot(1,3,1)
hold on
errorbar(x_u, mean_y, std_y, 's-', 'LineWidth', lw, ...
         'MarkerFaceColor', 'auto', 'MarkerSize', ms, 'Color', col)

ylabel('MSE')
xlabel('$M$')
xlim([5,75])
ylim([7e-2, 0.2])
yl = ylim;
plot([40,40], [yl(1), yl(2)], '--', 'Color', [1,1,1]*0.3)
set(gca, 'YScale', 'log')
set(gca, 'XTick', 10:10:60)

problem = 'Addition';
ids =  get_mask(configs, 'problem', '==', problem,...
                           'network_structure', '==', 'PLRNN', ...
                           'tau_A', '', 0, 'tau_W', '', 0, ...
                           'per_reg', '==', 0.5, ...
                           'T','==',200)';
Nmax = 5;
y = [results(ids).min_MSE];
x = [configs(ids).tau_A];
x_u = unique(x);
mean_y = zeros(1, length(x_u));
std_y = zeros(1,length(x_u));
for i=1:length(x_u)
    y_this = y(x == x_u(i));
    y_this = y_this(1:min(Nmax, length(y_this)));
    mean_y(i) = nanmean(y_this);
    std_y(i) = nanstd(y_this)/sqrt(length(y_this)-1);
end

subplot(1,3,2)
hold on
errorbar(x_u, mean_y, std_y, 'sk-', 'LineWidth', lw, ...
         'MarkerFaceColor', 'auto', 'MarkerSize', ms, 'Color', col)
plot([5,5], [1e-6,1], '--', 'Color', [1,1,1]*0.3)
grid('on'); box('on')
set(gca, 'XScale', 'log', 'YScale', 'log', ...
         'XTick', x_u, 'YTick', [1e-5,1e-4,1e-3,1e-2,1e-1], ...
         'XTick', [0.3,1,2,5,10,20,50,100],...
         'YMinorTick', 'off', 'YMinorGrid', 'off', ...
         'XMinorTick', 'off', 'XMinorGrid', 'off',...
         'FontSize', 9)
xlabel('$\tau$');
ylabel('MSE')
xlim([0.2, 130])
ylim([3e-6, 5e-1])

% Check for rPLRNN how to chose per_reg for fixed tau = 5
ids =  get_mask(configs, 'problem', '==', problem,...
                           'network_structure', '==', 'PLRNN', ...
                           'tau_A', '==', 5, 'tau_W', '==', 5, ...
                           'per_reg', '', 0.5, ...
                           'T','==',200)';
ids =  [ids; get_mask(configs, 'problem', '==', problem,...
                           'network_structure', '==', 'PLRNN', ...
                           'tau_A', '', 0, 'tau_W', '', 0, ...
                           'per_reg', '==', 0, ...
                           'T','==',200)'];
y = [results(ids).min_MSE];
x = [configs(ids).per_reg];
x_u = unique(x);
mean_y = zeros(1, length(x_u));
std_y = zeros(1,length(x_u));
for i=1:length(x_u)
    y_this = y(x == x_u(i));
    y_this = y_this(1:min(Nmax, length(y_this)));
    mean_y(i) = nanmean(y_this);
    std_y(i) = nanstd(y_this)/sqrt(length(y_this)-1);
end

subplot(1,3,3)
hold on
errorbar(x_u, mean_y, std_y, 'sk-', 'LineWidth',1, ...
        'MarkerFaceColor', 'auto', 'MarkerSize', 4, 'Color', col)
box;grid('on')
ylim([3e-6,5e-1])
xlim([-0.03, 1.03])
set(gca, 'YScale', 'log', ...
         'XTick', x_u, 'YTick', [1e-5,1e-4,1e-3,1e-2,1e-1], ...
         'XTick', [0.1:0.2:1], ...
         'YMinorTick', 'off', 'YMinorGrid', 'off', ...
         'XMinorTick', 'off', 'XMinorGrid', 'off',...
         'FontSize', 9)
xlabel('$M_{\mathrm{reg}}/M$')
ylabel('MSE')
plot([0.5,0.5], [1e-5,1], '--', 'Color', [1,1,1]*0.3)

for i=1:3
    subplot(1,3,i)
    set(gca, 'LineWidth', 1.2)
    set(gca, 'FontSize', 10)
    box('on'); grid('on')
end

annotation('textbox', [0.1, 0.99, 0, 0], 'string', '\textbf{A}', ...
        'Interpreter', 'latex', 'FontSize',14)
annotation('textbox', [0.375, 0.99, 0, 0], 'string', '\textbf{B}', ...
    'Interpreter', 'latex', 'FontSize',14)
annotation('textbox', [0.66, 0.99, 0, 0], 'string', '\textbf{C}', ...
    'Interpreter', 'latex', 'FontSize',14)
