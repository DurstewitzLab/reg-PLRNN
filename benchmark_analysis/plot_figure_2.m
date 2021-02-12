%% Plot figure 2 from paper:
% Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher, 
% Daniel Durstewitz, Identifying nonlinear dynamical systems with multiple 
% time scales and long-range dependencies, ICLR (2021)

%% Configuration starts here:
clear all
plotxoffset = 21; % If you want to move the plots to the right on your screen

colors1 = {[228,26,28]/255, [75,146,204]/255, [77,175,74]/255,...
           [152,78,163]/255, [255,127,0]/255, [5,55,151]/255,...
           [166,86,40]/255,  [51,160,44]/255*0.7, [1,1,1]*0,...
           [1,1,1]*0.5};
ordering = [1  3  5
            2  7  10
            8  4  6]';
ordering_MNIST = [1,2,3,8,7,4,5,6,10];
ordering = reshape(ordering, [1,length(ordering(:))]);
% 1 'RNN', 
% 2 'iRNN (Le et al., 2015)', 
% 3 'npRNN (Talathi \& Vartak, 2016)'
% 4 'PLRNN (Koppe et al., 2019)',
% 5 'iPLRNN', 
% 6 'rPLRNN', 
% 7 'LSTM (Hochreiter \& Schmidhuber, 1997)'
% 8 'oRNN (Vorontsov et al., 2017)'
% 9 'fL2PLRNN'
%10 'L2PLRNN'
%%
addpath(genpath('helper'))
load('benchmark_configs.mat')
load('benchmark_results.mat')
%% Plot comparison figures
plot_figure_2_comparison