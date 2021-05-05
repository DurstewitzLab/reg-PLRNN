clear all
load('Data_Fig3BC.mat');

colors = {[228,26,28]/255,[75,146,204]/255,[77,175,74]/255,...
            [152,78,163]/255,[255,127,0]/255,[5,55,151]/255,[166,86,40]/255};
%col1=colors{1}; %red
%col2=colors{2}; %light blue
col3=colors{3}; %light green
%col3=colors{4}; %purple
col4=colors{5}; %orange
col1=colors{6}; %dark blue

%filter outlier
%--------------------------------------------------------------------------
ind= isinf(res_mse) | isnan(res_mse) | isoutlier(sum(res_amp')); %isoutlier(res_highamp) | isoutlier(res_lowamp);

res_rsquared(ind)=[];
res_mse(ind)=[];
res_mse_high(ind)=[];
res_mse_low(ind)=[];
res_lam(ind)=[];
res_amp(ind,:)=[];
res_amp_true(ind,:)=[];


%normalize
res_mse_low=res_mse_low./sum(res_mse_low);
res_mse_high=res_mse_high./sum(res_mse_high);


%figures:
%--------------------------------------------------------------------------
lams_labs={'0','.006','.067','.667','6.67','66.7'};

lambdas=unique(res_lam);
h1=figure('color','white'); hold on; lw=1; fs=10;  ms=3;
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

for i=1:length(lambdas)
    ind=res_lam==lambdas(i);
    
    %MSE
    %----------------------------------------------------------------------
    x=res_mse(ind);    
    subplot(1,2,1); hold on; box on
    errorbar(i,nanmean(x),nanstd(x)/sqrt(length(x)),'.','color',col1,'LineWidth',lw);
    plot(i,nanmean(x),'s','color',col1,'LineWidth',lw,'MarkerFaceColor',col1,'MarkerSize',ms);
    ylabel('MSE');
    set(gca,'XTick',1:length(lambdas),'XTickLabel',lams_labs,'FontSize',fs);
    xlabel('$\tau$','FontSize',fs)
    xlim([.5 length(lambdas)+.5]);  
    
    x=res_mse_low(ind);
    subplot(1,2,2); hold on; box on
    errorbar(i+.1,nanmean(x),nanstd(x)/sqrt(length(x)),'.','color',col4,'LineWidth',lw);
    l1=plot(i+.1,nanmean(x),'s','color',col4,'LineWidth',lw,'MarkerFaceColor',col4,'MarkerSize',ms);
    ylabel('MSE');
    set(gca,'XTick',1:length(lambdas),'XTickLabel',lams_labs,'FontSize',fs);
    xlabel('$\tau$','FontSize',fs)
    xlim([.5 length(lambdas)+.5]);
    
    x=res_mse_high(ind);
    ind=x>.08;
    x(ind)=[];
    disp('caution: outlier removed by hand: line 190');
    
    errorbar(i,nanmean(x),nanstd(x)/sqrt(length(x)),'.','color',col3,'LineWidth',lw);
    l2=plot(i,nanmean(x),'s','color',col3,'LineWidth',lw,'MarkerFaceColor',col3,'MarkerSize',ms);
    ylabel('MSE');
    set(gca,'XTick',1:length(lambdas),'XTickLabel',lams_labs,'FontSize',fs);
    xlabel('$\tau$','FontSize',fs)
    xlim([.5 length(lambdas)+.5]);
    clear l1 l2
end
set(h1,'Position',[200 400 600 200])