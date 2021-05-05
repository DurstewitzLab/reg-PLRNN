clearvars -except filename
load('Data_Fig3A.mat');

colors = {[228,26,28]/255,[75,146,204]/255,[77,175,74]/255,...
        [152,78,163]/255,[255,127,0]/255,[5,55,151]/255,[166,86,40]/255};
    col2=colors{1}; %red
    %col2=colors{2}; %light blue
    %col3=colors{3}; %light green
    %col3=colors{4}; %purple
    %col4=colors{5}; %orange
    col1=colors{6}; %dark blue

sKLxs=KLxs;
sdlambda=dlambda;

KLxs=sKLxs;
dlambda=sdlambda;
lams_labs={'0','.006','.067','.667','6.67','66.7'};

ind=isnan(KLxs);
KLxs(ind)=[];
dlambda(ind)=[];

h2=figure('color','white'); hold on; box on; lw=1; fs=10; ms=3;
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex')
lambdas = unique(dlambda);
col3=[ 0.9100 0.4100 0.1700];
for i=1:length(lambdas)
    ind=dlambda==lambdas(i);
    tmp=KLxs(ind);
  %  plot(i,tmp,'o','color',col3,'LineWidth',lw);
    errorbar(i,nanmean(tmp),nanstd(tmp)/sqrt(sum(~isnan(tmp))),'.','color',col1,'LineWidth',lw);
    plot(i,nanmean(tmp),'s','color',col1,'LineWidth',lw,'MarkerFaceColor',col1,'MarkerSize',ms);

    mKLx(i)=nanmean(tmp);
end
set(gca,'XTick',1:length(lambdas),'XTickLabel',lams_labs,'FontSize',fs);

xlim([0.5 length(lambdas)+.5]); 
set(h2,'Position',[200 400 250 150])
ylabel('$\displaystyle D_{\mathrm{KL}}$','FontSize',fs)
xlabel('$\tau$','FontSize',fs)




