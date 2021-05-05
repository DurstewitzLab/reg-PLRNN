clear all; close all; clc
load('Data_Fig3D.mat','dat')

X=dat.Ym{:};
A=dat.A;
W=dat.W;
h=dat.h;
B=dat.B;
mu0=dat.mu0;
sd=1e-3;
S=eye(size(A)).*sd;

T=size(X,2);
t=1;
z(:,t)=mu0{:};
x(:,t)=B*max(z(:,t),0);
for t=2:T
    z(:,t)=A*z(:,t-1)+W*max(z(:,t-1),0)+h;%+ mvnrnd(zeros(1,size(S,1)),S)';
    x(:,t)=B*max(z(:,t),0);
end

h1=figure('color','white'); lw=1; fs=10;

t=1:1500;

colors = {[228,26,28]/255,[75,146,204]/255,[77,175,74]/255,...
    [152,78,163]/255,[255,127,0]/255,[5,55,151]/255,[166,86,40]/255};
col2=colors{1}; %red
%col2=colors{2}; %light blue
%col3=colors{3}; %light green
%col3=colors{4}; %purple
%col4=colors{5}; %orange
col1=colors{6}; %dark blue


for j=1:3
    subplot(1,3,j); hold on; box on
    l1=plot(X(j,t),'-','color',col1,'LineWidth',lw);

    l2=plot(x(j,t)','-','color',col2,'LineWidth',lw);
    xlabel('time in s','FontSize',fs);
    xtick=t(1:300:end)*.5;
    set(gca,'XTick',t(1:300:end),'XTickLabel',xtick-.5,'FontSize',fs)
end

set(h1,'Position',[300 300 900 150])
legend([l1 l2],{'true','generated'},'FontSize',fs,'Location','Best','Box','off')
subplot(1,3,1); ylabel('V','FontSize',fs);
subplot(1,3,2); ylabel('n','FontSize',fs);
subplot(1,3,3); ylabel('h','FontSize',fs);
        
set(gcf, 'Position', [100,100, 900,200])