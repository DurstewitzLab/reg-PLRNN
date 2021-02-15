function annealing(M,dat,pat_output,outputfile,tau)

if nargin<5, tau=0; end

X=dat.X; 
T=size(X,2);

Ym{1}=X;
Inpm{1}=zeros(M,T);
[N,T]=size(Ym{1});


%% initialization
%--------------------------------------------------------------------------
sd=1;
a=-1.5*sd; b=1.5*sd; 
W0=a+(b-a).*rand(M);
E=eig(W0);
redf=.95;
while find(abs(E)>=1), W0=W0*redf; E=eig(W0); end
%--------------------------------------------------------------------------
C=zeros(M);
h0=0.5*randn(M,1);
B0=randn(N,M);
for i=1:length(Ym), mu00{i}=randn(M,1); end


%% --- 1st ini step by LDS; B,G free to vary, S~G
%--------------------------------------------------------------------------
tol=1e-3;   % relative (to first LL value) tolerated increase in log-likelihood
MaxIter=20; % maximum number of EM iterations allowed
eps=1e-5;   % singularity parameter in StateEstPLRNN
fixedS=1;   % S to be considered fixed or to be estimated
fixedC=1;   % C to be considered fixed or to be estimated
fixedB=0;   % B to be considered fixed or to be estimated
fixedG=0;   % G to be considered fixed or to be estimated
CtrPar=[tol MaxIter eps fixedS fixedC fixedB fixedG];

S=eye(M);
G0=diag(var(cell2mat(Ym)'));  % take data var as initial estim (~1 here)

%  --> actually important to determine some scale of noise a priori!
[mu01,B1,G1,W1,h1,~,~,Ezi1,Vest1,LL1,mu01all]=EMiter_LDS2pari1(CtrPar,W0,C,S,Inpm,mu00,B0,G0,h0,Ym);
disp('first iteration done')

%% --- 2nd step: estimate PLRNN model with B,G free to vary, S~G
%--------------------------------------------------------------------------

fixedB=0;
fixedG=0;   
S=eye(M);

A1=diag(diag(W1)); W1=W1-A1;

tol2=1e-2;  % relative error tolerance in state estimation (see StateEstPLRNN)
flipOnIt=10; % parameter that controls switch from single (i<=flipOnIt) to all
FinOpt=0;   % quad. prog. step at end of E-iterations
CtrPar=[tol MaxIter tol2 eps flipOnIt FinOpt fixedS fixedC fixedB fixedG];


% specify regularization
%A and/or W->1 regulated by lambda here set tau, A and/or W->0 regulated by tau
LMask=zeros(size([A1 W1 h1]));

%pars->1
Aind=1:ceil(M/2);    %regularize half of the states with A->1;
LMask(Aind,1:M)=-1;

%pars->0
Wind=1:ceil(M/2);       %which half of the states with W->0;
LMask(Wind,M+1:2*M)=1;
hind=1:ceil(M/2);
LMask(hind,2*M+1)=1;

reg.Lreg=LMask;
reg.lambda=tau; %specifies strength of regularization on pars->1
reg.tau=tau;   %specifies strength of regularization on pars->0

[mu02,B2,G2,A2,W2,~,h2,~,Ezi2,Vest2,~,~,~,LL2,mu02all]= ...
    EMiter3pari1(CtrPar,A1,W1,C,S,Inpm,mu01all,B1,G1,h1,Ym,[],[],[],reg);
disp('second iteration done')


%% --- 3rd step: estimate PLRNN model with B fixed & smaller S<G
%--------------------------------------------------------------------------
S=0.1*eye(M);
fixedB=1;
CtrPar=[tol MaxIter tol2 eps flipOnIt FinOpt fixedS fixedC fixedB fixedG];

[mu03,B3,G3,A3,W3,~,h3,~,~,~,~,~,~,~,mu03all]= ...
    EMiter3pari1(CtrPar,A2,W2,C,S,Inpm,mu02all,B2,G2,h2,Ym,[],[],[],reg);
disp('third iteration done')

%% --- 4rd step: estimate PLRNN model with B fixed & smaller S<G
%--------------------------------------------------------------------------
S=0.01*eye(M);
[mu04,B4,G4,A4,W4,~,h4,~,~,~,~,~,~,~,~]= ...
    EMiter3pari1(CtrPar,A3,W3,C,S,Inpm,mu03all,B3,G3,h3,Ym,[],[],[],reg);
disp('fourth iteration done')

%% --- 4rd step: estimate PLRNN model with B fixed & smaller S<G
%--------------------------------------------------------------------------
S=0.001*eye(M);
[mu0,B,G,A,W,~,h,~,Ezi,Vest,~,~,~,LL,~]= ...
    EMiter3pari1(CtrPar,A4,W4,C,S,Inpm,mu04,B4,G4,h4,Ym,[],[],[],reg);
disp('fifth iteration done')

%% save to file
save([pat_output outputfile ],'Ym','mu0','B','G','A','W','h','S','C','LL','Ezi', ...
    'B0','G0','h0','W0','mu00',...
    'tau','Vest','reg');


