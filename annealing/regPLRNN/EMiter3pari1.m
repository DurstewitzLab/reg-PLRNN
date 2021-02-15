function [mu0sv,Bsv,Gsv,Asv,Wsv,Csv,hsv,Ssv,EziSv,VestSv,EphiziSv,EphizijSv,EziphizjSv,LL,mu0allSv,LL_pxz,LL_pz]= ...
    EMiter3pari1(CtrPar,A_est,W_est,C_est,S_est,Inp,mu0all,B_est,G_est,h_est,X,XZspl,Lb,Ub,reg)
% --- this is a parallel implementation based on integration of expectation
% sums across segments!
%
% please cite (& consult for further details): ******
% 
% implements EM iterations for PLRNN system
% z_t = A z_t-1 + W max(z_t-1,0) + h + Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t,0) + nu_t , nu_t ~ N(0,G)
%
% NOTE: Inp, mu0_est, X can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% CtrPar=[tol MaxIter tol2 eps flipOnIt]: vector of control parameters:
% -- tol: relative (to first LL value) tolerated increase in log-likelihood
% -- MaxIter: maximum number of EM iterations allowed
% -- tol2: relative error tolerance in state estimation (see StateEstPLRNN) 
% -- eps: singularity parameter in StateEstPLRNN
% -- flipOnIt: parameter that controls switch from single (i<=flipOnIt) to 
%              full (i>flipOnIt) constraint flipping in StateEstPLRNN
% A_est: initial estimate of MxM diagonal matrix of auto-regressive weights
% W_est: initial estimate of MxM off-diagonal matrix of interaction weights
% S_est: initial estimate of MxM diagonal process covariance matrix
%        (assumed to be constant here)
% Inp: MxT matrix of external inputs, or cell array of input matrices 
% mu0_est: initial estimate of Mx1 vector of initial values, or cell array of Mx1 vectors
% B_est: initial estimate of NxM matrix of regression weights
% G_est: initial estimate of NxN diagonal observation covariance matrix
% h: Mx1 vector of (fixed) thresholds
% X: NxT matrix of observations, or cell array of observation matrices
%
% OPTIONAL INPUTS:
% XZspl: vector [Nx Mz] which allows to assign certain states only to 
%        certain observations; specifically, the first Mz state var are
%        assigned to the first Nx obs, and the remaining state var to the
%        remaining obs; (1:Mz)-->(1:Nx), (Mz+1:end)-->(Nx+1:end)
% Lb: lower bounds on W matrix
% Ub: upper bounds on W matrix
%
%
% OUTPUTS:
% final estimates of network parameters {mu0_est,B_est,G_est,W_est,A_est,S_est}
% Ezi: MxT matrix of state expectancies as returned by StateEstPLRNN
% Vest: estimated state covariance matrix E[zz']-E[z]E[z]'
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% LL: log-likelihood (vector) as function of EM iteration 
% Err: final error returned by StateEstPLRNN
% NumIt: total number of EM + mode-search iterations


if nargin<12, XZspl=[]; end;
if nargin<13, Lb=[]; end;
if nargin<14, Ub=[]; end;
if nargin<15, reg=0; end;

tol=CtrPar(1);
MaxIter=CtrPar(2);
tol2=CtrPar(3);
eps=CtrPar(4);
flipOnIt=CtrPar(5);
FinOpt=CtrPar(6);   % quad. prog. step at end of E-iterations
fixedS=CtrPar(7);   % S to be considered fixed or to be estimated
fixedC=CtrPar(8);   % C to be considered fixed or to be estimated
fixedB=CtrPar(9);   % B to be considered fixed or to be estimated
fixedG=CtrPar(10);   % G to be considered fixed or to be estimated

NP=length(X);   % number parallel processes
%pp=parpool(NP);

%% EM loop
Ezi=cell(1,NP); Vest=cell(1,NP); 
%Ephizi=cell(1,NP); Ephizij=cell(1,NP); Eziphizj=cell(1,NP);
i=1; LLR=1e8; LL=[]; maxLL=-inf;
while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<MaxIter

        
    % E-step
    if i>flipOnIt, flipAll=true; else flipAll=false; end
%     % for parallel updating:
%     parfor np=1:NP
%         [Ezi{np},U]=StateEstPLRNN3c(A_est,W_est,C_est,S_est,Inp{np},mu0all{np},B_est,G_est,h_est,X{np},Ezi{np},[],tol2,eps,flipAll,FinOpt);
%         [Ephizi{np},Ephizij{np},Eziphizj{np},Vest{np}]=ExpValPLRNN2(Ezi{np},U);
%     end
%     EV=[]; Ntr=zeros(1,NP+1);
%     for np=1:NP
%         EV=UpdateExpSums(Ezi{np},Vest{np},Ephizi{np},Ephizij{np},Eziphizj{np},X{np},Inp{np},EV);
%         Ntr(np+1)=EV.ntr;
%     end
    % for sequential updating (NOTE: to reduce memory load, E-matrices may
    % be discarded after each run):
    EV=[]; Ntr=zeros(1,NP+1);
    for np=1:NP
        [Ezi{np},U]=StateEstPLRNN3c(A_est,W_est,C_est,S_est,Inp{np},mu0all{np},B_est,G_est,h_est,X{np},Ezi{np},[],tol2,eps,flipAll,FinOpt);
        [Ephizi{np},Ephizij{np},Eziphizj{np},Vest{np}]=ExpValPLRNN2(Ezi{np},U);
        EV=UpdateExpSums(Ezi{np},Vest{np},Ephizi{np},Ephizij{np},Eziphizj{np},X{np},Inp{np},EV);
        Ntr(np+1)=EV.ntr;
    end
%     % for sequential updating (NOTE: to reduce memory load, E-matrices may
%     % be discarded after each run):
%     EV=[]; Ntr=zeros(1,NP+1);
%     for np=1:NP
%         %disp(np)
%         [Ezi{np},U]=StateEstPLRNN3c(A_est,W_est,C_est,S_est,Inp{np},mu0all{np},B_est,G_est,h_est,X{np},Ezi{np},[],tol2,eps,flipAll,FinOpt);
%         [Ephizi,Ephizij,Eziphizj,Vest{np}]=ExpValPLRNN2(Ezi{np},U);
%         EV=UpdateExpSums(Ezi{np},Vest{np},Ephizi,Ephizij,Eziphizj,X{np},Inp{np},EV);
%         Ntr(np+1)=EV.ntr;
%     end
    
    % M-step
    if fixedS, S0=S_est; else S0=[]; end
    if fixedC, C0=C_est; else C0=[]; end
    if fixedB, B0=B_est; else B0=[]; end
    if fixedG, G0=G_est; else G0=[]; end
    
    [mu0_est,B_est,G_est,W_est,A_est,S_est,C_est,h_est]=ParEstPLRNNi1(EV,XZspl,S0,C0,B0,G0,Lb,Ub,reg);
    
    % compute log-likelihood (alternatively, use ELL output from ParEstPLRNN)
    EziAll=cell2mat(Ezi);
    if iscell(X{1})
        LL(i)=LogLikePLRNN4(A_est,W_est,C_est,S_est,Inp{1:end},mu0_est,B_est,G_est,h_est,X{1:end},EziAll,reg);
        for np=1:NP, mu0all{np}=mu0_est(Ntr(np)+1:Ntr(np+1)); end
    else
        [LL(i),LL_pxz(i), LL_pz(i)]=LogLikePLRNN4(A_est,W_est,C_est,S_est,Inp,mu0_est,B_est,G_est,h_est,X,EziAll,reg);
        mu0all=mu0_est;
    end
    disp(['LL= ' num2str(LL(i))]);
    
    if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end   % LL ratio 
    
    if LL(i)>maxLL
        Asv=A_est; Wsv=W_est; Csv=C_est; Ssv=S_est; mu0sv=mu0_est;
        Bsv=B_est; Gsv=G_est; hsv=h_est; EziSv=Ezi; mu0allSv=mu0all;
        VestSv=Vest; EphiziSv=Ephizi; EphizijSv=Ephizij; EziphizjSv=Eziphizj;
        maxLL=LL(i);
    end
    i=i+1;

end
disp(['LL= ' num2str(LL(end)) ', # iterations= ' num2str(i-1)]);
%delete(pp);


%%
% (c) 2016 ******
