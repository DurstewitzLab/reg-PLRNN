function [mu0_est,B_est,G_est,W_est,h_est,C_est,S_est,Ezi,Vest,LL,mu0all]= ...
    EMiter_LDS2pari1(CtrPar,W_est,C_est,S_est,Inp,mu0all,B_est,G_est,h_est,X,XZspl,Lb,Ub)
%
% please cite (& consult for further details): ******
% 
% implements EM iterations for LDS
% z_t = W z_t-1 + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B z_t + nu_t , nu_t ~ N(0,G)
%
% NOTE: Inp, mu0_est, X can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% CtrPar=[tol MaxIter __ eps]: vector of control parameters:
% -- tol: relative (to first LL value) tolerated increase in log-likelihood
% -- MaxIter: maximum number of EM iterations allowed
% -- __: 3rd param. is irrelevant for LDS (just included to parallel PLRNN files) 
% -- eps: singularity parameter in StateEstLDS
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
%
%
% OUTPUTS:
% final estimates of network parameters {mu0_est,B_est,G_est,W_est,A_est,S_est}
% Ezi: MxT matrix of state expectancies as returned by StateEstLDS
% Vest: estimated state covariance matrix E[zz']-E[z]E[z]'
% LL: log-likelihood (vector) as function of EM iteration 


if nargin<11, XZspl=[]; end;
if nargin<12, Lb=[]; end;
if nargin<13, Ub=[]; end;

tol=CtrPar(1);
MaxIter=CtrPar(2);
eps=CtrPar(3);
fixedS=CtrPar(4);   % S to be considered fixed or to be estimated
fixedC=CtrPar(5);   % C to be considered fixed or to be estimated
fixedB=CtrPar(6);   % S to be considered fixed or to be estimated
fixedG=CtrPar(7);   % C to be considered fixed or to be estimated

NP=length(X);   % number parallel processes
%pp=parpool(NP);


%% EM loop
i=1; LLR=1e8; LL=[];
Ezi=cell(1,NP);
%Vest=cell(1,NP);
while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<MaxIter
    
    % E-step
%     parfor np=1:NP
%         [Ezi{np},Vest{np}]=StateEstLDS2(W_est,C_est,S_est,Inp{np},mu0all{np},B_est,G_est,h_est,X{np},eps);
%     end
%     EV=[]; Ntr=zeros(1,NP+1);
%     for np=1:NP
%         EV=UpdateExpSumsLDS(Ezi{np},Vest{np},X{np},Inp{np},EV);
%         Ntr(np+1)=EV.ntr;
%     end
%     % for sequential updating (NOTE: to reduce memory load, E-matrices may
%     % be discarded after each run):
%     EV=[]; Ntr=zeros(1,NP+1);
%     for np=1:NP
%         [Ezi{np},Vest{np}]=StateEstLDS2(W_est,C_est,S_est,Inp{np},mu0all{np},B_est,G_est,h_est,X{np},eps);
%         EV=UpdateExpSumsLDS(Ezi{np},Vest{np},X{np},Inp{np},EV);
%         Ntr(np+1)=EV.ntr;
%     end

    % for sequential updating (NOTE: to reduce memory load, E-matrices may
    % be discarded after each run):
    
    EV=[]; Ntr=zeros(1,NP+1);
    for np=1:NP
        %disp(np)
        [Ezi{np},Vest]=StateEstLDS2(W_est,C_est,S_est,Inp{np},mu0all{np},B_est,G_est,h_est,X{np},eps);
        EV=UpdateExpSumsLDS(Ezi{np},Vest,X{np},Inp{np},EV);
        Ntr(np+1)=EV.ntr;
    end
    
    % M-step
    if fixedS, S0=S_est; else S0=[]; end
    if fixedC, C0=C_est; else C0=[]; end
    if fixedB, B0=B_est; else B0=[]; end
    if fixedG, G0=G_est; else G0=[]; end

   % [mu0_est,B_est,G_est,W_est,S_est,C_est,h_est]=ParEstLDSi0(EV,XZspl,S0,C0,B0,G0,Lb,Ub);

   [mu0_est,B_est,G_est,W_est,S_est,C_est,h_est]=ParEstLDSi1(EV,XZspl,S0,C0,B0,G0,Lb,Ub);
    
    % compute log-likelihood (alternatively, use ELL output from ParEstLDS)
    EziAll=cell2mat(Ezi);
    if iscell(X{1})
        LL(i)=LogLikeLDS2(W_est,C_est,S_est,Inp{1:end},mu0_est,B_est,G_est,h_est,X{1:end},EziAll);
        for np=1:NP, mu0all{np}=mu0_est(Ntr(np)+1:Ntr(np+1)); end
    else
        LL(i)=LogLikeLDS2(W_est,C_est,S_est,Inp,mu0_est,B_est,G_est,h_est,X,EziAll);
        mu0all=mu0_est;
    end
    disp(['LL= ' num2str(LL(i))]);
    
    if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end;    % LL ratio 
    i=i+1;
    
end;
disp(['fin LL = ' num2str(LL(end)) ' , #iterations = ' num2str(i-1)]);
%delete(pp);


%%
% (c)  ******
