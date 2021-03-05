function [mu0,B,G,W,A,S,C,h]=ParEstPLRNNi1(EV,XZsplit,S0,C0,B0,G0,Lb,Ub,reg)

%if nargin<9, lam=0; end

% please cite (& consult for further details): ******
%
% implements parameter estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1,0) + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t,0) + nu_t , nu_t ~ N(0,G)
%
% NOTE: X_, Inp_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; info is aggregated
% across trials, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% Ez: MxT matrix of state expectancies as returned by StateEstPLRNN
% V: state covariance matrix E[zz']-E[z]E[z]'
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% X_: NxT matrix of observations, or cell array of observation matrices
% Inp_: KxT matrix of external inputs, or cell array of input matrices
%
% OPTIONAL INPUTS:
% XZsplit: vector [Nx Mz] which allows to assign certain states only to
%          certain observations; specifically, the first Mz state var are
%          assigned to the first Nx obs, and the remaining state var to the
%          remaining obs; (1:Mz)-->(1:Nx), (Mz+1:end)-->(Nx+1:end)
% S0: fix process noise-cov matrix to S0
%
% OUTPUTS:
% mu0: Mx1 vector of initial values, or cell array of Mx1 vectors
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% W: MxM off-diagonal matrix of interaction weights
% A: MxM diagonal matrix of auto-regressive weights
% S: MxM diagonal covariance matrix (Gaussian process noise)
% C: MxK matrix of regression weights multiplying with Kx1 Inp
% h: Mx1 vector of bias terms
% ELL: expected (complete data) log-likelihood


eps=1e-5;   % minimum variance allowed for in S and G

E1=EV.E1; E2=EV.E2; E3=EV.E3; E4=EV.E4; E5=EV.E5; E1p=EV.E1p; E3pkk=EV.E3pkk; E3_=EV.E3_;
F1=EV.F1; F2=EV.F2; F3=EV.F3; F4=EV.F4; F5_=EV.F5_; F6_=EV.F6_;
f5_1=EV.f5_1; f6_1=EV.f6_1;
Zt1=EV.Zt1; Zt0=EV.Zt0; phiZ=EV.phiZ; InpS=EV.InpS;
T=EV.T;
ntr=EV.ntr;
m=size(E1,1);
Minp=size(InpS,1);

F5=F5_+f5_1;
F6=F6_+f6_1;


%% solve for parameters {B,G} of observation model
if nargin>1 && ~isempty(XZsplit)
    Nx=XZsplit(1); Mz=XZsplit(2);
    F1_=F1; F1_(1:Nx,Mz+1:end)=0; F1_(Nx+1:end,1:Mz)=0;
    E1p_=E1p; E1p_(1:Mz,Mz+1:end)=0; E1p_(Mz+1:end,1:Mz)=0;
    if nargin<5 || isempty(B0), B=F1_*E1p_^-1; else B=B0; end
    if nargin<6 || isempty(G0)
        G=diag(max(diag(F2-F1_*B'-B*F1_'+B*E1p_'*B')./T,eps));   % assumes G to be diag
    else G=G0; end
else
    if nargin<5 || isempty(B0), B=F1*E1p^-1; else B=B0; end
    if nargin<6 || isempty(G0)
        G=diag(max(diag(F2-F1*B'-B*F1'+B*E1p'*B')./T,eps));   % assumes G to be diag
    else G=G0; end
end



%% solve for
% - interaction weight matrix W
% - auto-regressive weights A
% - bias terms h
% - external regressor weights C
% in one go:
I=eye(m);
O=ones(m)-I;

if nargin<4 || isempty(C0)
    Mask=[I;O;ones(1,m);ones(Minp,m)];
    AWhC=zeros(m,m+1+Minp);
    EL=[E3,E4',Zt1,F3';E4,E1+lam*I,phiZ,F4';Zt1',phiZ',T-ntr,InpS';F3,F4,InpS,F6_]; %caution: GK: will give error with current setting
    ER=[E2,E5,Zt0,F5_];
else
    Mask=[I;O;ones(1,m)];
    AWhC=zeros(m,m+1);
    
    % EL=[E3,E4',Zt1;E4,E1+lam*I,phiZ;Zt1',phiZ',T-ntr];
    EL=[E3,E4',Zt1;E4,E1,phiZ;Zt1',phiZ',T-ntr];
    ER=[E2-C0*F3,E5-C0*F4,Zt0-C0*InpS];
end
W=zeros(m);

if nargin<7 || (isempty(Lb) && isempty(Ub))
    
    for i=1:m
        
        [Reg_EL, Reg_ER]=getORegularization(reg,Mask,S0,m,i); %get regularization 
        el=EL+Reg_EL;
        er=ER+Reg_ER;
        
        k=find(Mask(:,i));
        X=el(k,k);
        Y=er(i,k);
        AWhC(i,:)=Y*X^-1;
        W(i,:)=[AWhC(i,2:i) 0 AWhC(i,i+1:m)];
    end
    
else
    %S=S0;
    HH=[]; hh=[];
    for i=1:m
        k=find(Mask(:,i));
        X=EL(k,k);
        Y=ER(i,k);
        %HH=blkdiag(HH,S0(k,k)^-1*X);
        %hh=[hh,Y*S0(k,k)^-1];
        HH=blkdiag(HH,X');
        hh=[hh;Y'];
    end
    Lb=Lb'; lb=Lb(1:end)';
    Ub=Ub'; ub=Ub(1:end)';
    awhc=quadprog(HH,-hh,[],[],[],[],lb,ub);
    AWhC=reshape(awhc,size(AWhC,2),m)';
    for i=1:m, W(i,:)=[AWhC(i,2:i) 0 AWhC(i,i+1:m)]; end
end

A=diag(AWhC(:,1));
h=AWhC(:,m+1);
if nargin<4 || isempty(C0)
    C=AWhC(:,m+2:end);
else
    C=C0;
end


%% solve for trial-specific parameters mu0
for i=1:ntr
    mu0{i}=EV.AllIni0(:,i)-C*EV.AllInp(:,i);
end


%% solve for process noise covariance S, or use provided fixed S0
if nargin>2 && ~isempty(S0), S=S0;
else
    H=zeros(m);
    for i=1:ntr
        H=H+EV.Ezz0{i}-EV.AllIni0(:,i)*mu0{i}'-mu0{i}*EV.AllIni0(:,i)'+mu0{i}*mu0{i}'+mu0{i}*EV.AllInp(:,i)'*C'+C*EV.AllInp(:,i)*mu0{i}';
    end
    S=diag(diag(H+E3_'-F5*C'-C*F5'+C*F6*C'-E2*A'-A*E2' ...
        +A*E3'*A'-E5*W'-W*E5'+W*(E1+lam*I)'*W'+A*E4'*W'+W*E4*A'+A*F3'*C'+C*F3*A'+W*F4'*C'+C*F4*W' ...
        -Zt0*h'-h*Zt0'+A*Zt1*h'+h*Zt1'*A'+W*phiZ*h'+h*phiZ'*W'+(T-ntr).*h*h'+C*InpS*h'+h*InpS'*C'))./T;   % assumes S to be diag
end




% % %% compute expected log-likelihood (if desired)
% % (####################  not yet updated for this version)
% % if nargout>6
% %     E3p=E3+E3pkk;
% %     LL0=0;
% %     for i=1:ntr
% %         LL0=mu0{i}'*S^-1*mu0{i}+mu0{i}'*S^-1*C*Inp{i}(:,1)+ ...
% %             Inp{i}(:,1)'*C'*S^-1*mu0{i}-Ez(Lsum(i)+(1:m))'*S^-1*mu0{i}-mu0{i}'*S^-1*Ez(Lsum(i)+(1:m));
% %     end;
% %     LL1=trace(S^-1*E3p)-trace(S^-1*F5*C')-trace(S^-1*C*F5')+trace(S^-1*C*F6*C')-trace(S^-1*A*E2')-trace(A'*S^-1*E2) ...
% %         +trace(A'*S^-1*A*E3)-trace(S^-1*W*E5')-trace(W'*S^-1*E5)+trace(W'*S^-1*W*E1) ...
% %         +trace(A'*S^-1*W*E4)+trace(W'*S^-1*A*E4')+trace(A'*S^-1*C*F3)+trace(S^-1*A*F3'*C') ...
% %         +trace(W'*S^-1*C*F4)+trace(S^-1*W*F4'*C');
% %     LL2=trace(G^-1*F2)+trace(B'*G^-1*B*E1p)-trace(B'*G^-1*F1)-trace(G^-1*B*F1');
% %     ELL=-1/2*(LL0+LL1+LL2+sum(T)*log(det(G))+sum(T)*log(det(S)));
% % end;


%%
% (c) ******



%% NOTE - changes for dropping out **individual elements** xit from LL & param. estim.:
% --- B:
% - zero out i-th rows in F1-comp. for resp. time points tm(i)
% - for E1p, STRICTLY separate E1p terms would need to be computed for
% every i-component that is missing at least once ...
% - OR, more loosely, E1p is computed only once across all t, but is then
% individually adjusted by factor (T-#miss(i))/T for each component i
% - row vectors bi of matrix B then need to be computed separately for each
% i using i-specific E1p matrix
% --- G:
% - for each missing value xit at time point t, the i-th row and column
% needs to be zeroed for all 3 sum terms F2(t), F1(t)*B', B*E1p(t)*B',
% since they do not contribute to the var of xi or cov xi*xj; this implies
% that multiplications with B have to be performed at each time step
% separately
% - at the same time, counter Tij for each different (i,j)-pair has to
% count the number of non-miss occurrences, and G at the end needs to be
% divided by Tij instead of T
% - ALTERNATIVELY, all missing xit could be replaced by their predictions
% bi*phi(zt) (since then the resp. components would just cancel out from
% the LL-sum); G would then just need to be divided by Tij as above
% - this implies two separate runs have to be performed: B needs to be
% computed first as above, then F1, F2, E1p would need to be computed again
% with xit <-- bi*phi(zt); finally G=(...)./Tij
% for this, one would computed B*phi(z1...T) once and replace all NaN's in
% X by their predic.
% - ALTERNATIVELY, if G is diagonal, separate sums F1(i), F2(i), E1p(i)
% could be kept for each component i, and variances gii are then computed
% separately from these individ. sums


%get regularization matrices (per row) for rowwise regularization
%--------------------------------------------------------------------------
function [Reg_EL, Reg_ER]=getORegularization(reg,Mask,S0,m,irow)

Reg_ER=0;
sigma=unique(diag(S0));

%get mask defining elements to be regularized
%LMask is of size LReg=[A W h C] with -1 for pars->1 and 1 for pars->0
LMask=reg.Lreg;

%0th-order regularization for A/W/h->0
Reg0=0;
tau=reg.tau;
N=size(Mask,1);
if (~isempty(tau) && tau~=0)
    
    I=eye(N);
    tmp=LMask(irow,:);
    ind=find(tmp==1); %find indices of parameters to be regularized
    II=eye(N);
    II(ind,ind)=0;
    
    O1=I-II;     %this matrix needs to be applied per i
    Reg0=sigma*tau.*O1;
end

%0th-order regularization for diagonal A with A->1
Reg1=0;
lambda=reg.lambda;
if (~isempty(lambda) && lambda~=0)
    
    I=eye(N);
    tmp=LMask(irow,:);
    ind=find(tmp==-1); %find indices of parameters to be regularized
    II=eye(N);
    II(ind,ind)=0;
    
    O1=I-II;     
    O2=zeros(m,N);
    Aindex=ind(ismember(ind,1:m));
    II=eye(length(Aindex));
    O2(Aindex,Aindex)=II;
    
    %add for other pars if wanted

    Reg1=sigma*lambda.*O1;
    Reg_ER=sigma*lambda*O2;
end
Reg_EL=Reg0+Reg1;




