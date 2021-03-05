function [mu0,B,G,W,S,C,h]=ParEstLDSi1(EV,XZsplit,S0,C0,B0,G0,Lb,Ub)
%
% please cite (& consult for further details): ******
% 
% implements parameter estimation for LDS system
% z_t = W z_t-1 + h + C Inp_t + e_t , e_t ~ N(0,S), W=A+W
% x_t = B z_t + nu_t , nu_t ~ N(0,G)
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
% W: MxM matrix of interaction weights
% S: MxM diagonal covariance matrix (Gaussian process noise)
% C: MxK matrix of regression weights multiplying with Kx1 Inp
% h: Mx1 vector of bias terms
% ELL: expected (complete data) log-likelihood


eps=1e-5;   % minimum variance allowed for in S and G

E2=EV.E2; E3=EV.E3; E3pkk=EV.E3pkk; E3_=EV.E3_;
F1=EV.F1; F2=EV.F2; F3=EV.F3; F5_=EV.F5_; F6_=EV.F6_;
f5_1=EV.f5_1; f6_1=EV.f6_1;
Zt1=EV.Zt1; Zt0=EV.Zt0; InpS=EV.InpS;
T=EV.T;
ntr=EV.ntr;
m=size(E3,1);
Minp=size(InpS,1);

F5=F5_+f5_1;
F6=F6_+f6_1;
E3p=E3+E3pkk;


%% solve for parameters {B,G} of observation model
if nargin>1 && ~isempty(XZsplit)
    Nx=XZsplit(1); Mz=XZsplit(2);
    F1_=F1; F1_(1:Nx,Mz+1:end)=0; F1_(Nx+1:end,1:Mz)=0;
    E3p_=E3p; E3p_(1:Mz,Mz+1:end)=0; E3p_(Mz+1:end,1:Mz)=0;
    if nargin<5 || isempty(B0), B=F1_*E3p_^-1; else B=B0; end
    if nargin<6 || isempty(G0)
        G=diag(max(diag(F2-F1_*B'-B*F1_'+B*E3p_'*B')./sum(T),eps));   % assumes G to be diag
    else G=G0; end
else
    if nargin<5 || isempty(B0), B=F1*E3p^-1; else B=B0; end
    if nargin<6 || isempty(G0)
        G=diag(max(diag(F2-F1*B'-B*F1'+B*E3p'*B')./sum(T),eps));   % assumes G to be diag
    else G=G0; end
end



%% solve for 
% - weight matrix W
% - bias terms h
% - external regressor weights C
% in one go:
if nargin<4 || isempty(C0)
    EL=[E3,Zt1,F3';Zt1',T-ntr,InpS';F3,InpS,F6_];
    ER=[E2,Zt0,F5_];
else
    EL=[E3,Zt1;Zt1',T-ntr];
    ER=[E2-C0*F3,Zt0-C0*InpS];
end    

if nargin<7 || (isempty(Lb) && isempty(Ub))
    WhC=ER*EL^-1;
else
    HH=[];
    for i=1:m, HH=blkdiag(HH,EL'); end
    Y=ER';
    hh=Y(1:end)';
    Lb=Lb'; lb=Lb(1:end)';
    Ub=Ub'; ub=Ub(1:end)';
    whc=quadprog(HH,-hh,[],[],[],[],lb,ub);
    if nargin<4 || isempty(C0), WhC=reshape(whc,m+1+Minp,m)';
    else WhC=reshape(whc,m+1,m)'; end
end

W=WhC(1:m,1:m);
h=WhC(:,m+1);
if nargin<4 || isempty(C0)
    C=WhC(:,m+2:end);
else
    C=C0;
end


%% solve for trial-specific parameters mu0
for i=1:ntr
    mu0{i}=EV.AllIni0(:,i)-C*EV.AllInp(:,i);
end;


%% solve for process noise covariance S, or use provided fixed S0
if nargin>2 && ~isempty(S0), S=S0;
else
    H=zeros(m);
    for i=1:ntr
        H=H+EV.Ezz0{i}-EV.AllIni0(:,i)*mu0{i}'-mu0{i}*EV.AllIni0(:,i)'+mu0{i}*mu0{i}'+mu0{i}*EV.AllInp(:,i)'*C'+C*EV.AllInp(:,i)*mu0{i}';
    end;
    S=diag(diag(H+E3_'-F5*C'-C*F5'+C*F6*C'-E2*W'-W*E2' ...
        +W*E3'*W'+W*F3'*C'+C*F3*W' ...
        -Zt0*h'-h*Zt0'+W*Zt1*h'+h*Zt1'*W'+(T-ntr).*h*h'+C*InpS*h'+h*InpS'*C'))./sum(T);   % assumes S to be diag
end;


%%
% (c) ******
