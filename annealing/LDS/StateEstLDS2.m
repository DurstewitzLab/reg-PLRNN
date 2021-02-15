function [z,V]=StateEstLDS2(W0,C,S,Inp_,mu0_,B,G,h,X_,eps)
%
% please cite (& consult for further details): ******
% 
% implements state estimation for LDS
% z_t = A z_t-1 + W z_t-1 + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B z_t + nu_t , nu_t ~ N(0,G)
%
% NOTE: Inp_, mu0_, X_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% A: MxM diagonal matrix 
% W: MxM off-diagonal matrix
% S: MxM diagonal covariance matrix (Gaussian process noise)
% Inp_: MxT matrix of external inputs, or cell array of input matrices 
% mu0_: Mx1 vector of initial values, or cell array of Mx1 vectors
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% h: Mx1 vector of thresholds
% X_: NxT matrix of observations, or cell array of observation matrices
%
% OPTIONAL INPUTS:
% eps: small number added to state covariance matrix for avoiding
%      singularities (default: 0)
%
% OUTPUTS:
% z: estimated state expectations
% V: estimated state covariance matrix

A=diag(diag(W0));
W=W0-A;


if nargin<10, eps=[]; end;

m=length(W);    % # of latent states

if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_;
else X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; end;
ntr=length(X);  % # of distinct trials


%% construct block-banded components of (negative) Hessian 
u0=S^-1+A'*S^-1*A; K0=-A'*S^-1;
u2=W'*S^-1*W+B'*G^-1*B;
u1=W'*S^-1*A; K2=-W'*S^-1;
U0=[]; U2=[]; U1=[];
v0=[]; v1=[];
Tsum=0;
for i=1:ntr   % acknowledge temporal breaks between trials
    T=size(X{i},2); Tsum=Tsum+T;
    U0_ = repBlkDiag(u0,T);
    KK0 = repBlkDiag(K0,T);
    U2_ = repBlkDiag(u2,T);
    U1_ = repBlkDiag(u1,T);
    KK2 = repBlkDiag(K2,T);
    
    KK0=blkdiag(KK0,K0);
    kk=(T-1)*m+1:T*m; U0_(kk,kk)=S^-1;
    U0_=U0_+KK0(m+1:end,1:T*m);
    KK0=KK0'; U0_=U0_+KK0(1:T*m,m+1:end);
    U2_(kk,kk)=B'*G^-1*B;
    U1_(kk,kk)=0; KK2=blkdiag(KK2,K2);
    U1_=U1_+KK2(m+1:end,1:T*m);
    U0=sparse(blkdiag(U0,U0_)); U2=sparse(blkdiag(U2,U2_)); U1=sparse(blkdiag(U1,U1_));
    
    I=C*Inp{i}+repmat(h,1,T);
    vka=S^-1*I; vka(:,1)=vka(:,1)+S^-1*(mu0{i}-h); vkb=A'*S^-1*I(:,2:T);
    v0_=(vka(1:end)-[vkb(1:end) zeros(1,m)])'; v0=[v0;v0_];
    vka=B'*G^-1*X{i}; vkb=-W'*S^-1*I(:,2:T);
    v1_=(vka(1:end)+[vkb(1:end) zeros(1,m)])'; v1=[v1;v1_];
end;


%% solve for states (exact for LDS)
U=U0+U2+U1+U1';
if ~isempty(eps), U=U+eps*speye(size(U)); end;  % avoid singularities
z=U\(v0+v1);
z=reshape(z,m,Tsum);


%% compute block-tridiagonal covariance matrix
eps2=1e-9;
U0=zeros(m*Tsum,2*m);
for t=1:Tsum
    k0=(t-1)*m+1:t*m;
    k1=max(0,(t-2)*m)+1:t*m;
    U0(k0,1:2*m)=[zeros(m,(t<2)*m) U(k0,k1)];
end;
V0=invblocktridiag(U0,m);
v=V0(m+1:end,1:m)';
for t=1:Tsum-1, V0((t-1)*m+1:t*m,2*m+1:3*m)=v(1:m,(t-1)*m+1:t*m); end;
k0=(1:Tsum*m)'*ones(1,3*m);
k1=ones(Tsum*m,1)*(1:3*m);
k1(2*m+1:end,:)=k1(2*m+1:end,:)+reshape(repmat(m:m:(Tsum-2)*m,m,3*m),(Tsum-2)*m,3*m);
k1((Tsum-1)*m+1:Tsum*m,:)=k1((Tsum-2)*m+1:(Tsum-1)*m,:);

V0(1:m,:)=circshift(V0(1:m,:),-m,2);
V0((Tsum-1)*m+1:Tsum*m,:)=circshift(V0((Tsum-1)*m+1:Tsum*m,:),m,2);
V=sparse(k0,k1,V0);

% ensure proper covariance matrix
V=(V+V')./2;    % ensure symmetry
for i=1:m*Tsum, V(i,i)=max(V(i,i),eps2); end;   % limit min var


%%
% (c) ******


function [ BigM ] = repBlkDiag( M, number_rep )
% repeats the Matrix M NUMBER_REP times in the block diagonal

MCell = repmat({M}, 1, number_rep);
BigM = blkdiag(MCell{:});

end

end
