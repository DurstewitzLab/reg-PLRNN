function [z,U,d,Err]=StateEstPLRNN3c(A,W,C,S,Inp_,mu0_,B,G,h,X_,z0,d0,tol,eps,flipAll,FinOpt)
%
% *** Same as StateEstPLRNN, except that additional matrix C of weights for
% external regressors (i.e., inputs) is allowed for
%
% please cite (& consult for further details): ******
% 
% implements state estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1,0) + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t,0) + nu_t , nu_t ~ N(0,G)
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
% C: MxK matrix of regression weights multiplying with Inp
% S: MxM diagonal covariance matrix (Gaussian process noise)
% Inp_: KxT matrix of external inputs, or cell array of input matrices 
% mu0_: Mx1 vector of initial values, or cell array of Mx1 vectors
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% h: Mx1 vector of thresholds
% X_: NxT matrix of observations, or cell array of observation matrices
%
% OPTIONAL INPUTS:
% z0: initial guess of state estimates provided as (MxT)x1 vector
% d0: initial guess of constraint settings provided as 1x(MxT) vector 
% tol: acceptable relative tolerance for error increases (default: 1e-2)
% eps: small number added to state covariance matrix for avoiding
%      singularities (default: 0)
% flipAll: flag which determines whether all constraints are flipped at
%          once on each iteration (=true) or whether only the most violating
%          constraint is flipped on each iteration (=false)
%
% OUTPUTS:
% z: estimated state expectations
% U: Hessian
% Err: final total threshold violation error
% d: final set of constraints (ie, for which z>h) 

if nargin<13 || isempty(tol), tol=1e-2; end;
if nargin<14, eps=[]; end;
if nargin<15, flipAll=false; end;

m=length(A);    % # of latent states

if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_;
else X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; end;
ntr=length(X);  % # of distinct trials


%% construct block-banded components of Hessian U0, U1, U2, and 
% vectors/ matrices v0, v1, V2, V3, V4, as specified in the objective 
% function Q(Z), eq. 7, in Durstewitz (2017)
u0=S^-1+A'*S^-1*A; K0=-A'*S^-1;
Ginv=G^-1;
u2A=W'*S^-1*W; u2B=B'*Ginv*B; u2=u2A+u2B;
u1=W'*S^-1*A; K2=-W'*S^-1;
U0=[]; U2=[]; U1=[];
v0=[]; v1=[];
Tsum=0;
for i=1:ntr   % acknowledge temporal breaks between trials
    T=size(X{i},2); Tsum=Tsum+T;
    U0_ = repBlkDiag(u0,T);
    KK0 = repBlkDiag(K0,T);

    X0=X{i};
    tm=find(sum(isnan(X0)));
    
    if isempty(tm)
        U2_ = repBlkDiag(u2,T);
    else
        %U2_ = BlkDiagU2(u2A,u2B,X{i},B,Ginv);  % if just individ. elements xit are to be removed
        U2_ = BlkDiagU2(u2A,u2B,T,tm);  % removes whole time points tm with missing values
    end
    
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
    
    X0(:,tm)=0; % zero out time points with missing values completely;
    % to zero out only individ. components, for each component xit=nan the
    % i-th row of B has to be set to 0, ie corresp. columns of vka need to
    % be computed such that all rows of B and xt corresp. to missing val.
    % are =0.
    vka=B'*G^-1*X0;
    vkb=-W'*S^-1*I(:,2:T);
    v1_=(vka(1:end)+[vkb(1:end) zeros(1,m)])'; v1=[v1;v1_];
end;


%% initialize states and constraint vector
n=1; idx=0; k=[];
if nargin>10 && ~isempty(z0), z=z0(1:end)'; else z=randn(m*Tsum,1); end;
if nargin>11 && ~isempty(d0), d=d0; else d=zeros(1,m*Tsum); d(z>0)=1; end;
Err=1e16;
y=rand(m*Tsum,1); LL=d*y;  % define arbitrary projection vector for detecting already visited solutions 
% alternative: LL=bin2dec(num2str(d)), but doesn't work for large numbers (>63bit)
U=[]; dErr=-1e8;

% % compute initial log-likelihood (if desired)
% LogLike=[];
% d0=zeros(1,m*Tsum); d0(z>0)=1; D0=spdiags(d0',0,m*Tsum,m*Tsum);
% H=D0*U1; U=U0+D0*U2*D0+H+H';
% vv=v0+d0'.*v1;
% LogLike(n)=-1/2*(z'*U*z-z'*vv-vv'*z);


%% mode search iterations
while ~isempty(idx) && isempty(k) && dErr<tol*Err(n)
    % iterate as long as not all constraints are satisfied (idx), the
    % current solution has not already been visited (k), and the change in
    % error (dErr) remains below the tolerance level 
    
    % save last step
    zsv=z; Usv=U; dsv=d;
    if n>1
        if flipAll, dsv(idx)=1-d(idx);
        else dsv(idx(r))=1-d(idx(r)); end;
    end;
    
    % (1) solve for states Z given constraints d
    D=spdiags(d',0,m*Tsum,m*Tsum);
    H=D*U1; U=U0+D*U2*D+H+H';
    if ~isempty(eps), U=U+eps*speye(size(U)); end;  % avoid singularities
    z=U\(v0+d'.*v1);
    
    % (2) flip violated constraint(s)
    idx=find(abs(d-(z>0)'));
    ae=abs(z(idx));
    n=n+1; Err(n)=sum(ae); dErr=Err(n)-Err(n-1);
    if flipAll, d(idx)=1-d(idx);    % flip all constraints at once
    else [~,r]=max(ae); d(idx(r))=1-d(idx(r)); end; % flip constraints only one-by-one, 
            % choosing the one with largest violation in each step

    % terminate when revisiting already visited edges:
    l=d*y; k=find(LL==l); LL=[LL l]; 
    
    %disp(n)
    
%     % track log-likelihood (if desired)
%     d0=zeros(1,m*Tsum); d0(z>0)=1; D0=spdiags(d0',0,m*Tsum,m*Tsum);
%     H=D0*U1; U=U0+D0*U2*D0+H+H';
%     vv=v0+d0'.*v1;
%     LogLike(n)=-1/2*(z'*U*z-z'*vv-vv'*z);
    
end;
%if ~isempty(idx), warning('no exact solution found'); end;

if dErr<tol*Err(n)
    % if idx=[] or k!=[], display final error & change in error
    if flipAll, d(idx)=1-d(idx);
    else d(idx(r))=1-d(idx(r)); end;
    %disp(['#1 - ' num2str(dErr) '   ' num2str(Err(end))])
    Err=Err(2:end);
else
    % if dErr exceeded tolerance, display # of still violated constraints
    z=zsv;
    U=Usv;
    d=dsv;
    Err=Err(2:end-1);
    %disp(['#2 - ' num2str(length(idx)) '   ' num2str(length(k))])
end;


%% perform a final constrained optim step
if nargin>15 && ~isempty(idx) && FinOpt
    sn=d'; sn(sn==1)=-1; sn(sn==0)=1;
    Sn=spdiags(sn,0,m*Tsum,m*Tsum);
    z=quadprog(U,-(v0+d'.*v1),Sn,zeros(m*Tsum,1));
end;

z=reshape(z,m,Tsum);


%%
% (c) ******


function [ BigM ] = repBlkDiag( M, number_rep )
% repeats the Matrix M NUMBER_REP times in the block diagonal

MCell = repmat({M}, 1, number_rep);
BigM = blkdiag(MCell{:});

end


%     % block-diag matrix with missing observations
%     % --- excluding **individual components** {xit}:
%     function U2m=BlkDiagU2(u2A,u2B,X0,B,L)
%         U2m=[];
%         for t=1:size(X0,2)
%             s=find(~isnan(X0(:,t)));
%             if length(s)==size(X0,1), u2x=u2B;
%             elseif length(s)==0, u2x=0;
%             else u2x=B(s,:)'*L(s,s)*B(s,:); end 
%             U2m=blkdiag(U2m,u2A+u2x);
%         end
%     end

    % block-diag matrix with missing observations
    % --- excluding only whole time points in tm
    function U2m=BlkDiagU2(u2A,u2B,T,tm)
        U2m=[];
        for t=1:T
            if ismember(t,tm), U2m=blkdiag(U2m,u2A);
            else U2m=blkdiag(U2m,u2A+u2B); end
        end
    end


end
