function EV=UpdateExpSums(Ez,V,Ephizi,Ephizij,Eziphizj,X_,Inp_,EV)
%
% please cite (& consult for further details): ******
% 
% implements parameter estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1,0) + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t,0) + nu_t , nu_t ~ N(0,G)
%

if iscell(X_), X=X_; Inp=Inp_; else X{1}=X_; Inp{1}=Inp_; end;
ntr=length(X);
m=size(Ez,1);
N=size(X{1},1);
Minp=size(Inp{1},1);
T=cell2mat(cellfun(@size,X,'UniformOutput',false)'); T=T(:,2);
Tsum=cumsum([0 T']);
Lsum=Tsum.*m;

tm=find(sum(isnan(cell2mat(X))));   % time points with missing data

%% compute E[zz'] from state cov matrix V
Ez=Ez(1:end)';
Ezizi=sparse(m*sum(T),m*sum(T));
for i=1:ntr
    for t=Tsum(i)+1:(Tsum(i+1)-1)
        k0=(t-1)*m+1:t*m;
        k1=t*m+1:(t+1)*m;
        Ezizi(k0,[k0 k1])=V(k0,[k0 k1])+Ez(k0)*Ez([k0 k1])';
        Ezizi(k1,k0)=Ezizi(k0,k1)';
    end;
    Ezizi(k1,k1)=V(k1,k1)+Ez(k1)*Ez(k1)';
end;


%% compute all expectancy sums across trials & time points (eq. 16)

if nargin<8 || isempty(EV)
    E1=zeros(m); E2=E1; E3=E1; E4=E1; E5=E1; E1p=E1; E3pkk=E1; E3_=E1;
    F1=zeros(N,m); F2=zeros(N,N); F3=zeros(Minp,m); F4=F3; F5_=zeros(m,Minp); F6_=zeros(Minp,Minp);
    f5_1=F5_; f6_1=F6_;
    Zt1=zeros(m,1); Zt0=zeros(m,1); phiZ=zeros(m,1); InpS=zeros(Minp,1);
    EV.T=0; EV.ntr=0;
else
    E1=EV.E1; E2=EV.E2; E3=EV.E3; E4=EV.E4; E5=EV.E5; E1p=EV.E1p; E3pkk=EV.E3pkk; E3_=EV.E3_;
    F1=EV.F1; F2=EV.F2; F3=EV.F3; F4=EV.F4; F5_=EV.F5_; F6_=EV.F6_;
    f5_1=EV.f5_1; f6_1=EV.f6_1;
    Zt1=EV.Zt1; Zt0=EV.Zt0; phiZ=EV.phiZ; InpS=EV.InpS;
end

for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    Ephizi0=Ephizi(mt);
    Ezizi0=Ezizi(mt,mt);
    Ephizij0=Ephizij(mt,mt);
    Eziphizj0=Eziphizj(mt,mt);
    
    if ~ismember(Tsum(i)+1,tm)  % drop whole time points with missing data
        F1=F1+X{i}(:,1)*Ephizi0(1:m)';
        F2=F2+X{i}(:,1)*X{i}(:,1)';
    end
    f5_1=f5_1+Ez0(1:m)*Inp{i}(:,1)';
    f6_1=f6_1+Inp{i}(:,1)*Inp{i}(:,1)';
    for t=2:T(i)
        k0=(t-1)*m+1:t*m;   % t
        k1=(t-2)*m+1:(t-1)*m;   % t-1
        E1=E1+Ephizij0(k1,k1);
        E2=E2+Ezizi0(k0,k1);
        E3=E3+Ezizi0(k1,k1);
        E3_=E3_+Ezizi0(k0,k0);
        E4=E4+Eziphizj0(k1,k1)';
        E5=E5+Eziphizj0(k0,k1);
        if ~ismember(Tsum(i)+t,tm)  % drop whole time points with missing data
            F1=F1+X{i}(:,t)*Ephizi0(k0)';
            F2=F2+X{i}(:,t)*X{i}(:,t)';
            E1p=E1p+Ephizij0(k1,k1);
        end
        F3=F3+Inp{i}(:,t)*Ez0(k1)';
        F4=F4+Inp{i}(:,t)*Ephizi0(k1)';
        F5_=F5_+Ez0(k0)*Inp{i}(:,t)';
        F6_=F6_+Inp{i}(:,t)*Inp{i}(:,t)';
    end
    if ~ismember(Tsum(i)+T(i),tm)  % drop whole time points with missing data
        E1p=E1p+Ephizij0(k0,k0);
    end
    if nargout>6, E3pkk=E3pkk+Ezizi0(k0,k0); end;
    
    zz=reshape(Ez0,m,T(i))';
    Zt1=Zt1+sum(zz(1:end-1,:))';
    Zt0=Zt0+sum(zz(2:end,:))';
    pz=reshape(Ephizi0,m,T(i))';
    phiZ=phiZ+sum(pz(1:end-1,:))';
    InpS=InpS+sum(Inp{i}(:,2:end)')';
end


%% transfer back into EV
EV.E1=E1; EV.E2=E2; EV.E3=E3; EV.E4=E4; EV.E5=E5; EV.E1p=E1p; EV.E3pkk=E3pkk; EV.E3_=E3_;
EV.F1=F1; EV.F2=F2; EV.F3=F3; EV.F4=F4; EV.F5_=F5_; EV.F6_=F6_;
EV.f5_1=f5_1; EV.f6_1=f6_1;
EV.Zt1=Zt1; EV.Zt0=Zt0; EV.phiZ=phiZ; EV.InpS=InpS;
EV.T=EV.T+sum(T);
for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    EV.AllIni0(:,EV.ntr+i)=Ez0(1:m);
    Ezz0=Ezizi(mt,mt);
    EV.Ezz0{EV.ntr+i}=Ezz0(1:m,1:m);
    EV.AllInp(:,EV.ntr+i)=Inp{i}(:,1);
end
EV.ntr=EV.ntr+ntr;


%%
% (c) ******