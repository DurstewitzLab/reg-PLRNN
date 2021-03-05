function EV=UpdateExpSumsLDS(Ez,V,X_,Inp_,EV)
%
% please cite (& consult for further details): ******
% 
% implements parameter estimation for LDS system
% z_t = W z_t-1 + h + C Inp_t + e_t , e_t ~ N(0,S), W=A+W
% x_t = B z_t + nu_t , nu_t ~ N(0,G)
%


if iscell(X_), X=X_; Inp=Inp_; else X{1}=X_; Inp{1}=Inp_; end;
ntr=length(X);
m=size(Ez,1);
N=size(X{1},1);
Minp=size(Inp{1},1);
T=cell2mat(cellfun(@size,X,'UniformOutput',false)'); T=T(:,2);
Tsum=cumsum([0 T']);
Lsum=Tsum.*m;


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
if nargin<5 || isempty(EV)
    E2=zeros(m); E3=E2; E3pkk=E2; E3_=E2;
    F1=zeros(N,m); F2=zeros(N,N); F3=zeros(Minp,m); F5_=zeros(m,Minp); F6_=zeros(Minp,Minp);
    f5_1=F5_; f6_1=F6_;
    Zt1=zeros(m,1); Zt0=zeros(m,1); InpS=zeros(Minp,1);
    EV.T=0; EV.ntr=0;    
else
    E2=EV.E2; E3=EV.E3; E3pkk=EV.E3pkk; E3_=EV.E3_;
    F1=EV.F1; F2=EV.F2; F3=EV.F3; F5_=EV.F5_; F6_=EV.F6_;
    f5_1=EV.f5_1; f6_1=EV.f6_1;
    Zt1=EV.Zt1; Zt0=EV.Zt0; InpS=EV.InpS;
end


for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    Ezizi0=Ezizi(mt,mt);
    
    F1=F1+X{i}(:,1)*Ez0(1:m)';
    F2=F2+X{i}(:,1)*X{i}(:,1)';
    f5_1=f5_1+Ez0(1:m)*Inp{i}(:,1)';
    f6_1=f6_1+Inp{i}(:,1)*Inp{i}(:,1)';
    for t=2:T(i)
        k0=(t-1)*m+1:t*m;   % t
        k1=(t-2)*m+1:(t-1)*m;   % t-1
        E2=E2+Ezizi0(k0,k1);
        E3=E3+Ezizi0(k1,k1);
        E3_=E3_+Ezizi0(k0,k0);
        F1=F1+X{i}(:,t)*Ez0(k0)';
        F2=F2+X{i}(:,t)*X{i}(:,t)';
        F3=F3+Inp{i}(:,t)*Ez0(k1)';
        F5_=F5_+Ez0(k0)*Inp{i}(:,t)';
        F6_=F6_+Inp{i}(:,t)*Inp{i}(:,t)';
    end;
    E3pkk=E3pkk+Ezizi0(k0,k0);
    
    zz=reshape(Ez0,m,T(i))';
    Zt1=Zt1+sum(zz(1:end-1,:))';
    Zt0=Zt0+sum(zz(2:end,:))';
    InpS=InpS+sum(Inp{i}(:,2:end)')';
end;


%% transfer back into EV
EV.E2=E2; EV.E3=E3; EV.E3pkk=E3pkk; EV.E3_=E3_;
EV.F1=F1; EV.F2=F2; EV.F3=F3; EV.F5_=F5_; EV.F6_=F6_;
EV.f5_1=f5_1; EV.f6_1=f6_1;
EV.Zt1=Zt1; EV.Zt0=Zt0; EV.InpS=InpS;
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
