%% a function that computes the block diagonal and d-block-off-diagonal
%% elements of a symmetric block tridiagonal matrix' inverse

% INPUT:
% A: symmetric block tridiagonal matrix with blocks of size m x m
% m: optional, size of block, when m = 1, not block diagonal anymore
% A is input in two block columns of size n x m each; n is the number of blocks
% in A; i.e., full A has a size of (n x m)x(n x m) 
% first block column is for the lower off-diagonal elements; the first block in it is all
% zeros, since the off diagonal is 1 block shorter than diagonal
% second block column is the diagonal elements;
% d: number of off-diagonals to be returned; default (1)
% to return the diagonal only, set d to 0
% OUTPUT:
% Ainv: the block elements at the diagonal and d lower off-diagonal elements
% of A's inverse.


function Ainv = invblocktridiag(A,m,d)

if nargin < 1
  error('not enough input arguments to INVTRIDIAG');
end

% setting defaul values
if nargin < 2
  m = 1;
  d = 1;
elseif nargin < 3
    d = 1;
end
if isempty(m)
    m = 1;
end

nm = size(A,1);
n  = nm/m;
d  = min(n-1,d);
Ainv = zeros(nm,(d+1)*m);

% asymmetric matrix
if size(A,2)/m == 3
    error('asymmetric matrix; not yet implemented');

% symmetric matrix
elseif size(A,2)/m == 2
    cL = zeros(nm,m);
    dL = zeros(nm,m);
    dL(1:m,1:m) = A(1:m,m+1:end);
    for c1 = 1:n-1
        cL(c1*m+1:(c1+1)*m,:) = -A(c1*m+1:(c1+1)*m,1:m)/dL((c1-1)*m+1:c1*m,:);
        dL(c1*m+1:(c1+1)*m,:) = A(c1*m+1:(c1+1)*m,m+1:end) + cL(c1*m+1:(c1+1)*m,:)*A(c1*m+1:(c1+1)*m,1:m)';
    end
    
    dR = A(end-m+1:end,m+1:end);
    Ainv(end-m+1:end,end-m+1:end) = inv(dL(end-m+1:end,:));
    for c1 = n:-1:2
        dR = A((c1-2)*m+1:(c1-1)*m,m+1:end) - A((c1-1)*m+1:c1*m,1:m)'*(dR\A((c1-1)*m+1:c1*m,1:m));
        Ainv((c1-2)*m+1:(c1-1)*m,end-m+1:end) = inv(-A((c1-2)*m+1:(c1-1)*m,m+1:end)+dL((c1-2)*m+1:(c1-1)*m,:)+dR);
    end
    for c2 = 1:d
        for c1 = n:-1:c2+1
            Ainv((c1-1)*m+1:c1*m,end-(c2+1)*m+1:end-c2*m) = ...
                Ainv((c1-1)*m+1:c1*m,end-c2*m+1:end-(c2-1)*m)*cL((c1-c2)*m+1:(c1-c2+1)*m,:);
        end
    end
else
    error('input argument has wrong structure');
end

% (c) 04-01-2017 ******
