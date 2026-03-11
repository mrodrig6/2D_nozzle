function [resdata] = massmatrix( resdata )
% This function calculates the mass matrix
% LISA this function will no longer be necessary I have integrated this
% code into the prealloc subroutine as it will produce a bit more speed up
% calculation wise
detJ    = resdata.detJ;
phi2d   = resdata.phi2d;
w2d     = resdata.w2d;
nelem   = resdata.nelem;
p       = resdata.p;

nbf2d = (p+1)*(p+2)/2;      % # of basis functions

% initialize a sparse mass matrix
M = sparse(nbf2d*nelem,nbf2d*nelem);

for elem = 1:nelem
    W = diag(detJ(elem,:).*w2d');
    Mk = phi2d*W*phi2d'; % [nbf2d x nbf2d]
    idx = (elem-1)*nbf2d + [1:nbf2d];
    M(idx,idx) = Mk;
end

% calculate and store the inverse of the mass matrix
iM = inv(M);
resdata.iM = iM;
resdata.M = M;
end