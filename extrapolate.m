function [U] = extrapolate( Uold, resdata )
% EXTRAPOLATE - function approximates the order p solution as order p+1, 
%               want p+1 solution to be exact at the new p+1 quadpoints
% Input: resdata - data structure with a preallocated data
%        Uold    - data array with the state vector solution
% Output:U       - new data array with the extrapolated values for a higher
%                  order basis

% unrolling data from resdata
Q = resdata.Q; pold = resdata.p; nelem = resdata.nelem;
% storing new solution order and the number of basis functions for the
% number of basis functions
pnew = pold + 1;
nbfold = (pold+1)*(pold+2)/2;
nbfnew = (pnew+1)*(pnew+2)/2;
% setting the order for the number of quadrature points according to the
% new order of p
order = 2*pnew + 1 + 2*(Q-1);
% outputting the quadrature points and weights
q2ddata = quad2d(order); nqp2d = q2ddata.nqp2d;
% initializing the new state vector
U = zeros(nelem*nbfnew,4);
% creating basis functions for the new and old solution order for the
% quadrature points according to the new order solution
[phi2dold,~,~,~] = TriLagrange2D(pold,q2ddata.quad2xy,nqp2d);
[phi2dnew,~,~,~] = TriLagrange2D(pnew,q2ddata.quad2xy,nqp2d);
% creating the matrices a and b to solve for the A matrix to extrapolate
% the old state vector
a = phi2dnew*phi2dnew'; b = phi2dnew*phi2dold'; A = a\b;
% looping over the elements
for elem = 1:nelem
    % generating the indicies for the new and old arrays
    idxold = (elem-1)*nbfold + [1:nbfold]; 
    idxnew = (elem-1)*nbfnew + [1:nbfnew];
    % extrapolating the old solution vector for the new solution vector 
    U(idxnew,:) = A*Uold(idxold,:);
end
end