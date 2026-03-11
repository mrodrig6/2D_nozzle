function [ phi2d, phixi, phieta , nbf2d ] = TriLagrange2D(p, quad2xy,nqp2d);
% TRILANGRANGE2D - calculates basis functions and their gradients at quad 
%                  points for full-order Lagrange basis of order p
%                  reference element is a unit isoceles right triangle
% Input:    p - order of the basis
%           x - points where basis functions and gradients will be evaluated
% Output:   phi2d - basis functions at x, [# basis fcts, # points]
%           phixi - xi gradient at x, [# basis fcts, # points]
%           phieta- eta gradient at x, [# basis fcts, # points]
%           nbf2d     - number of basis functions

% generage the xi and eta spacing for the full-order Lagrange basis
xi  = linspace(0,1,p+1);  eta = xi;
nbf2d = (p+1)*(p+2)/2; % number of basis functions
% initialize A and C to solve for the basis coefficients
A = zeros(nbf2d,nbf2d); C = A;

for j=1:nbf2d  % loop over each basis function and build A-matrix
    i = 1;
    for iy=0:p, 
        for ix=0:p-iy  % loop over nodes
            k = 1;
            for s=0:p, 
                for r=0:p-s % loop over monomials
                    A(i,k) = xi(ix+1)^r * eta(iy+1)^s;
                    k = k+1;
                end
            end
            i = i + 1;
        end
    end;
    D = zeros(nbf2d,1); D(j) = 1; % generating the RHS term
    C(:,j) = A\D; % solve for coefficients
end

%%%%%%%%%%%%%%%%%%%
%%% 2D Basis %%%%%%
%%%%%%%%%%%%%%%%%%%

% evaluate phi2d at the 2d quadrature points
C=C'; phi2d = evalC( C, p, quad2xy, nbf2d, nqp2d );

%%%%%%%%%%%%%%%%%%%
%%% 2D GBasis %%%%%
%%%%%%%%%%%%%%%%%%%

% creating the coeffiecients for the gradients of the full order basis with
% respect to eta an xi
if p == 0
    Ceta = zeros(nbf2d,nqp2d);
    Cxi  = zeros(nbf2d,nqp2d);
else % for p > 0
    for j=1:nbf2d % loop over basis functions
        for s=0:p-1 
            for r=0:p-s-1  % loop over the nodes
                k=kmap(s,r,p-1); % obtaing the k index
                keta=kmap(s+1,r,p); % obtain the keta index
                Ceta(j,k)=C(j,keta)*(s+1); % evaluate and store coefficient
                kxi=kmap(s,r+1,p);         % obtain the kxi index  
                Cxi(j,k)=C(j,kxi)*(r+1);   % evaluate and store coefficient
            end
        end
    end
end
% evaluate phi2dxi and phi2deta at the 2d quadrature points
phixi  = evalC( Cxi,  p-1, quad2xy, nbf2d, nqp2d );
phieta = evalC( Ceta, p-1, quad2xy, nbf2d, nqp2d );
end

function k = kmap(s,r,p)
% this function provides the correct k index for s, r at a certain p for the
% coeffients array C
k=0;
for sd=0:s-1
    k=k+p+1-sd;
end
k=k+r+1; % return the index k
end

function phi = evalC( C, p, quad2xy, nbf2d, nqp2d)
% this function evaluates the basis function or gradient of basis functions
% provided the coeffient array C, the order p, the quadrature points, the
% number of basis functions, and the number of quadrature points

% initalize the array phi
phi=zeros(nbf2d,nqp2d);

for j=1:nbf2d % loop over the basis functions
    for s=0:p 
        for r=0:p-s % loop over the nodes
            k=kmap(s,r,p); % obtain the correct k index using kmap
            % evaluate the basis function at all of the quadrature points
            % and store them along the columns of phi
            phi(j,:)=phi(j,:) + (C(j,k)*quad2xy(:,1).^r.*quad2xy(:,2).^s)'; 
        end
    end
end
% return the phi array
end