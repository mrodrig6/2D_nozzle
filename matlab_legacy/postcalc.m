function [thrust,entropyer] = postcalc( resdata, U )
%POSTCALC - This subroutine calculates the thrust coefficient and the
%            entropy error for a specified p,Q,ref provided in resdat and U
% Input:   U         - state vector, stored unrolled
%          resdata   - residual data structure
% Output:  thrust    - vector of residuals, stored unrolled
%          entropyer - time step vector

gamma   = resdata.gamma;    % value of gamma
Minf    = resdata.Minf;     % Mach # at the inlet
p       = resdata.p;        % order of accuracy of U
nbedge  = resdata.nbedge;   % number of boundary edges
w1d     = resdata.w1d;      % quadrature weights in 1d
bnorm   = resdata.bnorm;    % [nbedge x 2 x nqp2d] normals at boundary edges
phiEdge = resdata.phiedgel; % [3 x nbf2d x nqp1d] basis function at edges
B2E     = resdata.B2E;      % [nbedge x 3] bedge to element connectivity
pt      = resdata.pt;       % stagnation pressure at inflow
nqp1d   = length(w1d);      % # of quad points 1d
nbf2d   = (p+1)*(p+2)/2;    % # of 2d basis functions
phi2d   = resdata.phi2d;    % [nbf2d x nqp2d] basis function at ref element
detJ    = resdata.detJ;     % [nelem x nqp2d] determinant of J
w2d     = resdata.w2d;      % quadrature weights in 2d
nqp2d   = length(w2d);      % number of quadrature points in 2d
nelem   = resdata.nelem;    % number of elements
Rgas    = resdata.Rgas;     % Gas constant
Tt      = resdata.Tt;       % Total temperature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Thrust calculation %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 0;
for bedge = 1:nbedge
    % extracting the data from the B2E array
    belem = B2E(bedge,1); edge = B2E(bedge,2);    
    bindex = B2E(bedge,3);     
    % Upper wall included in this section
    if( bindex == -2 )
        idxb  = (belem-1)*nbf2d + [1:nbf2d]; Ub = U(idxb,:);   
        % looping over the quadrature points and calculate p on them
        for qp = 1:nqp1d
            n = bnorm(bedge,:,qp)'; % extracting the bedge normals
            ub = phiEdge(edge,:,qp)*Ub;
            % calculating the conservative variables
            r = ub(1);
            u = ub(2)/r;
            v = ub(3)/r;
            rE= ub(4);        
            pr = (gamma-1)*(rE -.5*r.*(u^2+v^2));   
            pdotn(qp,:) = pr*n;
        end
        T = T - w1d'*pdotn(:,1);
    end    
end
% this is equation is from the isentropic relations to bring a flow from
% mach number, M, isentropically to M* = 1
pst = pt*(2/(gamma+1) + (gamma-1)/(gamma+1)*Minf^2)^(gamma/(gamma-1));
% h is given in the project 3 handout, page 1
h = .13989434;
% this equation in provided in the handout, page 3
tc = T/((gamma/2)*pst*h);
thrust = tc;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Entropy Error %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
st = (pt^(1-gamma))*(Rgas*Tt)^gamma;
eint = 0;   % entropy integral
A    = 0;   % domain area
for elem = 1:nelem   
    % get conservative variables from the U array    
    idx = (elem-1)*nbf2d + [1:nbf2d]; % index for the U array
    Um=U(idx,:);  
    % loop over the 2D quad points to calculate entropy error integral for
    % the element
    for qp = 1:nqp2d  
        % approximate state vector for the element using the basis fncts
        u = Um'*phi2d(:,qp);
        s_r(qp) = entropy_ratio( u, gamma, st );        
    end      
    eint = eint + (s_r-1).^2*(w2d.*detJ(elem,:)');
    A = A + ones(1,nqp2d)*(w2d.*detJ(elem,:)');
end 
% calculates the entropy error and outputs it for the postprocess script
eerror = sqrt(eint/A); 
entropyer = eerror;
end

function s_r = entropy_ratio( U, gam, st )
% this function calculates the entropy ratio s/st for a given state u

% generating the state variables from U and calculating the pressure and
% dennsity to calculate the entropy ratio
r = U(1); u = U(2)/r; v = U(3)/r; rE= U(4);
p = (gam-1)*(rE -.5*r.*(u^2+v^2));
s = p/(r^gam);
s_r=s/st;
end