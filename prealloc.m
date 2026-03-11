function [resdata] = prealloc(resdata)
% PREALLOC - this subroutine preallocates the basis functions and 
%            derivatives evaluated at the quad points using full order
%            basis
% Input:   resdata - contains all of the data of the mesh and connectivity
% Output:  phi2d   - basis functions at quadpoints
%          phi2dx  - x derivative of basis func at qp mapped to global space
%          phi2dy  - y derivative of basis func at qp mapped to global space
%          detJ    - determinant of jacobian of each element at each quadpoint    
%          phi2dM  - mesh basis functions at quadpoints
%          w2d     - weights for the 2d quad points
%          NOTE: reference element is a unit isoceles right triangle

% Extracting variables from resdata
p = resdata.p; E2N = resdata.E2N; Q = resdata.Q; nelem = resdata.nelem;
V = resdata.V; I2E = resdata.I2E; B2E = resdata.B2E;
niedge = resdata.niedge; nbedge = resdata.nbedge;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% 2D Data Allocation %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
order2d = 2*p + 1 + 2*(Q-1);   % order of accuracy in 2D

% storing the 2D order of accuracy for this code
resdata.order2d = order2d;
% using quad2d to get the quad points and weights into q2ddata structure
q2ddata = quad2d(order2d); nqp2d = q2ddata.nqp2d; 

% basis function and gradients at quad points
[ phi2d, phixi, phieta, nbf2d] = TriLagrange2D(p, q2ddata.quad2xy,nqp2d); 
% curved mesh basis functions and gradients evaluated at quadpoints
[ ~, phixiM, phietaM, ~]       = TriLagrange2D(Q, q2ddata.quad2xy,nqp2d);

% initializing the Jacobian matrix and gradients of the basis functions
Jd = zeros(nelem,nqp2d);
phi2dx = zeros(nelem,nbf2d,nqp2d);
phi2dy = zeros(nelem,nbf2d,nqp2d);
% loop over all of the elements to store the detJ and gradient of the basis
% functions at the quad points
for elem = 1:nelem
    nodes = E2N(elem,:); xglob = V(nodes,:)';
    % loop over quadpoints
    for qp = 1:nqp2d
        gradphi = [phixiM(:,qp) phietaM(:,qp)];
        J = xglob*gradphi;
        detJ=J(1,1)*J(2,2)-J(1,2)*J(2,1);
        Ji=zeros(size(J));
        Ji(1,1)=J(2,2)/detJ; Ji(2,2)=J(1,1)/detJ; 
        Ji(1,2)=-J(1,2)/detJ; Ji(2,1)=-J(2,1)/detJ; 
        Jd(elem,qp)=detJ; % store determinants
        % loop over basis functions to calculate derivatives at quadpoints
        % mapped to global space:
        for bf = 1:nbf2d
            % gradient of basis function j at quadpoint xp at element elem:
            G=[phixi(bf,qp) phieta(bf,qp)]*Ji;
            phi2dx(elem,bf,qp)=G(1);
            phi2dy(elem,bf,qp)=G(2);
        end
    end  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% 1D Data Allocation %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
order1d = 2*p + Q;    % order of accuracy in 1D
% storing the 1D order of accuracy for this code
resdata.order1d = order1d;
% using quad2d to get the quad points and weights into q2ddata structure
q1ddata = quad1d(order1d); nqp1d = q1ddata.nqp1d; w1d = q1ddata.quad1w;
nbf2dQ = (Q+1)*(Q+2)/2;
% looping of the edges and using the 2d basis functions to evaluate phi and
% dphix and dphiy at the edge quadrature points
phiedgel = zeros(3,nbf2d,nqp1d); phiedger = phiedgel;
phiedgex = zeros(3,nbf2dQ,nqp1d); phiedgey = phiedgex;
sigma = q1ddata.quad1x;
for edge = 1:3
    % map edge 1d quadpoints to 2d
    switch edge
        case 1
            % quadrature points along edge 1 in reference element
            xil  = 1 - sigma; etal = sigma;
            xisigl = -1; etasigl = 1;
            xir  = sigma; etar = 1 - sigma;
            %xisigr = 1; etasigr = -1;
        case 2
            % quadrature points along edge 2 in reference element
            xil  = zeros(size(sigma)); etal = 1 - sigma;
            xisigl = 0; etasigl = -1;
            xir  = zeros(size(sigma));  etar = sigma;
            %xisigr = 0;       etasigr = 1;
        case 3
            % quadrature points along edge 3 in reference element
            xil  = sigma; etal = zeros(size(sigma));
            xisigl = 1; etasigl = 0;
            xir  = 1 - sigma; etar = zeros(size(sigma));
            %xisigr = -1;     etasigr = 0;
    end
    % calculate 2d basis functions at 1d quadpoints at order p and gradient
    % of the 2d basis functions at order Q
    [ phi12l,~,~,~ ]      = TriLagrange2D(p,[xil etal],nqp1d); 
    [ phi12r,~,~,~ ]      = TriLagrange2D(p,[xir etar],nqp1d);
    [ ~,phi12x,phi12y,~]  = TriLagrange2D(Q,[xil etal],nqp1d);
    % storing values at phiEdge, phiEdgex, and phiEdgey arrays
    phiedgel(edge,:,:) = phi12l;
    phiedger(edge,:,:) = phi12r;
    phiedgex(edge,:,:) = phi12x*xisigl;
    phiedgey(edge,:,:) = phi12y*etasigl;
end
% calculating the normals, arc lengths at the quadrature points and element perimeters
% by looping over the edges and using the second approach in curved element handout
inorm = zeros(niedge,2,nqp1d); bnorm = zeros(nbedge,2,nqp1d);
ids = zeros(niedge,1); bds = zeros(nbedge,1);
for iedge = 1:niedge
    % extracting the left element local edge and nodes
    edge = I2E(iedge,2); nodes = E2N(I2E(iedge,1),:); xglob = V(nodes,:)'; ds = 0;
    % looping over the quadrature points
    for qp1d = 1:nqp1d
        % calculating and storing normals and arclengths
        t = xglob*(phiedgex(edge,:,qp1d)' + phiedgey(edge,:,qp1d)'); 
        inorm(iedge,:,qp1d) = [t(2) -t(1)];
        ds = ds  + w1d(qp1d)*norm(t);
    end
    ids(iedge,1) = ds;
end

% calculating the normals, arc lengths at the quadrature points and element perimeters
% by looping over the edges and using the second approach in curved element handout
for bedge = 1:nbedge
    % extracting the left element local edge and nodes
    edge = B2E(bedge,2); nodes = E2N(B2E(bedge,1),:); xglob = V(nodes,:)'; ds = 0;
    % looping over the quadrature points
    for qp1d = 1:nqp1d  
        % calculating and storing normals and arclengths
        t = xglob*(phiedgex(edge,:,qp1d)' + phiedgey(edge,:,qp1d)');
        bnorm(bedge,:,qp1d) = [t(2) -t(1)];
        ds = ds + w1d(qp1d)*norm(t);
    end
    bds(bedge,1) = ds;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Mass Matrix Allocation %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w2d = q2ddata.quad2w;
% SPEEDUP: store iM as [nbf2d x nbf2d x nelem] block-diagonal inverse.
% The mass matrix is block-diagonal (each element couples only to itself),
% so its inverse is also block-diagonal. Inverting each small [nbf2d x nbf2d]
% block independently reduces the apply cost from O((nelem*nbf2d)^2) to
% O(nelem*nbf2d^2) — a factor of nelem faster per RK4 stage.
iM = zeros(nbf2d, nbf2d, nelem);
for elem = 1:nelem
    W  = diag(w2d .* Jd(elem,:)');
    Mk = phi2d * W * phi2d';        % [nbf2d x nbf2d] local mass matrix
    iM(:,:,elem) = inv(Mk);         % invert the small local block only
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Resdata Storage Calls  %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% storing output to resdata structure
resdata.phi2d    = phi2d;   % [basis function][quad point]
resdata.phi2dx   = phi2dx;  % [element][basis function][quad point]
resdata.phi2dy   = phi2dy;  % [element][basis function][quad point]
resdata.detJ     = Jd;      % [element][quadpoint]    
resdata.w2d      = w2d;     % [# quad2d pts] 
resdata.w1d      = w1d;     % [# quad1d pts]
resdata.phiedgel = phiedgel;% [edge][basis function 2d][quadpoint 1d] left
resdata.phiedger = phiedger;% [edge][basis function 2d][quadpoint 1d] right
resdata.ids      = ids;     % [edge][1] arclength of the interior edge
resdata.bds      = bds;     % [edge][1] arclength of the bedge
resdata.inorm    = inorm;   % [edge][nx ny][quadpoint] normals in interior
resdata.bnorm    = bnorm;   % [edge][nx ny][quadpoint] normals at boundary
resdata.iM       = iM;      % inverse of the mass matrix
end