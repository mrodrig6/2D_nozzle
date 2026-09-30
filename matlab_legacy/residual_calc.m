function [Res, dt] = residual_calc(U, resdata)
% Residual Calc - this function calculates the residuals 
%
% Input:    U       - state vector, stored unrolled
%           resdata - residual data structure
% Output:   Res     - vector of residuals, stored unrolled
%           dt      - time step vector

% open data from resdata for quick access
gamma   = resdata.gamma;    % value of gamma
Rgas    = resdata.Rgas;     % gas constant
phi2dx  = resdata.phi2dx;   % [nelem x nbf2d x nqp2d] gradx basis func at ref elem
phi2dy  = resdata.phi2dy;   % [nelem x nbf2d x nqp2d] grady basis func at ref elem
phi2d   = resdata.phi2d;    % [nbf2d x nqp2d] basis function at ref element
detJ    = resdata.detJ;     % [nelem x nqp2d] determinant of J
w2d     = resdata.w2d;      % quadrature weights in 2d
p       = resdata.p;        % order of accuracy of U
nelem   = resdata.nelem;    % number of elements
niedge  = resdata.niedge;   % number of interior edges
nbedge  = resdata.nbedge;   % number of boundary edges
w1d     = resdata.w1d;      % quadrature weights in 1d
inorm   = resdata.inorm;    % [niedge x 2 x nqp2d] normals at interior edges
ids     = resdata.ids;      % interior edge arc lengths
bnorm   = resdata.bnorm;    % [nbedge x 2 x nqp2d] normals at boundary edges
bds     = resdata.bds;      % boundary edge arc lengths
phiedgel= resdata.phiedgel; % [3 x nbf2d x nqp1d] basis function at left edge
phiedger= resdata.phiedger; % [3 x nbf2d x nqp1d] basis function at right edge
I2E     = resdata.I2E;      % [niedge x 4] iedge to element connectivity
B2E     = resdata.B2E;      % [nbedge x 3] bedge to element connectivity
% uinf    = resdata.Uinf;     % state vector at inflow boundary
Tt      = resdata.Tt;       % stagnation temperature at inflow
pt      = resdata.pt;       % stagnation pressure at inflow
nqp1d   = length(w1d);      % # of quad points 1d
nqp2d   = length(w2d);      % # of quad points 2d
nbf2d   = (p+1)*(p+2)/2;    % # of 2d basis functions
% initialize the residual array and local time stepping
Res = zeros(size(U)); s_i = zeros(nelem,1); 
CFL = 2/(2*(p+1)); 
dt = zeros(size(U));

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Element Integral %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% loop over the elements add the element integral to Res:
for elem = 1:nelem   
    % get conservative variables from the U array    
    idx = (elem-1)*nbf2d + [1:nbf2d]; % index for the U array
    % storing the needed variables and initialzing the local Fx, Fy arrays
    Um=U(idx,:); Fx = zeros(nqp2d,4); Fy = zeros(nqp2d,4); 
    % initalizing the local phix and phiy arrays for matrix allocation
    phix = zeros(nbf2d,nqp2d); phiy = phix; 
    % loop over the 2D quad points to obtain Fx, Fy, phix, phiy
    for qp = 1:nqp2d  
        % approximate state vector for the element using the basis fncts
        u = Um'*phi2d(:,qp);
        [F,G] = eulerflux2D( u,gamma); % calculate flux using eulerflux fct
        % store the fluxes in Fx and Fy = [nqp2d x 4]
        Fx(qp,:) = F;  Fy(qp,:) = G;
        phix(:,qp) = phi2dx(elem,:,qp)'; % phix and phiy = [nbf2d x nqp2d]
        phiy(:,qp) = phi2dy(elem,:,qp)';
    end        
    % Now use matrix multiplication to simply compute the residual for the
    % element and its basis functions using a simple storage system
    Res(idx,1:4) = Res(idx,1:4) - (phix*diag(w2d.*detJ(elem,:)')*Fx +...
                                   phiy*diag(w2d.*detJ(elem,:)')*Fy);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Boundary Integrals %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_wspeed = 0; Flux = zeros(1,4); Fhat = zeros(nqp1d,4);
% loop over interior edges and add boundary integral to Res
for iedge = 1:niedge
    % extracting the data from the I2E array
    lelem  = I2E(iedge,1); ledge = I2E(iedge,2); s_edge = ids(iedge);
    relem  = I2E(iedge,3); redge = I2E(iedge,4);
    % using the appropriate indexing to obtain the U data for left and
    % right elements and then extracting from U
    idxl  = (lelem-1)*nbf2d + [1:nbf2d]; 
    idxr  = (relem-1)*nbf2d + [1:nbf2d];
    Ul = U(idxl,:); Ur = U(idxr,:); 
    % looping over the quadrature points to generate Fhat
    for qp = 1:nqp1d
        n = inorm(iedge,:,qp)'; % extracting iedge normals 
        ds = norm(n); n = n./ds;
        % calculating the left and right state at the edge quadrature point
        ul = phiedgel(ledge,:,qp)*Ul; ur = phiedger(redge,:,qp)*Ur;
        [Flux,max_wspeed] = roe_flux(ul,ur,n(1),n(2),gamma);
        Fhat(qp,:) = Flux*ds; % Roe solver Fhat = [nqp1d x 4]
    end
    % adding the boundary flux to the right and element locations
    squeezephil = zeros(nbf2d,nqp1d); squeezephir = zeros(nbf2d,nqp1d);
    for nbf = 1:nbf2d
        for nq = 1:nqp1d
            squeezephil(nbf,nq) = phiedgel(ledge,nbf,nq);
            squeezephir(nbf,nq) = phiedger(redge,nbf,nq);
        end
    end
    Res(idxl,1:4) = Res(idxl,1:4) + squeezephil*diag(w1d)*Fhat;
    Res(idxr,1:4) = Res(idxr,1:4) - squeezephir*diag(w1d)*Fhat;
    % calculating the average max wave speed for left and right elements
    s_i(lelem) = s_i(lelem) + max_wspeed*s_edge;
    s_i(relem) = s_i(relem) + max_wspeed*s_edge;
end

% %%% free stream testing code
% for bedge = 1:nbedge
%     % extracting the data from the B2E array
%     belem = B2E(bedge,1); edge = B2E(bedge,2); s_edge = bds(bedge);
%     % using the appropriate indexing to obtain the U data for boundary 
%     % elements and then extracting from U
%     idxb  = (belem-1)*nbf2d + [1:nbf2d]; Ub = U(idxb,:);   
%     % looping over the quadrature points to generate Fhat
%     for qp = 1:nqp1d
%         n = bnorm(bedge,:,qp)'; % extracting the bedge normals
%         uinf = uinf'; ub = phiedgel(edge,:,qp)*Ub;
%         [Flux,max_wspeed] = roe_flux(ub,uinf,n(1),n(2),gamma);
%         Fhat(qp,:) = Flux; % Roe solver Fhat = [nqp1d x 4]
%     end
%     % adding the boundary flux to the right and element locations
%      Res(idxb,1:4)=Res(idxb,1:4) + squeeze(phiedgel(edge,:,:))'*diag(w1d)*Fhat;
%     s_i(belem) = s_i(belem) + max_wspeed*s_edge;
% end

%%%%% actual bc code implementation
for bedge = 1:nbedge
    bindex = B2E(bedge,3); 
    % extracting the data from the B2E array
    belem = B2E(bedge,1); edge = B2E(bedge,2); s_edge = bds(bedge);
    % using the appropriate indexing to obtain the U data for boundary 
    % elements and then extracting from U
    idxb  = (belem-1)*nbf2d + [1:nbf2d]; Ub = U(idxb,:); Fhat = zeros(nqp1d,4);
    % looping over the quadrature points to generate Fhat
    for qp = 1:nqp1d
        n = bnorm(bedge,:,qp)'; % extracting the bedge normals
        ub = Ub'*phiedgel(edge,:,qp)';
        ds = norm(n); n = n./ds;
        if(bindex == -1) % inflow case, % define UL, Tt, pt, Rgas in initial
            %[Flux,max_wspeed] = roe_flux(ub,ub,n(1),n(2),gamma);
            [Flux,max_wspeed] = inflow(ub, Tt, pt, 0, n, Rgas, gamma);  
        elseif(bindex == -2) % inviscid wall & symmetry
            [Flux,max_wspeed] = inviscid_wall_bc(ub,n(1),n(2),gamma);   
        elseif(bindex == -3)    % outflow
            %[Flux,max_wspeed] = roe_flux(ub,ub,n(1),n(2),gamma);
            [F, G] = eulerflux2D(ub,gamma);
            Flux = F*n(1) +G*n(2);
        end
        Fhat(qp,:) = Flux*ds; % Roe solver Fhat = [nqp1d x 4]
    end
    % adding the boundary flux to the right and element locations
    squeezephil = zeros(nbf2d,nqp1d); 
    for nbf = 1:nbf2d
        for nq = 1:nqp1d
            squeezephil(nbf,nq) = phiedgel(edge,nbf,nq);
        end
    end
    Res(idxb,1:4) = Res(idxb,1:4) + squeezephil*diag(w1d)*Fhat;
    s_i(belem) = s_i(belem) + max_wspeed*s_edge;
end
% the following code is used to calculate the time step according to the
% formulation in the 2D Euler Equations course handout, here detJ is 2*A
dt_1 = max(detJ,[],2)*CFL./s_i; 
% looping over all of the elements to calculate the time step
for elem = 1:nelem
    dt_elem = dt_1(elem)*ones(nbf2d,4);
    idx = (elem-1)*nbf2d + [1:nbf2d];
    dt(idx,:) = dt_elem;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Euler Flux 2D %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
function [F,G] = eulerflux2D( U, gamma )
% This function calculates the flux for the 2D Euler equations
%
% input :   u   - State vector
% output:   F   - Euler flux aligned with x
%           G   - Euler flux aligned with y
rho=U(1); u=U(2)/U(1); v=U(3)/U(1); E=U(4)/U(1); 
p=(gamma-1)*(rho*E-.5*rho*(v^2+u^2)); H=E+p/rho;
F=[rho.*u, rho.*u.^2+p, rho.*u.*v, rho.*u.*H];
G=[rho.*v, rho.*v.*u, rho.*v.^2+p, rho.*v.*H];
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Roe Flux 2D  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
function [Flux,max_wspeed] = roe_flux(Ul,Ur,nx,ny,gamma)
% ROE_FLUX subroutine to calculate the F_L, F_R, RHS, and
% maximum wave speed for each edge
%   INPUT : Ul    = [4 x 1] array with left element variables
%           Ur    = [4 x 1] array with right element variables
%           nx    = [1 x 1] x-normal of the edge
%           ny    = [1 x 1] y-normal of the edge
%           gamma = 1.4
%  OUTPUT : Flux   = [4 x 1] array containing the Roe flux
%           max_wspeed = [1 x 1] maximum wave speed from the eigenvalues

% calculating the coordinate free left element flux
pl = (gamma-1)*(Ul(4)-0.5*(Ul(2)^2+Ul(3)^2)/Ul(1)); Hl = (Ul(4)+pl)/Ul(1);
Fli = [Ul(2), (Ul(2)^2)/Ul(1) + pl, Ul(2)*Ul(3)/Ul(1), Ul(2)*Hl];
Flj = [Ul(3), Ul(2)*Ul(3)/Ul(1), (Ul(3)^2)/Ul(1) + pl, Ul(3)*Hl];
F_L = Fli*nx+Flj*ny;

% calculating the coordinate free right element flux
pr = (gamma-1)*(Ur(4)-0.5*(Ur(2)^2+Ur(3)^2)/Ur(1)); Hr = (Ur(4)+pr)/Ur(1);
Fri = [Ur(2), (Ur(2)^2)/Ur(1) + pr, Ur(2)*Ur(3)/Ur(1), Ur(2)*Hr];
Frj = [Ur(3), Ur(2)*Ur(3)/Ur(1), (Ur(3)^2)/Ur(1) + pr, Ur(3)*Hr];
F_R = Fri*nx+Frj*ny;

% calculating the Roe-averaged state
rhol = sqrt(Ul(1)); rhor = sqrt(Ur(1)); rhosum = rhol+rhor;
u = (rhol*Ul(2)/Ul(1) + rhor*Ur(2)/Ur(1))/rhosum; 
v = (rhol*Ul(3)/Ul(1) + rhor*Ur(3)/Ur(1))/rhosum;
q2 = u^2 + v^2; H = (rhol*Hl + rhor*Hr)/rhosum;

% safety check to prevent code crashing
radicand = (gamma-1)*(H-0.5*q2);
if (sign(radicand) == -1)
    error 'negative roe-averaged speed of sound'
end

% Roe averaged speed of sound
c = sqrt(radicand);

% calculating the wavespeeds and performing the entropy fix
epsilon = 0.05*c; un = u*nx + v*ny; eig = abs([un+c,un-c,un,un]);
max_wspeed = max(eig);
for i = 1:4
    if (eig(i)<epsilon)
        eig(i) = (eig(i)^2 + epsilon^2)/(2*epsilon);
    end
end

% calculating the values for the RHS as seen in page 4 of hand out 2
delr = Ur(1) - Ul(1); delru = Ur(2)-Ul(2) ; 
delrv = Ur(3)-Ul(3); delrE = Ur(4)-Ul(4);
G1 = (gamma-1)*(0.5*q2*delr - u*delru - v*delrv + delrE);
G2 = -un*delr + delru*nx + delrv*ny;
s1 = 0.5*(eig(1) + eig(2)); s2 = 0.5*(eig(1) - eig(2)); 
C1 = (G1/c^2)*(s1-eig(3)) + (G2/c)*s2; C2 = G1*s2/c + (s1-eig(3))*G2;
% compiling results into the RHS see page 3 of handout 2
RHS = [eig(3)*delr + C1;
       eig(3)*delru + C1*u + C2*nx;
       eig(3)*delrv + C1*v + C2*ny;
       eig(3)*delrE + C1*H + C2*un];
% calculates the total flux using the Roe method
Flux = 0.5*(F_L + F_R) - 0.5*RHS';
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Inviscid Wall %%%%
%%%%%%%%%%%%%%%%%%%%%%%%
function [Flux,max_wspeed] = inviscid_wall_bc(U_plus,nx,ny,gamma)
% INVISCID_WALL_BC subroutine to calculate the wall boundary condition
%   INPUT : U_plus= [4 x 1] array with boundary element variables
%           nx    = [1 x 1] x-normal of the edge
%           ny    = [1 x 1] y-normal of the edge
%           gamma = 1.4
%  OUTPUT : Flux  = [4 x 1] array with the boundary flux
%           max_wspeed = [1 x 1] array with the maximum wave speed

% extracting the necessary conservative variables
rho_E = U_plus(4); rho_plus = U_plus(1); 

% calculating the velocity tangent to the boundary edge
v_plus = [U_plus(2);U_plus(3)]./rho_plus; n = [nx;ny];  
v_parallel = v_plus - dot(v_plus,n)*n;

% using invisid formulation in page 5 of handout 2 for the boundary
% pressure
p_b = (gamma-1)*(rho_E - 0.5*rho_plus*(v_parallel(1)^2+v_parallel(2)^2));
max_wspeed = sqrt(gamma*p_b/rho_plus);

% evaluating boundary flux
Flux = [0, p_b*nx, p_b*ny, 0];
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Inflow BC     %%%%
%%%%%%%%%%%%%%%%%%%%%%%%
function [H, smax] = inflow(UL, Tt, pt, alpha, nvec, Rgas, gamma)
% [H, smax] = inflow(UL, Tt, pt, alpha, nvec, gamma)
%
% This function calculates the flux for the Euler 
% equations at the inlet (assumed to be subsonic).
%
% The inputs are:
%
%       UL: state vector in left cell (in the domain)
%       Tt: total/stagnation temperature
%       pt: total/stagnation pressure
%    alpha: inlet flow angle
%     nvec: normal pointing from the left cell out of the domain
%     Rgas: gas constant
%    gamma: ratio of specific heats
%
% The outputs are:
%     H: the flux out of the left cell
%  smax: the maximum propagation speed of disturbance
%

% a0inl = inlet stagnation speed of sound
a0inl = sqrt(gamma*Rgas*Tt);
% r0inl = inlet stagnation density
r0inl = gamma*pt/(a0inl^2);
% Determine outgoing Riemann invariant from interior (UL)
n = nvec/norm(nvec);
rL  = UL(1);
uL  = UL(2)/rL;
vL  = UL(3)/rL;
unL = uL*n(1) + vL*n(2);
pL  = (gamma-1)*(UL(4) - 0.5*rL*(uL^2 + vL^2));
aL  = sqrt(gamma*pL/rL);

Jp  = unL + 2/(gamma-1)*aL;
% Solve quadratic equation for Mach number
beta = (Jp/a0inl)^2;
nfac = n(1)*cos(alpha) + n(2)*sin(alpha);

aa = 0.5*(gamma-1)*beta - nfac^2;
bb = -4*nfac/(gamma-1);
cc = beta - (2/(gamma-1))^2;

Mfac = sqrt(bb^2 - 4*aa*cc);
MR = (-bb + Mfac)/(2*aa);

% Calculate rest of inflow state
aR = sqrt(a0inl^2/(1 + 0.5*(gamma-1)*MR^2));
qR = MR*aR;
uR = qR*cos(alpha);
vR = qR*sin(alpha);
rR = r0inl*(1 + 0.5*(gamma-1)*MR^2)^(-1/(gamma-1));
pR = (rR*aR^2)/gamma;
HR = (a0inl^2)/(gamma-1);

unR = uR*n(1) + vR*n(2);

H = zeros(1,4);
H(1) = rR*unR;
H(2) = rR*uR*unR + pR*n(1);
H(3) = rR*vR*unR + pR*n(2);
H(4) = rR*HR*unR;
smax = abs(unR + aR);
end