function [ U, resdata ] = initial( resdata)
% INITIAL - this subroutine initializes the state vector U with free stream
% conditions and calculates the inflow state
%
% Inputs:    phi2d - [basis function][quad point], basis functions at quadpoints
%            nelem - number of elments
% Outputs:   U     - state vector, stored unrolled

phi2d = resdata.phi2d;
nelem = resdata.nelem;

% Initializing the freestream conditions 
M_inf = 0.95; gamma =1.4; 
Tt = 1; pt = 1; Rgas = 0.4;
% inlet stagnation speed of sound
at = sqrt(gamma*Rgas*Tt);
% inlet stagnation density
rhot = gamma*pt/(at^2);

%Ti = Tt/( 1 + (gamma-1)/2*M_inf^2 );
pi = pt/((1+((gamma-1)/2)*M_inf^2)^(gamma/(gamma-1)));
rhoi = rhot/((1+((gamma-1)/2)*M_inf^2)^(1/(gamma-1)));
c = sqrt(gamma*pi/rhoi);

vx_i = c*M_inf; vy_i = 0;
rhoE_i = pi/(gamma-1) + 0.5*rhoi*(vx_i^2 + vy_i^2);

% state vector is constant through out nozzle 
Uinf = [rhoi; rhoi*vx_i; rhoi*vy_i; rhoE_i];  
nqp2d = size(phi2d,2);        % number of quad points
nbf2d = size(phi2d,1);        % number of basis functions
U   = zeros(nelem*nbf2d,4);   % initializing total element state vector
Umj = zeros(nbf2d,4);         % initializing the local element state vector

% solving for the initial state vector at the freestream element at the
% quad points for 2D, solving equation 1 in DG for EU Eqs Handout
for state = 1:4
    ust = ones(nqp2d,1)*Uinf(state);
    Umj(:,state) = phi2d'\ust;
end

% Using Umj vector to store the U vector for all of the elements-> basis
% functions -> number of state variables (4) in this case
for elem = 1:nelem
    for bf = 1:nbf2d
        % obtain the index for the U array
        idx = (elem-1)*nbf2d + bf;
        % store the freestream state
        U(idx,:) = Umj(bf,:);
    end
end

resdata.Minf = M_inf;  % Mach # at inlet
resdata.Uinf = Uinf;   % state at stagnation
resdata.gamma = gamma; % specific heat ratio
resdata.Rgas = Rgas;   % gas constant
resdata.Tt = Tt;       % stagnation temperature at inflow
resdata.pt = pt;       % stagnation pressure at inflow

end
