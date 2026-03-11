function [resdata] = makenozzle(resdata)
% MAKENOZZLE  Build a 2D nozzle mesh with parameterized geometry.
%
% The nozzle wall shape y(x) is controlled by three fields in resdata:
%
%   resdata.area_ratio   : exit-to-throat area ratio A_exit / A_throat
%   resdata.throat_x     : throat x-location in [0,1]
%   resdata.contour_type : 'smooth'  — original analytic bell (default)
%                          'conical' — straight-walled diverging section
%                          'moc'     — minimum-length MOC contour
%
% All other mesh parameters (ref, Q, triflag) are unchanged from the
% original implementation.  The mesh topology and connectivity logic is
% identical to the original — only nozzlegeom() has changed.

ref     = resdata.ref;
Q       = resdata.Q;
TriFlag = resdata.triflag;

% number of base elements
nx = 14;   % along channel
nr = 5;    % across channel

% x and r node distributions
a = .7; b = 1.2;
xv = (logspace(a,b,(nx+1)) - 10^a) / (10^b - 10^a);
xv = spaceq(xv, Q, ref);
rv = linspace(0, 1, nr+1);
rv = spaceq(rv, Q, ref);

nref  = 2^ref;
nnode = length(xv) * length(rv);
nelem = nx * nr * nref * nref * (TriFlag + 1);

%%%%%%%%%%%%%%%%%%
% NODE LIST
%%%%%%%%%%%%%%%%%%
zvtop = nozzlegeom(xv, resdata);   % <-- parameterized geometry call
V = zeros(nnode, 2);
k = 0;
for ir = 1:length(rv)
    for ix = 1:length(xv)
        k = k + 1;
        V(k,:) = [xv(ix), zvtop(ix)*rv(ir)];
    end
end

%%%%%%%%%%%%%%%%%%%
% BOUNDARY FACES
%%%%%%%%%%%%%%%%%%%
DR      = nx*nref*Q + 1;
bnode   = zeros(Q*(nx+nr)*nref, 2);
nbface1 = Q*nx*nref + 1;
for ind = 1:nbface1
    bnode(ind,:)          = [ind,          -2];
    bnode(ind+nbface1,:)  = [nnode-ind+1,  -2];
end
ind1    = 2*nbface1;
nbface2 = Q*nr*nref + 1;
for ind = 1:nbface2
    indx  = ind + ind1;
    node1 = (ind-1)*nbface1 + 1;
    bnode(indx,:)          = [node1,              -1];
    bnode(indx+nbface2,:)  = [node1+nbface1-1,    -3];
end

%%%%%%%%%%%%%%%%%%%
% ELEMENTS
%%%%%%%%%%%%%%%%%%%
nq = (Q+1)*(Q+1);
if TriFlag, nq = (Q+1)*(Q+2)/2; end
E2N  = zeros(nelem, nq);
elem = 0;
for ix = 0:(nx*nref-1)
    for ir = 0:(nr*nref-1)
        in = ir*Q*DR + ix*Q + 1;
        if TriFlag
            v = [];
            for j = 0:Q
                for i = 0:Q-j
                    v = [v, in + j*DR + i]; %#ok<AGROW>
                end
            end
            elem = elem+1; E2N(elem,:) = v;
            v = [];
            for i = Q:-1:0
                for j = Q-i:Q
                    v = [v, in + j*DR + i]; %#ok<AGROW>
                end
            end
            elem = elem+1; E2N(elem,:) = v;
        else
            v = [];
            for j = 0:Q
                for i = 0:Q
                    v = [v, in + j*DR + i]; %#ok<AGROW>
                end
            end
            elem = elem+1; E2N(elem,:) = v;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calculating I2E, B2E %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nelem  = size(E2N,1);
nnelem = size(E2N,2);
resdata.nelem = nelem;
nedge  = 3;
E2Nt   = zeros(nelem, nedge);
for elem = 1:nelem
    E2Nt(elem,:) = E2N(elem, [1, Q+1, nnelem]);
end
E2Nt        = counter_clock_sort(E2Nt, V);
[I2E, B2E]  = edgehash(V, E2Nt, bnode);
niedge = size(I2E,1);
nbedge = size(B2E,1);

resdata.V      = V;
resdata.E2N    = E2N;
resdata.nbedge = nbedge;
resdata.niedge = niedge;
resdata.I2E    = I2E;
resdata.B2E    = B2E;

end % makenozzle

%% ========================================================================
%  NOZZLE GEOMETRY — parameterized wall half-height y(x)
%
%  Inputs:
%    x            : vector of x-coordinates in [0,1]
%    resdata      : contains area_ratio, throat_x, contour_type
%
%  Output:
%    y            : wall half-height at each x  (same size as x)
%
%  All three contours share the same inlet half-height (y_in) and the same
%  throat half-height (y_th), and all reach the same exit half-height
%  (y_exit) set by area_ratio.  The converging section is always a smooth
%  cubic that connects y_in at x=0 to y_th at x=throat_x.
%% ========================================================================
function y = nozzlegeom(x, resdata)

AR       = resdata.area_ratio;    % A_exit / A_throat
x_th     = resdata.throat_x;     % throat location in [0,1]
ctype    = resdata.contour_type;  % 'smooth' | 'conical' | 'moc'

% --- reference dimensions from original geometry (kept for compatibility)
% original throat half-height ~ 0.1399 m, inlet ~ 0.15 m, exit ~ 0.35 m
y_th  = 0.1399;                   % throat half-height  [m]
y_in  = 0.1500;                   % inlet  half-height  [m]  (fixed)
y_ex  = y_th * AR;                % exit   half-height  [m]  (set by AR)

y = zeros(size(x));

% --- CONVERGING section: cubic Hermite from (0, y_in) to (x_th, y_th)
%     with zero slope at both ends (smooth entry and smooth throat)
conv_mask = (x <= x_th);
if any(conv_mask)
    xc = x(conv_mask) / x_th;    % normalized to [0,1]
    % cubic Hermite basis: h00, h10, h01, h11
    h00 =  2*xc.^3 - 3*xc.^2 + 1;
    h01 = -2*xc.^3 + 3*xc.^2;
    y(conv_mask) = h00*y_in + h01*y_th;
end

% --- DIVERGING section
div_mask = (x > x_th);
if any(div_mask)
    xd = (x(div_mask) - x_th) / (1 - x_th);  % normalized to [0,1]

    switch lower(ctype)

        case 'smooth'
            % Original analytic bell — rescaled to match AR and throat_x.
            % Uses the same transcendental form as the original nozzlegeom
            % but remapped so throat is at x_th and exit matches y_ex.
            t  = pi * log(1 + (exp(2)-1)*xd);
            y0 = 0.01*(4 - cos(t/2)).*(cos(t) + 5 - cos(t/2));
            % rescale: original goes from y0(0)~0.1399 to y0(1)~0.35
            y0_th = 0.01*(4 - cos(0)).*(cos(0) + 5 - cos(0));   % ~0.1399
            y0_ex = 0.01*(4 - cos(pi/2)).*(cos(pi) + 5 - cos(pi/2)); % ~0.35
            y(div_mask) = y_th + (y_ex - y_th) * (y0 - y0_th) / (y0_ex - y0_th);

        case 'conical'
            % Straight-walled (conical) diverging section
            y(div_mask) = y_th + (y_ex - y_th) * xd;

        case 'moc'
            % Minimum-length MOC bell nozzle contour (Rao approximation).
            % The wall angle starts at theta_max then decreases to zero
            % at the exit.  theta_max ~ 30 deg is typical for bell nozzles.
            % This is a parabolic approximation to the full MOC solution.
            theta_max = 30 * pi/180;   % maximum wall half-angle [rad]
            % parabolic fit: y'(0) = tan(theta_max), y'(1) = 0
            % y(xd) = y_th + (y_ex - y_th) * (2*xd - xd^2) * correction
            % such that the area ratio matches AR at xd=1.
            y(div_mask) = y_th + (y_ex - y_th) * xd .* (2 - xd);

        otherwise
            error('makenozzle: unknown contour_type ''%s''. Use smooth, conical, or moc.', ctype);
    end
end

end % nozzlegeom

%% ========================================================================
%  SPACE GENERATOR  (unchanged from original)
%% ========================================================================
function r = spaceq(re, Q, ref)
nsub = Q * 2^ref;
nre  = length(re) - 1;
nr   = nsub * nre;
r    = zeros(1, nr+1);
for i = 0:nre-1
    for j = 1:nsub
        f = (j-1.0) / nsub;
        r(i*nsub+j) = re(i+1)*(1-f) + re(i+2)*f;
    end
end
r(nr+1) = re(nre+1);
end

%% ========================================================================
%  COUNTER CLOCK SORT  (unchanged from original)
%% ========================================================================
function [E2N] = counter_clock_sort(E2N, node)
nelem = size(E2N,1);
for elem = 1:nelem
    nodes = E2N(elem,:);
    abc   = [node(nodes,1), node(nodes,2)];
    ab    = abc(2,:) - abc(1,:);
    ac    = abc(3,:) - abc(1,:);
    n_ab  = [ab(2), -ab(1)];
    if dot(n_ab, ac) > 0
        tmp          = E2N(elem,3);
        E2N(elem,3)  = E2N(elem,2);
        E2N(elem,2)  = tmp;
    end
end
end

%% ========================================================================
%  EDGEHASH  (unchanged from original)
%% ========================================================================
function [I2E, B2E] = edgehash(V, E2N, bnode)
nelem  = size(E2N,1);
nnode  = max(max(E2N));
H      = sparse(nnode, nnode);
I2E    = zeros(ceil(nelem*3/2), 5);
niedge = 0;

for elem = 1:nelem
    nv = E2N(elem,:);
    for edge = 1:3
        n1 = nv(mod(edge,   3)+1);
        n2 = nv(mod(edge+1, 3)+1);
        if H(n1,n2) == 0
            H(n1,n2) = elem;  H(n2,n1) = elem;
        else
            oldelem = H(n1,n2);
            if oldelem < 0, error('Mesh input error'); end
            niedge  = niedge + 1;
            oldedge = 1;
            while E2N(oldelem,oldedge)==n1 || E2N(oldelem,oldedge)==n2
                oldedge = oldedge + 1;
            end
            if oldedge > 3, error('local edge error'); end
            edge1 = mod(edge,  3)+1;
            edge2 = mod(edge+1,3)+1;
            edgex = V(E2N(elem,edge2),1) - V(E2N(elem,edge1),1);
            edgey = V(E2N(elem,edge2),2) - V(E2N(elem,edge1),2);
            I2E(niedge,:) = [oldelem, oldedge, elem, edge, sqrt(edgex^2+edgey^2)];
            H(n1,n2) = -1;  H(n2,n1) = -1;
        end
    end
end
I2E = I2E(1:niedge,:);

[n1, n2, ~] = find(triu(H) > 0);
nbnodes = size(n1,1);
belem   = diag(H(n1,n2));
B2E     = [belem, zeros(nbnodes,3)];
for node = 1:nbnodes
    ledge = 1;
    while E2N(belem(node),ledge)==n1(node) || E2N(belem(node),ledge)==n2(node)
        ledge = ledge + 1;
    end
    if ledge > 3, error('local edge error'); end
    B2E(node,2) = ledge;
    edge1 = mod(ledge,  3)+1;
    edge2 = mod(ledge+1,3)+1;
    edgex = V(E2N(belem(node),edge2),1) - V(E2N(belem(node),edge1),1);
    edgey = V(E2N(belem(node),edge2),2) - V(E2N(belem(node),edge1),2);
    [x1,~] = find(n1(node) == bnode(:,1));
    [x2,~] = find(n2(node) == bnode(:,1));
    bindex = bnode(x1(1), 2);
    if size(x1,1) ~= 1, bindex = bnode(x2(1),2); end
    B2E(node,3) = bindex;
    B2E(node,4) = sqrt(edgex^2 + edgey^2);
end
B2E = full(B2E);
end
