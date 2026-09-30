function plotsolution(resdata, U, sname, xyrange)
% function PlotSolution(Mesh, U, sname, srange, xyrange)
%
% PURPOSE: plots a 2D solution
%          note, plot is on gcf, not cleared
%
% INPUTS:
%    Mesh    : mesh structure
%    U       : solution vector structure
%    sname   : name of scalar, see fcns below
%    srange  : range for scalar: [min, max]
%    xyrange : window extent: [xmin, xmax, ymin, ymax]
%
% OUTPUTS:
%    none : mesh is plotted using defaults
 
%--------------------------------------
% parameters
xyrangepad = 0.1; % zoom padding for elem rejection criteria

%------------------------------------
% identify method for computing scalar
StateName = {'Density', 'XMomentum', 'YMomentum', 'Energy'};
smethod = GetScalarMethod(StateName, sname);

%--------------------------------------
% compute plotting coords and triangles
% P = [x';y'] for all points
% T = [n1'; n2'; n3'] ni = ith node over all tris
% S = scalars at all points
P=[]; T=[]; S=[];

nelem  = resdata.nelem;             % # elements
QBasis = 'trilagrange';             % elem geom basis ('trilagrange', 'quadlagrange')
QOrder = resdata.Q;                 % elem geom order, Q
UBasis = 'trilagrange';             % solution basis ('trilagrange', 'quadlagrange')
UOrder = resdata.p;                 % solution order, p
Ur     = (UOrder+1)*(UOrder+2)/2;   % # of unknowns per element
% plot order = max of geometry/approximation orders (can use more here)
POrder = max(QOrder, UOrder);

% pre calculate triangular subdivision
[XYR, NS] = RefSplitElem(QBasis, POrder);
phiQ = EvalBasis(QBasis, QOrder, XYR); % geometry approximation basis
phiU = EvalBasis(UBasis, UOrder, XYR); % state approximation basis

% loop over elements
for elem = 1:nelem,

  % geometry nodes
  igN = resdata.E2N(elem,:);  % nodes for this element
  xyN = resdata.V(igN, :);    % coordinates of these nodes
  
  % skip if not in box
  if (NotInZoom(xyN, xyrange, xyrangepad)) continue; end;
      
  % global coordinates at subdivision nodes
  XYG = phiQ*xyN;
  
  % state at subdivision nodes
  ind = (elem-1)*Ur +[1:Ur];
  u = phiU*U(ind,:);

  % scalar at nodes
  s = ComputeScalar(smethod, u);
  
  % place info in global arrays
  T = [T, NS'+size(P,2)];
  P = [P, XYG'];
  S = [S, s'];

end;
 

%--------------------------------------
% Set colorbar axis, colormap
colormap(jet(2^5));

%--------------------------------------
% plot using pdeplot
T(4,:) = 0;
pdeplot(P, [],T,'xydata',S,'xystyle','interp', 'colormap', 'jet');
% 'contour','on',  'levels', 20
%caxis(srange);
% h = colorbar('East');
% set(h, 'fontsize', 20);
% set(h, 'ytick', linspace(srange(1), srange(end), 5));
% set(gcf,'Renderer','zbuffer')

%-------------------------------------
% set zoom
axis equal;
%axis(xyrange);
% axis off;
% AR = (xyrange(2)-xyrange(1))/(xyrange(4)-xyrange(3));
% DY = 600; % plot size in y (not relevant for width/height of export)
% set(gcf, 'position', [100, 100, DY*AR, DY]);
%set(gca, 'position', [0,0,1,1]);
xlabel('x[m]','Interpreter','Latex','Fontsize',16);
ylabel('y[m]','Interpreter','Latex','Fontsize',16);
set(gca,'FontName','Times New Roman','FontSize',18);


%%%%%%%%%%%%%
% FUNCTIONS %
%%%%%%%%%%%%%

%-------------------------------------------------------
% Computes scalar method: string -> integer for fast access
function smethod = GetScalarMethod(StateName, sname)

StateNameNS = {'Density', 'XMomentum', 'YMomentum', 'Energy'};

% identify equation set based on state names
if (all(strcmp(lower(StateName), lower(StateNameNS)))),
  switch lower(sname)
    case 'density'
      smethod = 1;
    case 'xmomentum'
      smethod = 2;
    case 'ymomentum'
      smethod = 3;
    case 'zmomentum'
      smethod = 4;
    case 'energy'
      smethod = 5;
    case 'mach'
      smethod = 6;
    case 'entropy',
      smethod = 7;
    case 'pressure',
      smethod = 8;
    otherwise
      error 'unsupported Navier-Stokes scalar';
  end
else
  error 'unsupported equation set';
end


%-------------------------------------------------------
% Computes scalar based on method (stored as an integer)
function scalar = ComputeScalar(smethod, u)

sr = size(u,2);

% Navier-Stokes
gam = 1.4; R = 0.4;   % Hardcoded, might need to change these!
gmi = gam-1;
r  = u(:,1);
ru = u(:,2);
rv = u(:,3);
rE = u(:,sr);

% choose appropriate scalar method
switch smethod
    % Navier-Stokes
    case 1 % density
        scalar = r;
    case 2 % x-mom
        scalar = ru;
    case 3 % y-mom
        scalar = rv;
    case 4 % z-mom
        scalar = u(:,4);
    case 5 % energy
        scalar = rE;
    case 6 % mach
        q = sqrt(ru.^2 + rv.^2) ./ r;
        p = gmi * (rE - 0.5*q.*q.*r);
        c = sqrt(gam * p ./ r);
        scalar = q ./ c;
    case 7 % entropy
        q = sqrt(ru.^2 + rv.^2) ./ r;
        p = gmi * (rE - 0.5*q.*q.*r);
        scalar = R/gmi*(log(p) - gam*log(r));
    case 8 % pressure
        q = sqrt(ru.^2 + rv.^2) ./ r;
        scalar = gmi * (rE - 0.5*q.*q.*r);
    otherwise
        error 'unsupported scalar computation method'
end


%-------------------------------------------------------
% checks if all points are inside xyrange
function bool = NotInZoom(xyN, xyrange, xyrangepad)

bool = true;
bool = bool && ( (max(xyN(:,1))<xyrange(1)) || (min(xyN(:,1))>xyrange(2)) );
bool = bool && ( (max(xyN(:,2))<xyrange(3)) || (min(xyN(:,2))>xyrange(4)) );


%-------------------------------------------------------
% subdivides a reference element into triangles
function [XY, N] = RefSplitElem(Basis, n)
% INPUTS:
%   Basis : describes shape of element
%   n     : number of subdvisions in 1D
% OUTPUTS:
%   XY : nnode x 2 vector of node ref coords
%   N  : ntri x 3 node indices for all tris

if (n < 1), n = 1; end;
x = [0:1/n:1];

switch lower(Basis)
  case {'trilagrange'}
    
    nn = (n+1)*(n+2)/2;
    ii = zeros(nn,1); jj = ii;
    kk = zeros(nn,nn);
    XY = zeros(nn,2);
    
    k = 0;
    for i = 1:(n+1),
      for j = 1:(n-i+2),
        k = k + 1;
        ii(k) = i; jj(k) = j; kk(i,j) = k;
        XY(k,1) = x(i); XY(k,2) = x(j);
      end
    end
    
    if (k ~= nn), error 'Error in RefSplitElem'; end;
    nt = n*(n+1)/2 + n*(n-1)/2;
    N = zeros(nt, 3);
    k = 0;
    
    for i = 1:n,
      for j = 1:(n-i+1),
        k = k + 1;
        N(k,:) = [ kk(i,j), kk(i+1,j), kk(i,j+1)];
      end
    end
    
    for i = 1:(n-1),
      for j = 1:(n-i),
        k = k + 1;
        N(k,:) = [ kk(i,j+1), kk(i+1,j), kk(i+1,j+1)];
      end
    end
    
    if (k ~= nt), error 'Error in RefSplitElem'; end;
    
  case 'quadlagrange'
    nn = (n+1)^2;
    XY = zeros(nn,2);
    k = 0;
    for j = 1:(n+1),
      for i = 1:(n+1),
        k = k+1;
        XY(k,1) = x(i); XY(k,2) = x(j);
      end
    end
    nt = 2*n^2;
    N = zeros(nt,3);
    k = 0;
    for j = 1:n,
      for i = 1:n,
        k = k+1;
        ii = (j-1)*(n+1)+i; % lower left corner
        N(k,:) = [ii, ii+1, ii+n+1];
        k = k+1;
        N(k,:) = [ii+1, ii+n+2, ii+n+1];
      end
    end
  otherwise
    error 'unsupported basis in RefSplitElem';
end


%-------------------------------------------------------
% Basis functions 
function phi = EvalBasis(Basis, Order, xref);
% Basis = string identifying the type of basis
% Order = nonnegative integer
% xref = npoint x dim array of ref coords

npoint = size(xref,1);
x = xref(:,1);
y = xref(:,2);

nn = (Order+1)*(Order+2)/2;
phi = zeros(length(x), nn);

switch lower(Basis)
  case 'trilagrange'
    phi = EvalBasis_TriLagrange(Order,x,y);
  case 'quadlagrange'
    phi = EvalBasis_QuadLagrange(Order,x,y);
  otherwise
    error 'Unknown basis'
end



%-------------------------------------------------------
% TriLagrange Basis functions 
function phi = EvalBasis_TriLagrange(order, x, y);

nn = (order+1)*(order+2)/2;
phi = zeros(length(x), nn);

if (order == 0),
  phi(:,0+1) = 1.0;
elseif (order == 1),
  phi(:,0+1) = 1-x-y;
  phi(:,1+1) =   x  ;
  phi(:,2+1) =     y;
elseif (order == 2),
  phi(:,0+1) = 1.0-3.0.*x-3.0.*y+2.0.*x.*x+4.0.*x.*y+2.0.*y.*y;
  phi(:,2+1) = -x+2.0.*x.*x;
  phi(:,5+1) = -y+2.0.*y.*y;
  phi(:,4+1) = 4.0.*x.*y;
  phi(:,3+1) = 4.0.*y-4.0.*x.*y-4.0.*y.*y;
  phi(:,1+1) = 4.0.*x-4.0.*x.*x-4.0.*x.*y;
elseif (order == 3),
  phi(:,0+1) = 1.0-11.0/2.0.*x-11.0/2.0.*y+9.0.*x.*x+18.0.*x.*y+9.0.*y.*y-9.0/2.0.*x.*x.*x-27.0/2.0.*x.*x.*y-27.0/2.0.*x.*y.*y-9.0/2.0.*y.*y.*y;
  phi(:,3+1) = x-9.0/2.0.*x.*x+9.0/2.0.*x.*x.*x;
  phi(:,9+1) = y-9.0/2.0.*y.*y+9.0/2.0.*y.*y.*y;
  phi(:,6+1) = -9.0/2.0.*x.*y+27.0/2.0.*x.*x.*y;
  phi(:,8+1) = -9.0/2.0.*x.*y+27.0/2.0.*x.*y.*y;
  phi(:,7+1) = -9.0/2.0.*y+9.0/2.0.*x.*y+18.0.*y.*y-27.0/2.0.*x.*y.*y-27.0/2.0.*y.*y.*y;
  phi(:,4+1) = 9.0.*y-45.0/2.0.*x.*y-45.0/2.0.*y.*y+27.0/2.0.*x.*x.*y+27.0.*x.*y.*y+27.0/2.0.*y.*y.*y;
  phi(:,1+1) = 9.0.*x-45.0/2.0.*x.*x-45.0/2.0.*x.*y+27.0/2.0.*x.*x.*x+27.0.*x.*x.*y+27.0/2.0.*x.*y.*y;
  phi(:,2+1) = -9.0/2.0.*x+18.0.*x.*x+9.0/2.0.*x.*y-27.0/2.0.*x.*x.*x-27.0/2.0.*x.*x.*y;
  phi(:,5+1) = 27.0.*x.*y-27.0.*x.*x.*y-27.0.*x.*y.*y;
elseif (order == 4),
    phi(:,0+1) = 1.0-25.0/3.0.*x-25.0/3.0.*y+70.0/3.0.*x.*x+140.0/3.0.*x.*y+70.0/3.0.*y.*y-80/3.0.*x.*x.*x-80.0.*x.*x.*y-80.0.*x.*y.*y-80.0...
        /3.0.*y.*y.*y+32.0/3.0.*x.*x.*x.*x+128/3.0.*x.*x.*x.*y+64.0.*x.*x.*y.*y+128.0/3.0.*x.*y.*y.*y+32.0/3.0.*y.*y.*y.*y;
    phi(:,4+1) = -x+22.0./3.0.*x.*x-16.0.*x.*x.*x+32.0/3.0.*x.*x.*x.*x;
    phi(:,14+1) = -y+22.0/3.0.*y.*y-16.0.*y.*y.*y+32.0/3.0.*y.*y.*y.*y;
    phi(:,8+1) = 16.0/3.0.*x.*y-32.0.*x.*x.*y+128.0/3.0.*x.*x.*x.*y;
    phi(:,11+1) = 4.0.*x.*y-16.0.*x.*x.*y-16.0.*x.*y.*y+64.0.*x.*x.*y.*y;
    phi(:,13+1) = 16.0/3.0.*x.*y-32.0.*x.*y.*y+128.0/3.0.*x.*y.*y.*y;
    phi(:,12+1) = 16.0/3.0.*y-16.0/3.0.*x.*y-112.0/3.0.*y.*y+32.0.*x.*y.*y+224.0/3.0.*y.*y.*y-128.0/3.0.*x.*y.*y.*y-128.0/3.0.*y.*y.*y.*y;
    phi(:,9+1) = -12.0.*y+28.0.*x.*y+76.0.*y.*y-16.0.*x.*x.*y-144.0.*x.*y.*y-128.0.*y.*y.*y+64.0.*x.*x.*y.*y+128.0.*x.*y.*y.*y+64.0.*y.*y.*y.*y;
    phi(:,5+1) = 16.0.*y-208.0/3.0.*x.*y-208.0/3.0.*y.*y+96.0.*x.*x.*y+192.0.*x.*y.*y+96.0.*y.*y.*y-128.0/3.0.*x.*x.*x.*y- 128.*x.*x.*y.*y-128.0.*x.*y.*y.*y-128.0/3.0.*y.*y.*y.*y;
    phi(:,1+1) = 16.0.*x-208.0./3.0.*x.*x-208.0/3.0.*x.*y+96.0.*x.*x.*x+192.0.*x.*x.*y+96.0.*x.*y.*y-128.0/3.0.*x.*x.*x.*x- 128.*x.*x.*x.*y-128.0.*x.*x.*y.*y-128.0/3.0.*x.*y.*y.*y;
    phi(:,2+1) = -12.0.*x+76.0.*x.*x+28.0.*x.*y-128.0.*x.*x.*x-144.0.*x.*x.*y-16.0.*x.*y.*y+64.0.*x.*x.*x.*x+128.0.*x.*x.*x.*y+64.0.*x.*x.*y.*y;
    phi(:,3+1) = 16.0/3.0.*x-112.0/3.0.*x.*x-16.0/3.0.*x.*y+224.0/3.0.*x.*x.*x+32.0.*x.*x.*y-128.0/3.0.*x.*x.*x.*x-128.0/3.0.*x.*x.*x.*y;
    phi(:,6+1) = 96.0.*x.*y-224.0.*x.*x.*y-224.0.*x.*y.*y+128.0.*x.*x.*x.*y+256.0.*x.*x.*y.*y+128.0.*x.*y.*y.*y;
    phi(:,7+1) = -32.0.*x.*y+160.0.*x.*x.*y+32.0.*x.*y.*y-128.0.*x.*x.*x.*y-128.0.*x.*x.*y.*y;
    phi(:,10+1) = -32.0.*x.*y+32.0.*x.*x.*y+160.0.*x.*y.*y-128.0.*x.*x.*y.*y-128.0.*x.*y.*y.*y;
else   

  error 'order not supported in TriLagrange'
end


%-------------------------------------------------------
% QuadLagrange Basis functions 
function phi = EvalBasis_QuadLagrange(order, x, y);

n = order+1;
nn = n*n;
phi = zeros(length(x), nn);
xn = linspace(0,1,n); % equal node spacing

for jj=1:nn,
  j1 = mod(jj-1,n)+1;
  j2 = floor((jj-1)/n)+1;
  phi1 = lagrange1D(xn, j1, x)';
  phi2 = lagrange1D(xn, j2, y)';
  phi(:,jj)  = phi1.*phi2;
end


%-------------------------------------------------------
% computes values of jth 1D Lagrange function
function phi = lagrange1D(xn, j, x);
n = length(xn);
if (n == 1)
  phi = ones(size(x)); return;
end
xnj = xn([1:j-1,j+1:n]);
den = prod(xn(j)-xnj);
num = prod(repmat(x,1,n-1) - repmat(xnj,length(x),1), 2);
phi = num/den;




