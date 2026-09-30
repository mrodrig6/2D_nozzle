function plotmesh(resdata);
% PLOTMESH - Plots a high-order mesh with all edge points
% INPUTS:
%   V       : node coordinates
%   E2N     : element-to-node matrix
%   Q       : geometry order
%   TriFlag : 0 for quadrilaterals; 1 for triangles

V = resdata.V; E2N = resdata.E2N; Q = resdata.Q;  B2E = resdata.B2E;
TriFlag = resdata.triflag;
% determine fvec (loc nodes on each edge of an element)
switch TriFlag
  case 0
    nnode = (Q+1)^2;
    nedge = 4;
    fvec = [1:(Q+1); (Q+1):(Q+1):(Q+1)^2; ...
	    (Q+1)^2:-1:(Q*(Q+1)+1); 1:(Q+1):(Q*(Q+1)+1)];
  case 1
    nnode = (Q+1)*(Q+2)/2;
    nedge = 3;
    fvec = zeros(3,Q+1);
    fvec(1,:) = [1:(Q+1)];
    v = Q+1; d = Q;
    for k=1:(Q+1), fvec(2,k) = v; v = v+d; d = d-1; end;
    v = 1; d = Q+1;
    for k=1:(Q+1), fvec(3,k) = v; v = v+d; d = d-1; end;
  otherwise
    error('element type not understood');
end
% plotting
for elem = 1:size(E2N,1),
  for edge=1:nedge,
      plotedge(V(E2N(elem,fvec(edge,:)),:)); hold on;
  end
end
axis equal
xlabel('x[m]','Interpreter','Latex','Fontsize',16);
ylabel('y[m]','Interpreter','Latex','Fontsize',16);
set(gca,'FontName','Times New Roman','FontSize',18);

%------------------------------------------------
function plotedge(V);
x = V(:,1);
y = V(:,2);
plot(x,y, '.-k', 'LineWidth', 0.5);