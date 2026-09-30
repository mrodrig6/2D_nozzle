function mach_profile( resdata, U, ppelem )
% This function calculates and plots the Mach number profile at the exit
% input: ppelem  - resolution, points per element that will be plotted

B2E     = resdata.B2E;      % [nbedge x 3] bedge to element connectivity
V       = resdata.V;        % node coordinates
nbedge  = resdata.nbedge;   % number of boundary edges
p       = resdata.p;        % order of accuracy of U
E2N     = resdata.E2N;
Q       = resdata.Q;
gamma   = resdata.gamma;    % value of gamma

xref = linspace(0,1,ppelem);
k=1;
% iterate over boundaries
for bedge = 1:nbedge
    % extracting the data from the B2E array
    belem = B2E(bedge,1); edge = B2E(bedge,2);    
    bindex = B2E(bedge,3); 
    if( bindex == -3 )
        % get the node coordinates:
        nodes = E2N(belem,:); xglob = V(nodes,:);
        
        % depending on the local edge, find reference points on 2d ref
        % element
        switch edge
            case 1
                % quadrature points along edge 1 in reference element
                xi  = 1 - xref; eta = xref;
            case 2
                % quadrature points along edge 2 in reference element
                xi  = zeros(size(xref)); eta = xref(end:-1:1);
            case 3
                % quadrature points along edge 3 in reference element
                xi  = xref; eta = zeros(size(xref));
        end
        % evaluate the solution basis at the reference points:
        [ phi2d, ~ , ~ , nbf2d ] = TriLagrange2D(p, [xi; eta]' ,ppelem);
        % evaluate the solution at the refernce points
        idxb  = (belem-1)*nbf2d + [1:nbf2d]; Ub = U(idxb,:);   
        uexit = Ub'*phi2d;
        
        % calculate the Mach number
        r = uexit(1,:);
        u = uexit(2,:)./uexit(1,:);
        v = uexit(3,:)./uexit(1,:);
        rE= uexit(4,:);        
        pr = (gamma-1)*(rE -.5*r.*(u.^2+v.^2));
        M = sqrt((u.^2+v.^2)./(gamma*pr./r));
        
        % now map the reference coordinates back to global space
        % therefore: get mesh basis        
        [ phi2dM, ~ , ~ , ~ ] = TriLagrange2D(Q,[xi; eta]' ,ppelem);
        % xexit = phi2dM'*xglob(:,1);
        yexit = phi2dM'*xglob(:,2);
        
        Mprofile(k:k+ppelem-1,:) = [yexit,M'];
        k = k + ppelem;
    end
end
% plot processing code
plot(Mprofile(:,1),Mprofile(:,2));
title('Mach number profile at the exit','Interpreter','Latex','FontSize',14)
ylabel('Mach number','Interpreter','Latex','FontSize',14)
xlabel('y [m]','Interpreter','Latex','FontSize',14)
set(gca,'FontName','Times New Roman','FontSize',10)
