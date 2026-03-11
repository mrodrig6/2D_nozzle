function [] = postprocess( )
% POSTPROCESS - this subroutine is the post processing script
%%
% this part of the code load the save data, plots the mesh and mach data
% and saves the data according to p, Q, and ref
load p0Q1ref2.mat
% creating the name for the mesh and mach solution 
mesh = sprintf('meshp%iQ%iref%i',p,Q,ref);
usol = sprintf('usolp%iQ%iref%i.eps',p,Q,ref);
%%%% plotting results
figure(1) % plot the mesh uses Q for plotting
plotmesh(resdata);
saveas(gcf,mesh,'eps'); close; % saving the plot and closing it
figure(2) % plot the solution's mach number profile
plotsolution(resdata, U, 'mach', [0 1 0 .35])
saveas(gcf,usol,'eps2c'); close; % saving the plot and closing it
%%
% this part of the code loads the saved data, uses postcalc subroutine to
% calculate the thrust coefficent and the entropy error and stores it
% accordingly
%t = []; e= []; n = [];
load p1Q2ref2.mat
% calculates the number of degrees of freedom
ndof = resdata.nelem*(p+1)*(p+2)/2;
% calculates the thrust coefficient and entropy error
[thrust,entropy] =postcalc( resdata, U );
% these lines of code are commented accordingly for plotting purposes
%tmax = thrust;
n = [n, 1/sqrt(ndof)];
t = [t, abs(thrust-tmax)];
%e = [e entropy];
%%
% these two segments are for the plotting of the error in the thrust
% coefficient and the entropy error
figure(3)
loglog(n,t,'.-k');
ylabel('Error in Thrust Coefficient','Interpreter','Latex','FontSize',14);
xlabel('1/$\surd DOF$','Interpreter','Latex','FontSize',14);
set(gca,'FontName','Times New Roman','FontSize',10); hold on;
%%
% these two segments are for the plotting of the error in the thrust
% coefficient and the entropy error
figure(4)
loglog(n,e,'.-b');
ylabel('Entropy Error','Interpreter','Latex','FontSize',14);
xlabel('1/$\surd DOF$','Interpreter','Latex','FontSize',14);
set(gca,'FontName','Times New Roman','FontSize',10); hold on;
legend('ref = 0','ref = 1','ref = 2','Interpreter','Latex','FontSize',14)
%%
% this segment of code is for plotting the mach profile at the outlet of
% the nozzle
%clear all; close all;
load p2Q2ref2.mat
figure(5) % line-out plot
mach_profile( resdata, U, 10 ); hold on;
%legend('p = 0','p = 1','p = 2','Interpreter','Latex','FontSize',14)
end