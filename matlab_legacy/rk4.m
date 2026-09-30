function [U] = rk4(resdata, U)
% RK4  Pseudo-time RK4 integration to steady state.
%
% Applies the block-diagonal inverse mass matrix efficiently:
%   iM is [nbf2d x nbf2d x nelem] — one small block per element.
%   Applied via pagemtimes (R2020b+) or an element loop fallback.
%
% Prints residual every 500 iterations so progress is visible.

iM    = resdata.iM;
nbf2d = size(iM, 1);
nelem = size(iM, 3);

has_pagemtimes = (exist('pagemtimes','builtin') == 5) || ...
                 (exist('pagemtimes','file')    >  0);

res      = 1e3;
epsilon  = 1e-6;
iter     = 0;
print_interval = 500;

fprintf('  Iter       Residual\n');
fprintf('  ----       --------\n');

while res > epsilon
    [R, dt] = residual_calc(U,               resdata);
    F0 = -apply_iM(iM, R, nbf2d, nelem, has_pagemtimes);

    [R, ~]  = residual_calc(U + 0.5*F0.*dt, resdata);
    F1 = -apply_iM(iM, R, nbf2d, nelem, has_pagemtimes);

    [R, ~]  = residual_calc(U + 0.5*F1.*dt, resdata);
    F2 = -apply_iM(iM, R, nbf2d, nelem, has_pagemtimes);

    [R, ~]  = residual_calc(U + F2.*dt,     resdata);
    F3 = -apply_iM(iM, R, nbf2d, nelem, has_pagemtimes);

    res  = norm(F3);
    U    = U + (F0 + 2*F1 + 2*F2 + F3) .* dt / 6;
    iter = iter + 1;

    if mod(iter, print_interval) == 0
        fprintf('  %5d      %.4e\n', iter, res);
    end
end
fprintf('  %5d      %.4e  (converged)\n', iter, res);
end

%% -----------------------------------------------------------------------
function F = apply_iM(iM, R, nbf2d, nelem, has_pagemtimes)
if has_pagemtimes
    R3 = reshape(R,  nbf2d, nelem, 4);
    R3 = permute(R3, [1 3 2]);
    F3 = pagemtimes(iM, R3);
    F3 = permute(F3, [1 3 2]);
    F  = reshape(F3, nelem*nbf2d, 4);
else
    F = zeros(size(R));
    for elem = 1:nelem
        idx      = (elem-1)*nbf2d + (1:nbf2d);
        F(idx,:) = iM(:,:,elem) * R(idx,:);
    end
end
end
