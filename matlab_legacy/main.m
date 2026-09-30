function [] = main()
% MAIN  Driver for the 2D Euler DG nozzle solver.
%
% WORKFLOW
%   The solver uses p-continuation to reach a resolved solution efficiently:
%     Step 1 — Solve on coarsest representation (p=0, ref=0)
%     Step 2 — Extrapolate to p=1, solve to convergence
%     Step 3 — Extrapolate to p=2, solve to convergence  (optional)
%   Each step warm-starts from the previous solution, dramatically reducing
%   the number of iterations needed at higher order.
%
% GEOMETRY PARAMETERS (edit these to design your nozzle)
%   area_ratio     : exit-to-throat area ratio  (controls design Mach number)
%   throat_x       : throat x-location in [0,1] (0=inlet, 1=exit)
%   contour_type   : 'smooth' (default analytic) | 'conical' | 'moc'
%
% FLOW PARAMETERS
%   p_back_ratio   : back pressure / stagnation pressure  (controls shock location)
%                    1.0  = no flow
%                    ~0.5 = choked subsonic
%                    design value depends on area_ratio (see isentropic tables)
%
% MESH PARAMETERS
%   ref_max        : maximum refinement level (0=coarse, 1=medium, 2=fine)
%                    ref=0: ~140 elements, ~30s
%                    ref=1: ~560 elements, several minutes
%                    ref=2: ~2240 elements, use MEX binary
%   p_max          : maximum DG polynomial order (0, 1, or 2)
%   Q              : geometry order (1=linear elements, 2=curved)

%==========================================================================
%  USER INPUTS — edit this block only
%==========================================================================
area_ratio   = 2.50;      % exit/throat area ratio  (2.50 -> M_design ~ 2.44)
throat_x     = 0.14;      % throat location along nozzle (normalized 0-1)
contour_type = 'smooth';  % 'smooth' | 'conical' | 'moc'

p_back_ratio = 0.15;      % back pressure ratio (pt_back / pt_inlet)

ref_max      = 1;         % max refinement level (0, 1, or 2)
p_max        = 1;         % max polynomial order (0, 1, or 2)
Q            = 1;         % geometry order (keep at 1 unless using curved mesh)
triflag      = 1;         % 1 = triangles, 0 = quads (keep at 1)
%==========================================================================

% pack geometry and flow config into resdata
resdata.area_ratio   = area_ratio;
resdata.throat_x     = throat_x;
resdata.contour_type = contour_type;
resdata.p_back_ratio = p_back_ratio;
resdata.Q            = Q;
resdata.triflag      = triflag;

fprintf('\n=== DG Nozzle Solver ===\n');
fprintf('Geometry  : %s  |  Area ratio = %.3f  |  Throat at x = %.3f\n', ...
        contour_type, area_ratio, throat_x);
fprintf('Target    : ref=%d, p=%d\n\n', ref_max, p_max);

%--------------------------------------------------------------------------
% STEP 1: solve at p=0, ref=0 (always the starting point)
%--------------------------------------------------------------------------
fprintf('--- Step 1: p=0, ref=0 (initial coarse solution) ---\n');
resdata.ref = 0;
resdata.p   = 0;
resdata     = makenozzle(resdata);
resdata     = prealloc(resdata);
[U, resdata] = initial(resdata);

tic;
U = rk4(resdata, U);
t = toc;
fprintf('  Converged in %.1f s\n', t);

save('nozzle_p0_ref0.mat', 'U', 'resdata');

if p_max == 0 && ref_max == 0
    postprocess(resdata, U);
    return;
end

%--------------------------------------------------------------------------
% STEP 2: extrapolate to p=1, solve at ref=0 then ref=ref_max
%--------------------------------------------------------------------------
if p_max >= 1
    for ref = 0:ref_max
        fprintf('\n--- Step 2: p=1, ref=%d ---\n', ref);

        % rebuild mesh at this refinement
        resdata_new      = resdata;
        resdata_new.ref  = ref;
        resdata_new.p    = 1;
        resdata_new      = makenozzle(resdata_new);
        resdata_new      = prealloc(resdata_new);
        [~, resdata_new] = initial(resdata_new);

        % warm start: extrapolate from previous solution
        U = extrapolate(U, resdata_new);
        resdata = resdata_new;

        tic;
        U = rk4(resdata, U);
        t = toc;
        fprintf('  Converged in %.1f s\n', t);

        fname = sprintf('nozzle_p1_ref%d.mat', ref);
        save(fname, 'U', 'resdata');
    end
end

%--------------------------------------------------------------------------
% STEP 3: extrapolate to p=2, solve at ref=ref_max (optional)
%--------------------------------------------------------------------------
if p_max >= 2
    fprintf('\n--- Step 3: p=2, ref=%d ---\n', ref_max);

    resdata_new      = resdata;
    resdata_new.p    = 2;
    resdata_new      = makenozzle(resdata_new);
    resdata_new      = prealloc(resdata_new);
    [~, resdata_new] = initial(resdata_new);

    U       = extrapolate(U, resdata_new);
    resdata = resdata_new;

    tic;
    U = rk4(resdata, U);
    t = toc;
    fprintf('  Converged in %.1f s\n', t);

    fname = sprintf('nozzle_p2_ref%d.mat', ref_max);
    save(fname, 'U', 'resdata');
end

%--------------------------------------------------------------------------
% POST-PROCESSING
%--------------------------------------------------------------------------
fprintf('\n=== Solution complete — running post-processing ===\n');
postprocess(resdata, U);

end
