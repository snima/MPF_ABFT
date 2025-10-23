% MPF_ABFT.m
% Extended bit-level analysis: FP16 + FP64 precision comparison
% COMPLETE VERSION with all configurations and sweeps

clear; clc; close all;

% ---- Quiet mode + safety ----
S_WARN = warning; warning('off','all');
cleanupWarn = onCleanup(@() warning(S_WARN));
SAFE_MAX_N = 4096;

format compact;
ticTotal = tic;

% ========================= EASY CONTROL BLOCK ============================
PROFILE     = 'large';    % 'tiny', 'small', 'medium', 'large'
SEED_MULT   = 1.0;         % Seed multiplier (0.5 = half seeds for speed)
SIZE_PRESET = 'multi';    % 'single' = use PROFILE base size only, 'multi' = sweep sizes
BASE_PINJECT= 1.0;

% Headless figures
set(groot,'defaultFigureVisible','off');
  
% Profile presets
switch lower(PROFILE)
    case 'tiny'
        m = 64; n = 64; wb = 8;
    case 'small'
        m = 128; n = 128; wb = 32;
    case 'medium'
        m = 256; n = 256; wb = 64;
    case 'large'
        m = 512; n = 512; wb = 64;
    otherwise
        error('Unknown PROFILE: %s', PROFILE);
end

assert(mod(n,wb)==0, 'wb must divide n.');

useGPU   = false;
pinject  = BASE_PINJECT;
verbosePanels = false;
make3DPerRun  = false;

% Section toggles
doPropagationFPR     = true;
doPinjectSweep       = true;
doSizeSweep          = true;
doOverheadGrid       = false;  % Disabled for speed
doBitPerIndexSweep   = true;
doBitClassSweep      = true;
doFP64BitSweeps      = true;   % NEW: FP64 extended analysis

% Optional grids (disabled by default)
doSizeWbGrid         = false;
doRepSizeSweep       = false;
doRepSizeWbGrid      = false;
doKappaSweep         = false;
doProductionAnalysis = false;

% Size presets: align with PROFILE base size
switch lower(SIZE_PRESET)
    case 'single'
        % All tests use ONE consistent size from PROFILE
        sizesVec = m;
    case 'multi'
        % Sweep multiple sizes (only for dedicated size studies)
        switch lower(PROFILE)
            case 'tiny'
                sizesVec = [32 64];
            case 'small'
                sizesVec = [64 128];
            case 'medium'
                sizesVec = [128 256];
            case 'large'
                sizesVec = [256 512];
            otherwise
                sizesVec = [256 512];
        end
    otherwise
        sizesVec = m;  % Default to base size
end

sizesVecGrid    = 512;
wbVecGrid       = 64;

% Safety check
maxNRequested = max([m, n, sizesVec(:).', sizesVecGrid(:).']);
if maxNRequested > SAFE_MAX_N
    error('SAFE_MAX_N exceeded: requested n=%d > %d.', maxNRequested, SAFE_MAX_N);
end

% Memory check (Windows)
if ispc
    try
        [usr, sys] = memory;
        bytesPerMatrix = 8 * double(maxNRequested) * double(maxNRequested);
        needBytes = 4 * bytesPerMatrix;
        if needBytes > sys.PhysicalMemory.Available
            error('Estimated memory need (%.1f GB) exceeds available (%.1f GB).', ...
                  needBytes/2^30, sys.PhysicalMemory.Available/2^30);
        end
    catch
    end
end

% =================== ENHANCED BASE COMPARISON SETTINGS ===================
nSeedsBaseComp    = max(1, ceil(10 * SEED_MULT));
baseSizesVec      = [];
baseCompSeedStart = 8000;

% Seed configurations
pinjectVec       = [0.01 0.10 0.40];
nSeedsPinj       = max(1, ceil([15 5 4] * SEED_MULT));
pinjSeedStart    = 6000;

nSeedsSize       = max(1, ceil(4  * SEED_MULT));
sizeSeedStart    = 12000;
sizeCalNullRuns  = 2;
pinjectSizeSweep = pinject;

nSeedsGrid       = max(1, ceil(2  * SEED_MULT));
gridSeedStart    = 15000;
gridCalNullRuns  = 2;
pinjectGrid      = pinject;

nSeedsOverhead   = max(1, ceil(2 * SEED_MULT));
overheadSeedStart= 21000;

nSeedsBit        = max(1, ceil(5 * SEED_MULT));

% Detector list
detectorsWanted = {'relative', 'ratio', 'hybrid', 'crosscheck'};
doAddEnsemble   = false;

% Hybrid controls
q               = 8;
beta_lo_base    = 16;
beta_hi_base    = 128;
maskZeroRows    = true;
gammaMask       = 1e-9;
guardBetaScaleGE1 = true;

% Thresholds
alphaRelMult    = 16;
alpha           = 1.6;
alphaHybridRatio= alpha;

% Seeds for calibration
seedA   = 1;
seedInj = 2;

% Injection configuration
injCfg = struct();
injCfg.mode              = 'bitflip';
injCfg.target            = 'either';
injCfg.precision         = 'fp16';  

injCfg.enforceUUpperTri  = true;
injCfg.enforceLLowerTri  = true;
injCfg.bit.fixedIndices  = [];
injCfg.bit.class         = 'any';
injCfg.bit.kFlips        = 1;
injCfg.multiFaultProb    = 0.0;

% Visualization
doLogYInjected     = true;
logFloor           = 1e-12;
logYMaxQuantile    = 0.999;
showMinorGridLog   = true;
limitLogTicks      = true;
maxLogTicks        = 6;
annotateScoreDefs  = true;
scoreCaptionAtBottom = true;

% Output settings
saveOutputs  = true;
outBaseDir   = fullfile(pwd, 'mpft_out');
saveFigures  = true;
figFormats   = {'png', 'eps', 'pdf'}; 
figDPI       = 300;
saveCSV      = true;
saveMAT      = false;
saveJSONMeta = true;
runLabel     = sprintf('%s_fp64_extended', PROFILE);

if saveOutputs
    tstamp = datestr(now,'yyyymmdd_HHMMSS');
    outDir = fullfile(outBaseDir, sprintf('%s_%s', tstamp, runLabel));
    ensure_dir(outDir);
else
    outDir = fullfile(outBaseDir, 'nosave');
end

% Colors/names (internal=lowercase, display=Capitalized)
C.relative = [0.25 0.45 0.95];
C.ratio    = [1.00 0.60 0.00];
C.hybrid   = [0.20 0.70 0.30];
C.crosscheck = [0.70 0.20 0.60];
C.ensemble = [0.30 0.30 0.30];

if doAddEnsemble
    detectorsWanted = [detectorsWanted, {'ensemble'}];
end

% Map internal detector names to display names (Capitalized)
displayNameMap = containers.Map( ...
    {'relative', 'ratio', 'hybrid', 'crosscheck', 'ensemble'}, ...
    {'Relative', 'Ratio', 'Hybrid', 'CrossCheck', 'Ensemble'});

detNames = string(displayNameMap.values(detectorsWanted));  % Capitalized for display

detColsMap = containers.Map( ...
    {'relative','ratio','hybrid','crosscheck','ensemble'}, ...
    {C.relative, C.ratio, C.hybrid, C.crosscheck, C.ensemble});
detCols = zeros(numel(detNames),3);
for i=1:numel(detNames)
    detCols(i,:) = detColsMap(char(detectorsWanted(i)));
end

% Execution aggregator
EXEC = exec_init(detNames);

% ============================ Data prep =================================
fprintf('=================================================================\n');
fprintf('  MIXED-PRECISION FAULT-TOLERANT LU + FP64 EXTENDED ANALYSIS\n');
fprintf('=================================================================\n');
fprintf('Matrix size: %dx%d (wb=%d, %d panels)\n', m, n, wb, n/wb);
fprintf('Profile: %s | Seed mult: %.1fx\n', PROFILE, SEED_MULT);
fprintf('Detectors: %s\n', strjoin(detNames, ', '));
fprintf('Sections enabled:\n');
fprintf('  - Base comparison: %d matrices\n', nSeedsBaseComp);
fprintf('  - Pinject sweep: %s\n', tern(doPinjectSweep, 'YES', 'NO'));
fprintf('  - Size sweep: %s\n', tern(doSizeSweep, 'YES', 'NO'));
fprintf('  - FP16 bit sweeps: %s\n', tern(doBitPerIndexSweep || doBitClassSweep, 'YES', 'NO'));
fprintf('  - FP64 bit sweeps: %s (NEW!)\n', tern(doFP64BitSweeps, 'YES', 'NO'));
fprintf('  - Overhead analysis: %s\n', tern(doOverheadGrid, 'YES', 'NO'));
fprintf('=================================================================\n\n');

% Generate calibration matrix
rng(seedA);
fprintf('Generating calibration matrix %dx%d...\n', m, n);
A0 = randn(m,n,'double');
nb = n / wb;

% ============================ Null calibration ===========================
targetFPR   = 0.01;
nNullRuns   = 2;

fprintf('\n=================================================================\n');
fprintf('  NULL CALIBRATION (pinject=0, targetFPR=%.4f)\n', targetFPR);
fprintf('=================================================================\n');

alphaStar = calibrate_ratio_alpha(A0, m, n, wb, useGPU, ...
    q, beta_lo_base, beta_hi_base, maskZeroRows, gammaMask, targetFPR, nNullRuns, injCfg, alphaRelMult);
alpha = alphaStar; alphaHybridRatio = alphaStar;
fprintf('  ✓ Ratio alpha = %.3f\n', alpha);

alphaRelStar = calibrate_relative_alpha_mult(A0, m, n, wb, useGPU, ...
    q, beta_lo_base, beta_hi_base, maskZeroRows, gammaMask, targetFPR, nNullRuns, injCfg, alphaRelMult);
alphaRelMult = alphaRelStar;
fprintf('  ✓ Relative alphaRelMult = %.3f\n', alphaRelMult);

alphaCrossCheckStar = calibrate_crosscheck_alpha_mult(A0, m, n, wb, useGPU, ...
    q, beta_lo_base, beta_hi_base, maskZeroRows, gammaMask, targetFPR, nNullRuns, injCfg, alphaRelMult);
alphaCrossCheckMult = alphaCrossCheckStar;
fprintf('  ✓ CrossCheck alphaCrossCheckMult = %.3f\n', alphaCrossCheckMult);

[~, ~, sStar, baseFPR] = calibrate_hybrid_beta_scale( ...
    A0, m, n, wb, useGPU, alphaHybridRatio, beta_lo_base, beta_hi_base, q, maskZeroRows, gammaMask, targetFPR, nNullRuns, injCfg, alphaRelMult);
if guardBetaScaleGE1, sUse = max(1, sStar); else, sUse = sStar; end
beta_lo_mult = beta_lo_base * sUse;
beta_hi_mult = beta_hi_base * sUse;
fprintf('  ✓ Hybrid beta scale s* = %.3f (used: %.3f)\n', sStar, sUse);
fprintf('    Null FPR ≈ %.4f\n', baseFPR);
drawnow;

% ============================ ENHANCED BASE COMPARISON ===================
fprintf('\n=================================================================\n');
fprintf('  ENHANCED BASE COMPARISON (Multi-Matrix)\n');
fprintf('=================================================================\n');
fprintf('Configuration:\n');
fprintf('  - Random matrices: %d\n', nSeedsBaseComp);
fprintf('  - Injection rate: %.2f\n', pinject);

% Determine sizes to test
if isempty(baseSizesVec)
    sizesToTest = [m];
    wbToTest = [wb];
    fprintf('  - Size: %dx%d (single)\n', m, n);
else
    sizesToTest = baseSizesVec;
    wbToTest = repmat(wb, size(baseSizesVec));
    fprintf('  - Sizes: %s (multi-size)\n', mat2str(sizesToTest));
end
fprintf('  - Total evaluations: %d matrices × %d panels = %d\n', ...
    nSeedsBaseComp, n/wb, nSeedsBaseComp * n/wb);
fprintf('-----------------------------------------------------------------\n');

% Initialize pooled counters
nd = numel(detectorsWanted);
TP_pool = zeros(nd,1); FP_pool = zeros(nd,1);
FN_pool = zeros(nd,1); TN_pool = zeros(nd,1);
REF_pool = zeros(nd,1);
FPclean_pool = zeros(nd,1); Nclean_pool = zeros(nd,1);
nTotalRuns = 0;

% Store individual runs
allRuns = cell(nd, 1);
for i=1:nd, allRuns{i} = []; end

% Loop over sizes
for sIdx = 1:numel(sizesToTest)
    mCur = sizesToTest(sIdx);
    nCur = sizesToTest(sIdx);
    wbCur = wbToTest(sIdx);
    
    fprintf('\nSize %dx%d (wb=%d):\n', mCur, nCur, wbCur);
    
    % Loop over multiple random matrices
    for seedIdx = 1:nSeedsBaseComp
        seedMat = baseCompSeedStart + 1000*sIdx + seedIdx;
        
        % Generate random matrix
        rng(seedMat);
        A0_cur = randn(mCur, nCur, 'double');
        
        % Generate injection schedule
        seedInj_cur = seedMat + 5000;
        injEvents_cur = make_injection_schedule(mCur, nCur, wbCur, pinject, seedInj_cur, injCfg);
        
        % Run all detectors
        runs = cell(1, nd);
        for i = 1:nd
            detName = detectorsWanted{i};
            localAlpha = pick_alpha(detName, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult);
            
            runs{i} = run_one_detector( ...
                A0_cur, mCur, nCur, wbCur, useGPU, detName, ...
                localAlpha, alphaRelMult, q, beta_lo_mult, beta_hi_mult, ...
                maskZeroRows, gammaMask, injEvents_cur, verbosePanels, make3DPerRun, injCfg);
        end
        
        % Aggregate results
        for i=1:nd
            p = runs{i}.panel;
            inj = [p.injected]; flg = [p.flagged];
            
            TP_pool(i) = TP_pool(i) + sum(inj & flg);
            FP_pool(i) = FP_pool(i) + sum(~inj & flg);
            FN_pool(i) = FN_pool(i) + sum(inj & ~flg);
            TN_pool(i) = TN_pool(i) + sum(~inj & ~flg);
            REF_pool(i) = REF_pool(i) + sum(flg);
            
            [~, Nclean, FPclean] = compute_propagation_free_FPR_single(p);
            FPclean_pool(i) = FPclean_pool(i) + FPclean;
            Nclean_pool(i) = Nclean_pool(i) + Nclean;
            
            % Store for detailed plots
            allRuns{i} = [allRuns{i}, runs{i}];
        end
        
        nTotalRuns = nTotalRuns + 1;
        
        % Progress indicator
        if mod(seedIdx, max(1, floor(nSeedsBaseComp/4))) == 0 || seedIdx == nSeedsBaseComp
            fprintf('  Progress: %d/%d matrices (%.0f%% complete)\n', ...
                seedIdx, nSeedsBaseComp, 100*seedIdx/nSeedsBaseComp);
        end
    end
end

fprintf('\n✓ Completed %d total runs across %d size(s)\n', ...
    nTotalRuns, numel(sizesToTest));

% Compute pooled metrics with CIs
P_pool = TP_pool + FN_pool;
N_pool = FP_pool + TN_pool;
T_pool = P_pool + N_pool;

TPR_pool = TP_pool ./ max(P_pool, 1);
FPR_pool = FP_pool ./ max(N_pool, 1);
MR_pool  = FN_pool ./ max(P_pool, 1);
RR_pool  = REF_pool ./ max(T_pool, 1);
FPRpf_pool = FPclean_pool ./ max(Nclean_pool, 1);

alpha_ci = 0.05;
[TPR_lo, TPR_hi] = binom_wilson_ci(TP_pool, P_pool, alpha_ci);
[FPR_lo, FPR_hi] = binom_wilson_ci(FP_pool, N_pool, alpha_ci);
[MR_lo, MR_hi]   = binom_wilson_ci(FN_pool, P_pool, alpha_ci);
[RR_lo, RR_hi]   = binom_wilson_ci(REF_pool, T_pool, alpha_ci);
[FPRpf_lo, FPRpf_hi] = binom_wilson_ci(FPclean_pool, Nclean_pool, alpha_ci);

% Update EXEC
EXEC.TP = TP_pool'; EXEC.FP = FP_pool';
EXEC.FN = FN_pool'; EXEC.TN = TN_pool';
EXEC.REF = REF_pool';
EXEC.FPclean = FPclean_pool'; EXEC.Nclean = Nclean_pool';
EXEC.nRuns = nTotalRuns;

% ==================== ENHANCED BASE COMPARISON PLOTS =====================

% Plot 1: Summary Rates with CIs
figure('Name','Enhanced Summary Rates (Pooled)','Color','w');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on; title(sprintf('TPR (n=%d runs)', nTotalRuns));
for i=1:nd
    bar(i, TPR_pool(i), 0.6, 'FaceColor', detCols(i,:));
    errorbar(i, TPR_pool(i), TPR_pool(i)-TPR_lo(i), TPR_hi(i)-TPR_pool(i), ...
        'k', 'LineWidth', 1.5, 'CapSize', 10, 'HandleVisibility','off');
end
xlim([0.5 nd+0.5]); ylim([0 1]); xticks(1:nd); xticklabels(detNames);
ylabel('Rate'); grid on; hold off;

nexttile; hold on; title(sprintf('FPR (n=%d runs)', nTotalRuns));
for i=1:nd
    bar(i, FPR_pool(i), 0.6, 'FaceColor', detCols(i,:));
    errorbar(i, FPR_pool(i), FPR_pool(i)-FPR_lo(i), FPR_hi(i)-FPR_pool(i), ...
        'k', 'LineWidth', 1.5, 'CapSize', 10, 'HandleVisibility','off');
end
xlim([0.5 nd+0.5]); ylim([0 1]); xticks(1:nd); xticklabels(detNames);
ylabel('Rate'); grid on; hold off;

nexttile; hold on; title('Miss Rate');
for i=1:nd
    bar(i, MR_pool(i), 0.6, 'FaceColor', detCols(i,:));
    errorbar(i, MR_pool(i), MR_pool(i)-MR_lo(i), MR_hi(i)-MR_pool(i), ...
        'k', 'LineWidth', 1.5, 'CapSize', 10, 'HandleVisibility','off');
end
xlim([0.5 nd+0.5]); ylim([0 1]); xticks(1:nd); xticklabels(detNames);
ylabel('Rate'); grid on; hold off;

nexttile; hold on; title('Refactorization Rate');
for i=1:nd
    bar(i, RR_pool(i), 0.6, 'FaceColor', detCols(i,:));
    errorbar(i, RR_pool(i), RR_pool(i)-RR_lo(i), RR_hi(i)-RR_pool(i), ...
        'k', 'LineWidth', 1.5, 'CapSize', 10, 'HandleVisibility','off');
end
xlim([0.5 nd+0.5]); ylim([0 1]); xticks(1:nd); xticklabels(detNames);
ylabel('Rate'); grid on; hold off;

drawnow;
maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);

% Plot 2: Panel Timeline
firstRuns = cell(1, nd);
for i=1:nd, firstRuns{i} = allRuns{i}(1); end
plot_panel_timeline(firstRuns, detNames);
maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);

% Plot 3: Metric Separation (pooled)
plot_metric_separation_pooled(allRuns, detNames, detCols, doLogYInjected, ...
    logFloor, logYMaxQuantile, showMinorGridLog, limitLogTicks, ...
    maxLogTicks, annotateScoreDefs, scoreCaptionAtBottom);
maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);

% Console Summary
fprintf('\n=============== POOLED SUMMARY (%d total runs) =================\n', nTotalRuns);
fprintf('%-10s | TP    FP   FN   TN   | TPR    [95%% CI]        | FPR    [95%% CI]        \n', 'Detector');
fprintf('-----------|----------------------|----------------------|----------------------\n');
for i = 1:nd
    fprintf('%-10s | %4d %4d %4d %4d | %.3f [%.3f,%.3f] | %.3f [%.3f,%.3f]\n', ...
        detNames(i), TP_pool(i), FP_pool(i), FN_pool(i), TN_pool(i), ...
        TPR_pool(i), TPR_lo(i), TPR_hi(i), ...
        FPR_pool(i), FPR_lo(i), FPR_hi(i));
end
fprintf('=================================================================\n');

if doPropagationFPR
    figure('Name','FPR vs Propagation-free (Pooled)','Color','w');
    tl = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    
    nexttile; hold on; title('FPR (nominal, pooled)');
    for i=1:nd
        bar(i, FPR_pool(i), 0.6, 'FaceColor', detCols(i,:));
        errorbar(i, FPR_pool(i), FPR_pool(i)-FPR_lo(i), FPR_hi(i)-FPR_pool(i), ...
            'k', 'LineWidth',1.5, 'HandleVisibility','off');
    end
    xlim([0.5 nd+0.5]); ylim([0 1]); xticks(1:nd); xticklabels(detNames);
    ylabel('Rate'); grid on; hold off;
    
    nexttile; hold on; title('FPR (propagation-free, pooled)');
    for i=1:nd
        bar(i, FPRpf_pool(i), 0.6, 'FaceColor', detCols(i,:));
        errorbar(i, FPRpf_pool(i), FPRpf_pool(i)-FPRpf_lo(i), FPRpf_hi(i)-FPRpf_pool(i), ...
            'k', 'LineWidth',1.5, 'HandleVisibility','off');
    end
    xlim([0.5 nd+0.5]); ylim([0 1]); xticks(1:nd); xticklabels(detNames);
    ylabel('Rate'); grid on; hold off;
    
    fprintf('\nPropagation-Free FPR:\n');
    for i=1:nd
        fprintf('  %-10s | FPR_pf=%.3f [%.3f,%.3f] (FP=%d / N_clean=%d)\n', ...
            detNames(i), FPRpf_pool(i), FPRpf_lo(i), FPRpf_hi(i), ...
            FPclean_pool(i), Nclean_pool(i));
    end
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
end

% Detection delay histogram
figure('Name','Detection Delay (Pooled)','Color','w');
tiledlayout(nd,1,'Padding','compact','TileSpacing','compact');
for i = 1:nd
    allDelays = [];
    for r = 1:numel(allRuns{i})
        delays = detection_delays(allRuns{i}(r).panel);
        allDelays = [allDelays, delays];
    end
    nexttile; histogram(allDelays, 'BinMethod','integers','FaceColor',detCols(i,:));
    xlabel('Panels from injection to flag'); ylabel('Count');
    title(sprintf('%s: delay histogram (n=%d runs)', detectorsWanted{i}, nTotalRuns));
    grid on;
end
maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);

% Memory cleanup
clear allRuns firstRuns A0_cur injEvents_cur runs p;
clear TP_pool FP_pool FN_pool TN_pool P_pool N_pool T_pool;
clear TPR_pool FPR_pool MR_pool RR_pool FPRpf_pool;
clear TPR_lo TPR_hi FPR_lo FPR_hi MR_lo MR_hi RR_lo RR_hi FPRpf_lo FPRpf_hi;

% ============================ Pinject sweep ==============================
if doPinjectSweep
    fprintf('\n=================================================================\n');
    fprintf('  PINJECT SWEEP\n');
    fprintf('=================================================================\n');
    fprintf('Rates: %s, Seeds: %s\n', mat2str(pinjectVec), mat2str(nSeedsPinj));
    
    PS = sweep_pinject_all_detectors_extended(pinjectVec, nSeedsPinj, pinjSeedStart, ...
        A0, m, n, wb, useGPU, detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injCfg, doPropagationFPR);

    figure('Name','TPR vs pinject','Color','w'); hold on;
    set(gca, 'XScale', 'log');
    hMain = gobjects(numel(detectorsWanted),1);
    for i=1:numel(detectorsWanted)
        hMain(i) = plot(pinjectVec, PS.TPR_mean(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth',1.5, 'DisplayName', detNames(i));
        errorbar(pinjectVec, PS.TPR_mean(i,:), PS.TPR_mean(i,:)-PS.TPR_lo(i,:), PS.TPR_hi(i,:)-PS.TPR_mean(i,:), 'Color', detCols(i,:), 'LineStyle','none', 'HandleVisibility','off');
    end
    xlabel('pinject (log)'); ylabel('TPR'); grid on; ylim([0 1]); legend(hMain, detNames,'Location','SouthEast'); hold off;
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);

    figure('Name','FPR vs pinject','Color','w'); hold on;
    set(gca, 'XScale', 'log');
    hMain = gobjects(numel(detectorsWanted),1);
    for i=1:numel(detectorsWanted)
        hMain(i) = plot(pinjectVec, PS.FPR_mean(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth',1.5, 'DisplayName', detNames(i));
        errorbar(pinjectVec, PS.FPR_mean(i,:), PS.FPR_mean(i,:)-PS.FPR_lo(i,:), PS.FPR_hi(i,:)-PS.FPR_mean(i,:), 'Color', detCols(i,:), 'LineStyle','none', 'HandleVisibility','off');
    end
    xlabel('pinject (log)'); ylabel('FPR'); grid on; ylim([0 1]); legend(hMain, detNames,'Location','SouthEast'); hold off;
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
    clear PS;
end

% ============================ Size sweep =================================
if doSizeSweep
    fprintf('\n=================================================================\n');
    fprintf('  SIZE SWEEP\n');
    fprintf('=================================================================\n');
    fprintf('Sizes: %s, Seeds per size: %d\n', mat2str(sizesVec), nSeedsSize);
    
    SS = size_sweep_with_CIs(sizesVec, wb, A0, seedA, useGPU, ...
        detectorsWanted, q, beta_lo_base, beta_hi_base, maskZeroRows, gammaMask, ...
        targetFPR, sizeCalNullRuns, pinjectSizeSweep, nSeedsSize, sizeSeedStart, injCfg, guardBetaScaleGE1, alphaRelMult);

    figure('Name','Size Sweep: TPR','Color','w'); hold on;
    hMain = gobjects(numel(detectorsWanted),1);
    for i=1:numel(detectorsWanted)
        shaded_CI(sizesVec, SS.TPR_lo(i,:), SS.TPR_hi(i,:), detCols(i,:), 0.12);
        hMain(i) = plot(sizesVec, SS.TPR_mean(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth',1.5, 'DisplayName', detNames(i));
    end
    xlabel('n (m=n)'); ylabel('TPR'); grid on; ylim([0 1]); legend(hMain, detNames,'Location','SouthEast'); hold off;
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);

    figure('Name','Size Sweep: FPR','Color','w'); hold on;
    hMain = gobjects(numel(detectorsWanted),1);
    for i=1:numel(detectorsWanted)
        shaded_CI(sizesVec, SS.FPR_lo(i,:), SS.FPR_hi(i,:), detCols(i,:), 0.12);
        hMain(i) = plot(sizesVec, SS.FPR_mean(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth',1.5, 'DisplayName', detNames(i));
    end
    xlabel('n (m=n)'); ylabel('FPR'); grid on; ylim([0 1]); legend(hMain, detNames,'Location','SouthEast'); hold off;
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
    clear SS;
end

% ============================ FP16 BIT SWEEPS ============================
if doBitPerIndexSweep
    fprintf('\n=================================================================\n');
    fprintf('  FP16 PER-BIT TPR SWEEP\n');
    fprintf('=================================================================\n');
    fprintf('Seeds per bit: %d\n', nSeedsBit);
    
    bitIdxVec = 0:15; injSeeds = 3000 + (1:nSeedsBit);
    S_L = sweep_bit_index_all_detectors(bitIdxVec, 'L', 'fp16', A0, m, n, wb, useGPU, detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injSeeds, injCfg);
    S_U = sweep_bit_index_all_detectors(bitIdxVec, 'U', 'fp16', A0, m, n, wb, useGPU, detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injSeeds, injCfg);

    figure('Name','FP16 Per-bit TPR','Color','w');
    tl = tiledlayout(2,1,'Padding','compact','TileSpacing','compact');
    nexttile; hold on; title(sprintf('FP16 L-only: TPR vs bit (nSeeds=%d)', nSeedsBit));
    for i=1:numel(detectorsWanted), plot(bitIdxVec, S_L.TPR(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:)); end
    decorate_bitaxis_fp16(); ylim([0 1]); ylabel('TPR'); grid on; legend(detNames, 'Location','South'); hold off;
    nexttile; hold on; title(sprintf('FP16 U-only: TPR vs bit (nSeeds=%d)', nSeedsBit));
    for i=1:numel(detectorsWanted), plot(bitIdxVec, S_U.TPR(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:)); end
    decorate_bitaxis_fp16(); ylim([0 1]); xlabel('Bit index'); ylabel('TPR'); grid on; legend(detNames, 'Location','South'); hold off;
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
end

if doBitClassSweep
    fprintf('\n=================================================================\n');
    fprintf('  FP16 PER-CLASS TPR SWEEP\n');
    fprintf('=================================================================\n');
    fprintf('Seeds per class: %d\n', nSeedsBit);
    
    injSeeds = 4000 + (1:nSeedsBit); 
    classes = {'frac','exp','sign'};
    classCols = [0.26 0.52 0.96; 1.00 0.60 0.00; 0.20 0.70 0.30];
    
    Scls_L = sweep_bit_class_all_detectors(classes, 'L', 'fp16', A0, m, n, wb, useGPU, detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injSeeds, injCfg);
    Scls_U = sweep_bit_class_all_detectors(classes, 'U', 'fp16', A0, m, n, wb, useGPU, detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injSeeds, injCfg);

    figure('Name','FP16 Per-class TPR','Color','w');
    tl = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    nexttile; title(sprintf('FP16 L-only (nSeeds=%d)', nSeedsBit)); hold on;
    YL = Scls_L.TPR; hb = bar(YL, 'grouped');
    for c=1:numel(classes), hb(c).FaceColor = classCols(c,:); end
    set(gca,'XTick',1:numel(detectorsWanted),'XTickLabel',detNames);
    ylim([0 1]); grid on; ylabel('TPR'); legend(classes, 'Location','South');
    nexttile; title(sprintf('FP16 U-only (nSeeds=%d)', nSeedsBit)); hold on;
    YU = Scls_U.TPR; hb = bar(YU, 'grouped');
    for c=1:numel(classes), hb(c).FaceColor = classCols(c,:); end
    set(gca,'XTick',1:numel(detectorsWanted),'XTickLabel',detNames);
    ylim([0 1]); grid on; ylabel('TPR'); legend(classes, 'Location','South');
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
end

% ============================ FP64 BIT SWEEPS ============================
if doFP64BitSweeps
    fprintf('\n=================================================================\n');
    fprintf('  FP64 EXTENDED ANALYSIS (64 bits)\n');
    fprintf('=================================================================\n');
    
    bitIdxVec64 = 0:63;  % All 64 bits of FP64
    injSeeds = 5000 + (1:nSeedsBit);
    
    fprintf('Running FP64 L-factor sweep (64 bits × %d seeds)...\n', nSeedsBit);
    tic;
    S64_L = sweep_bit_index_all_detectors(bitIdxVec64, 'L', 'fp64', A0, m, n, wb, useGPU, ...
        detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, ...
        q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injSeeds, injCfg);
    fprintf('  ✓ FP64 L-factor done in %.1f sec\n', toc);
    
    fprintf('Running FP64 U-factor sweep (64 bits × %d seeds)...\n', nSeedsBit);
    tic;
    S64_U = sweep_bit_index_all_detectors(bitIdxVec64, 'U', 'fp64', A0, m, n, wb, useGPU, ...
        detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, ...
        q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injSeeds, injCfg);
    fprintf('  ✓ FP64 U-factor done in %.1f sec\n', toc);
    
    % FP64 class sweeps
    injSeeds = 6000 + (1:nSeedsBit);
    
    fprintf('Running FP64 L-factor class sweep...\n');
    tic;
    Scls64_L = sweep_bit_class_all_detectors(classes, 'L', 'fp64', A0, m, n, wb, useGPU, ...
        detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, ...
        q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injSeeds, injCfg);
    fprintf('  ✓ FP64 L-factor done in %.1f sec\n', toc);
    
    fprintf('Running FP64 U-factor class sweep...\n');
    tic;
    Scls64_U = sweep_bit_class_all_detectors(classes, 'U', 'fp64', A0, m, n, wb, useGPU, ...
        detectorsWanted, alphaRelMult, alpha, alphaHybridRatio, alphaCrossCheckMult, ...
        q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injSeeds, injCfg);
    fprintf('  ✓ FP64 U-factor done in %.1f sec\n', toc);
    
    % Plot FP64 per-bit
    figure('Name','FP64 Per-bit TPR','Color','w');
    tiledlayout(2,1,'Padding','compact','TileSpacing','compact');
    nexttile; hold on; title(sprintf('FP64 L-only: TPR vs bit (nSeeds=%d)', nSeedsBit));
    for i=1:numel(detectorsWanted)
        plot(bitIdxVec64, S64_L.TPR(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth', 1.5, 'MarkerSize', 3);
    end
    decorate_bitaxis_fp64(); ylim([0 1]); ylabel('TPR'); grid on; legend(detNames, 'Location','South'); hold off;
    
    nexttile; hold on; title(sprintf('FP64 U-only: TPR vs bit (nSeeds=%d)', nSeedsBit));
    for i=1:numel(detectorsWanted)
        plot(bitIdxVec64, S64_U.TPR(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth', 1.5, 'MarkerSize', 3);
    end
    decorate_bitaxis_fp64(); ylim([0 1]); xlabel('Bit index'); ylabel('TPR'); grid on; legend(detNames, 'Location','South'); hold off;
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
    % Plot FP64 per-class
    figure('Name','FP64 Per-class TPR','Color','w');
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    nexttile; title(sprintf('FP64 L-only (nSeeds=%d)', nSeedsBit)); hold on;
    YL64 = Scls64_L.TPR; hb = bar(YL64, 'grouped');
    for c=1:numel(classes), hb(c).FaceColor = classCols(c,:); end
    set(gca,'XTick',1:numel(detectorsWanted),'XTickLabel',detNames);
    ylim([0 1]); grid on; ylabel('TPR'); legend(classes, 'Location','South');
    
    nexttile; title(sprintf('FP64 U-only (nSeeds=%d)', nSeedsBit)); hold on;
    YU64 = Scls64_U.TPR; hb = bar(YU64, 'grouped');
    for c=1:numel(classes), hb(c).FaceColor = classCols(c,:); end
    set(gca,'XTick',1:numel(detectorsWanted),'XTickLabel',detNames);
    ylim([0 1]); grid on; ylabel('TPR'); legend(classes, 'Location','South');
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
   % ===== Figure 1: L-factor Analysis (FP16 vs FP64) =====
figure('Name','L-factor: FP16 vs FP64','Color','w','Position',[100 100 900 800]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

% Subplot 1: L-factor per-bit (FP16 overlaid on FP64)
nexttile; hold on; title('L-factor: FP16 bits on FP64 scale');

h_legend = [];
legend_labels = {};

for i=1:numel(detectorsWanted)
    % Plot FP64 as background (dashed, lighter)
    h_fp64 = plot(bitIdxVec64, S64_L.TPR(i,:), '--', 'Color', detCols(i,:)*0.4 + [0.6 0.6 0.6], 'LineWidth', 1.5);
    h_legend(end+1) = h_fp64;
    legend_labels{end+1} = sprintf('%s FP64', detNames{i});
    
    % Plot FP16 overlaid (if available)
    if exist('S_L', 'var')
        h_fp16 = plot(0:15, S_L.TPR(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth', 1.5, 'MarkerSize', 6);
        h_legend(end+1) = h_fp16;
        legend_labels{end+1} = sprintf('%s FP16', detNames{i});
    end
end
xlim([0 63]); xticks(0:5:63); ylim([0 1]); ylabel('TPR'); grid on; 
legend(h_legend, legend_labels, 'Location','eastoutside', 'NumColumns', 1, 'FontSize', 9); 
hold off;

% Subplot 2: L-factor class comparison
nexttile; hold on; grid on;
classes_labels = {'frac', 'exp', 'sign'};
x_pos = 1:3;

h_legend = [];
legend_labels = {};

% Extract class data
if exist('Scls_L', 'var')
    mean_L_frac16 = Scls_L.TPR(:, 1);
    mean_L_exp16  = Scls_L.TPR(:, 2);
    mean_L_sign16 = Scls_L.TPR(:, 3);
else
    mean_L_frac16 = zeros(numel(detectorsWanted),1);
    mean_L_exp16  = zeros(numel(detectorsWanted),1);
    mean_L_sign16 = zeros(numel(detectorsWanted),1);
end

mean_L_frac64 = Scls64_L.TPR(:, 1);
mean_L_exp64  = Scls64_L.TPR(:, 2);
mean_L_sign64 = Scls64_L.TPR(:, 3);

% Plot each detector with consistent colors
for i = 1:numel(detectorsWanted)
    % FP64 first (dashed line with empty squares) - same color as top plot
    h_fp64 = plot(x_pos, [mean_L_frac64(i), mean_L_exp64(i), mean_L_sign64(i)], ...
        '--s', 'Color', detCols(i,:), 'LineWidth', 2.5, 'MarkerSize', 9, ...
        'MarkerFaceColor', 'none', 'MarkerEdgeColor', detCols(i,:));
    h_legend(end+1) = h_fp64;
    legend_labels{end+1} = sprintf('%s FP64', detNames{i});
    
    % FP16 (solid line with filled circles) - same color as top plot
    h_fp16 = plot(x_pos, [mean_L_frac16(i), mean_L_exp16(i), mean_L_sign16(i)], ...
        '-o', 'Color', detCols(i,:), 'LineWidth', 2.5, 'MarkerSize', 9, ...
        'MarkerFaceColor', detCols(i,:));
    h_legend(end+1) = h_fp16;
    legend_labels{end+1} = sprintf('%s FP16', detNames{i});
end

xticks(x_pos); xticklabels(classes_labels);
ylabel('TPR'); ylim([0 1.05]);
title('L-factor: Class comparison (FP16 vs FP64)');
legend(h_legend, legend_labels, 'Location', 'eastoutside', 'NumColumns', 1, 'FontSize', 9);
set(gca, 'FontSize', 11);

maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);

% ===== Figure 2: U-factor Analysis (FP16 vs FP64) =====
figure('Name','U-factor: FP16 vs FP64','Color','w','Position',[100 100 900 800]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

% Subplot 1: U-factor per-bit (FP16 overlaid on FP64)
nexttile; hold on; title('U-factor: FP16 bits on FP64 scale');

h_legend = [];
legend_labels = {};

for i=1:numel(detectorsWanted)
    % Plot FP64 as background (dashed)
    h_fp64 = plot(bitIdxVec64, S64_U.TPR(i,:), '--', 'Color', detCols(i,:)*0.4 + [0.6 0.6 0.6], 'LineWidth', 1.5);
    h_legend(end+1) = h_fp64;
    legend_labels{end+1} = sprintf('%s FP64', detNames{i});
    
    % Plot FP16 overlaid (if available)
    if exist('S_U', 'var')
        h_fp16 = plot(0:15, S_U.TPR(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth', 1.5, 'MarkerSize', 6);
        h_legend(end+1) = h_fp16;
        legend_labels{end+1} = sprintf('%s FP16', detNames{i});
    end
end
xlim([0 63]); xticks(0:5:63); ylim([0 1]); ylabel('TPR'); grid on; 
legend(h_legend, legend_labels, 'Location','eastoutside', 'NumColumns', 1, 'FontSize', 9); 
hold off;

% Subplot 2: U-factor class comparison
nexttile; hold on; grid on;

h_legend = [];
legend_labels = {};

% Extract class data
if exist('Scls_U', 'var')
    mean_U_frac16 = Scls_U.TPR(:, 1);
    mean_U_exp16  = Scls_U.TPR(:, 2);
    mean_U_sign16 = Scls_U.TPR(:, 3);
else
    mean_U_frac16 = zeros(numel(detectorsWanted),1);
    mean_U_exp16  = zeros(numel(detectorsWanted),1);
    mean_U_sign16 = zeros(numel(detectorsWanted),1);
end

mean_U_frac64 = Scls64_U.TPR(:, 1);
mean_U_exp64  = Scls64_U.TPR(:, 2);
mean_U_sign64 = Scls64_U.TPR(:, 3);

% Plot each detector with consistent colors
for i = 1:numel(detectorsWanted)
    % FP64 first (dashed line with empty squares) - same color as top plot
    h_fp64 = plot(x_pos, [mean_U_frac64(i), mean_U_exp64(i), mean_U_sign64(i)], ...
        '--s', 'Color', detCols(i,:), 'LineWidth', 2.5, 'MarkerSize', 9, ...
        'MarkerFaceColor', 'none', 'MarkerEdgeColor', detCols(i,:));
    h_legend(end+1) = h_fp64;
    legend_labels{end+1} = sprintf('%s FP64', detNames{i});
    
    % FP16 (solid line with filled circles) - same color as top plot
    h_fp16 = plot(x_pos, [mean_U_frac16(i), mean_U_exp16(i), mean_U_sign16(i)], ...
        '-o', 'Color', detCols(i,:), 'LineWidth', 2.5, 'MarkerSize', 9, ...
        'MarkerFaceColor', detCols(i,:));
    h_legend(end+1) = h_fp16;
    legend_labels{end+1} = sprintf('%s FP16', detNames{i});
end

xticks(x_pos); xticklabels(classes_labels);
ylabel('TPR'); ylim([0 1.05]);
title('U-factor: Class comparison (FP16 vs FP64)');
legend(h_legend, legend_labels, 'Location', 'eastoutside', 'NumColumns', 1, 'FontSize', 9);
set(gca, 'FontSize', 11);

maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
    % Save FP64 results
    if saveOutputs && saveCSV
        % FP64 per-bit tables
        T64_L = array2table(S64_L.TPR', 'VariableNames', detNames, ...
            'RowNames', arrayfun(@(x) sprintf('bit%02d', x), bitIdxVec64, 'UniformOutput', false));
        T64_U = array2table(S64_U.TPR', 'VariableNames', detNames, ...
            'RowNames', arrayfun(@(x) sprintf('bit%02d', x), bitIdxVec64, 'UniformOutput', false));
        writetable(T64_L, fullfile(outDir, 'fp64_L_per_bit_TPR.csv'), 'WriteRowNames', true);
        writetable(T64_U, fullfile(outDir, 'fp64_U_per_bit_TPR.csv'), 'WriteRowNames', true);
        
        % FP64 per-class tables
        T64_cls_L = array2table(Scls64_L.TPR', 'VariableNames', detNames, 'RowNames', classes);
        T64_cls_U = array2table(Scls64_U.TPR', 'VariableNames', detNames, 'RowNames', classes);
        writetable(T64_cls_L, fullfile(outDir, 'fp64_L_per_class_TPR.csv'), 'WriteRowNames', true);
        writetable(T64_cls_U, fullfile(outDir, 'fp64_U_per_class_TPR.csv'), 'WriteRowNames', true);
        
        % FP16 tables (if available)
        if exist('S_L', 'var')
            T_L = array2table(S_L.TPR', 'VariableNames', detNames, ...
                'RowNames', arrayfun(@(x) sprintf('bit%02d', x), 0:15, 'UniformOutput', false));
            writetable(T_L, fullfile(outDir, 'fp16_L_per_bit_TPR.csv'), 'WriteRowNames', true);
        end
        if exist('S_U', 'var')
            T_U = array2table(S_U.TPR', 'VariableNames', detNames, ...
                'RowNames', arrayfun(@(x) sprintf('bit%02d', x), 0:15, 'UniformOutput', false));
            writetable(T_U, fullfile(outDir, 'fp16_U_per_bit_TPR.csv'), 'WriteRowNames', true);
        end
        if exist('Scls_L', 'var')
            T_cls_L = array2table(Scls_L.TPR', 'VariableNames', detNames, 'RowNames', classes);
            writetable(T_cls_L, fullfile(outDir, 'fp16_L_per_class_TPR.csv'), 'WriteRowNames', true);
        end
        if exist('Scls_U', 'var')
            T_cls_U = array2table(Scls_U.TPR', 'VariableNames', detNames, 'RowNames', classes);
            writetable(T_cls_U, fullfile(outDir, 'fp16_U_per_class_TPR.csv'), 'WriteRowNames', true);
        end
        
        fprintf('  ✓ FP64 and FP16 CSV files saved\n');
    end
    
    %clear S64_L S64_U Scls64_L Scls64_U bitIdxVec64;

    %% ========================================================================
    %  ENHANCED VISUALIZATIONS: FP16 vs FP64 COMPARISON
    %% ========================================================================
    
    fprintf('\n=== Generating FP16 vs FP64 Comparison Plots ===\n');
    
    % Check which data we have available
    have_fp16_L = exist('S_L', 'var');
    have_fp16_U = exist('S_U', 'var');
    have_fp16_cls_L = exist('Scls_L', 'var');
    have_fp16_cls_U = exist('Scls_U', 'var');
    
    %% OPTION 1: Side-by-Side Per-Bit Comparison
    fprintf('Creating Option 1: Side-by-Side Comparison...\n');
    fig_side = figure('Position', [100 100 1600 800], 'Name', 'FP16_vs_FP64_SideBySide', 'Color', 'w');
    tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    % Subplot 1: L-factor comparison
    nexttile; hold on; title('L-factor: FP16 (overlay) vs FP64 (background)', 'FontSize', 13);
    for i = 1:numel(detectorsWanted)
        % FP64 background (lighter, dashed)
        plot(0:63, S64_L.TPR(i,:), '--', 'Color', detCols(i,:)*0.5 + [0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', sprintf('%s FP64', detNames{i}));
        
        % FP16 overlay (if available)
        if have_fp16_L
            plot(0:15, S_L.TPR(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', sprintf('%s FP16', detNames{i}));
        end
    end
    xlim([0 63]); ylim([0 1]); xlabel('Bit Index'); ylabel('TPR'); grid on;
    legend('Location', 'eastoutside', 'NumColumns', 1, 'FontSize', 9);
    decorate_bitaxis_fp64();
    
    % Subplot 2: U-factor comparison
    nexttile; hold on; title('U-factor: FP16 (overlay) vs FP64 (background)', 'FontSize', 13);
    for i = 1:numel(detectorsWanted)
        % FP64 background
        plot(0:63, S64_U.TPR(i,:), '--', 'Color', detCols(i,:)*0.5 + [0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', sprintf('%s FP64', detNames{i}));
        
        % FP16 overlay (if available)
        if have_fp16_U
            plot(0:15, S_U.TPR(i,:), '-o', 'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', sprintf('%s FP16', detNames{i}));
        end
    end
    xlim([0 63]); ylim([0 1]); xlabel('Bit Index'); ylabel('TPR'); grid on;
    legend('Location', 'eastoutside', 'NumColumns', 1, 'FontSize', 9);
    decorate_bitaxis_fp64();
    
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
    
    %% OPTION 2: Class Comparison (Enhanced Bar Chart)
    fprintf('Creating Option 2: Bit Class Comparison...\n');
    fig_class = figure('Position', [100 100 1400 700], 'Name', 'FP16_vs_FP64_Classes', 'Color', 'w');
    tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    x_pos = 1:3;
    class_labels = {'Fraction', 'Exponent', 'Sign'};
    
    % Subplot 1: L-factor class comparison
    nexttile; hold on; grid on;
    title('L-factor: Bit Class Comparison', 'FontSize', 13);
    
    for i = 1:numel(detectorsWanted)
        % Extract FP64 class data
        fp64_vals = [Scls64_L.TPR(i,1), Scls64_L.TPR(i,2), Scls64_L.TPR(i,3)];
        
        % Plot FP64 (dashed line, empty markers)
        plot(x_pos, fp64_vals, '--s', 'Color', detCols(i,:), 'LineWidth', 2.5, 'MarkerSize', 10, ...
            'MarkerFaceColor', 'none', 'MarkerEdgeColor', detCols(i,:), 'DisplayName', sprintf('%s FP64', detNames{i}));
        
        % Plot FP16 (solid line, filled markers) - if available
        if have_fp16_cls_L
            fp16_vals = [Scls_L.TPR(i,1), Scls_L.TPR(i,2), Scls_L.TPR(i,3)];
            plot(x_pos, fp16_vals, '-o', 'Color', detCols(i,:), 'LineWidth', 2.5, 'MarkerSize', 10, ...
                'MarkerFaceColor', detCols(i,:), 'DisplayName', sprintf('%s FP16', detNames{i}));
        end
    end
    
    xticks(x_pos); xticklabels(class_labels);
    ylabel('TPR', 'FontSize', 11); ylim([0 1.05]);
    legend('Location', 'eastoutside', 'NumColumns', 1, 'FontSize', 9);
    
    % Subplot 2: U-factor class comparison
    nexttile; hold on; grid on;
    title('U-factor: Bit Class Comparison', 'FontSize', 13);
    
    for i = 1:numel(detectorsWanted)
        % Extract FP64 class data
        fp64_vals = [Scls64_U.TPR(i,1), Scls64_U.TPR(i,2), Scls64_U.TPR(i,3)];
        
        % Plot FP64
        plot(x_pos, fp64_vals, '--s', 'Color', detCols(i,:), 'LineWidth', 2.5, 'MarkerSize', 10, ...
            'MarkerFaceColor', 'none', 'MarkerEdgeColor', detCols(i,:), 'DisplayName', sprintf('%s FP64', detNames{i}));
        
        % Plot FP16 - if available
        if have_fp16_cls_U
            fp16_vals = [Scls_U.TPR(i,1), Scls_U.TPR(i,2), Scls_U.TPR(i,3)];
            plot(x_pos, fp16_vals, '-o', 'Color', detCols(i,:), 'LineWidth', 2.5, 'MarkerSize', 10, ...
                'MarkerFaceColor', detCols(i,:), 'DisplayName', sprintf('%s FP16', detNames{i}));
        end
    end
    
    xticks(x_pos); xticklabels(class_labels);
    ylabel('TPR', 'FontSize', 11); ylim([0 1.05]);
    legend('Location', 'eastoutside', 'NumColumns', 1, 'FontSize', 9);
    
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
    
 %% OPTION 3: Enhanced Heatmap with Clear Divisions
fprintf('Creating Option 3: Enhanced Heatmap...\n');
fig_heat = figure('Position', [100 100 1400 900], 'Name', 'FP16_vs_FP64_Heatmap', 'Color', 'w');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% L-factor FP16 heatmap
if have_fp16_L
    ax1 = nexttile;
    imagesc(ax1, 0:15, 1:numel(detectorsWanted), S_L.TPR);
    colorbar(ax1); caxis(ax1, [0 1]); 
    colormap(ax1, flipud(hot));
    xlabel(ax1, 'Bit Index'); ylabel(ax1, 'Detector');
    yticks(ax1, 1:numel(detectorsWanted)); yticklabels(ax1, detNames);
    title(ax1, 'FP16 L-factor (16 bits)', 'FontSize', 12, 'FontWeight', 'bold');
    xlim(ax1, [0 16]);
    
    % Add clear divisions
    hold(ax1, 'on');
    xline(ax1, 9.5, 'k-', 'LineWidth', 3);
    xline(ax1, 14.5, 'k-', 'LineWidth', 3);
    xline(ax1, 9.5, 'w--', 'LineWidth', 1.5);
    xline(ax1, 14.5, 'w--', 'LineWidth', 1.5);
    
    ylims = ylim(ax1);
    patch(ax1, [0 9.5 9.5 0], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [0.7 0.9 1], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    patch(ax1, [9.5 14.5 14.5 9.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [1 0.9 0.7], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    patch(ax1, [14.5 16 16 14.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [1 0.7 0.7], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    
    text(ax1, 4.75, ylims(2)+0.4, 'FRAC', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Color', [0 0.4 0.8]);
    text(ax1, 12, ylims(2)+0.4, 'EXP', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Color', [0.8 0.4 0]);
    text(ax1, 15.25, ylims(2)+0.4, 'S', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Color', [0.8 0 0]);
    hold(ax1, 'off');
end

% L-factor FP64 heatmap
ax2 = nexttile;
imagesc(ax2, 0:63, 1:numel(detectorsWanted), S64_L.TPR);
colorbar(ax2); caxis(ax2, [0 1]); 
colormap(ax2, flipud(hot));
xlabel(ax2, 'Bit Index'); ylabel(ax2, 'Detector');
yticks(ax2, 1:numel(detectorsWanted)); yticklabels(ax2, detNames);
title(ax2, 'FP64 L-factor (64 bits)', 'FontSize', 12, 'FontWeight', 'bold');
xlim(ax2, [0 64]);

% Add clear divisions
hold(ax2, 'on');
xline(ax2, 51.5, 'k-', 'LineWidth', 3);
xline(ax2, 62.5, 'k-', 'LineWidth', 3);
xline(ax2, 51.5, 'w--', 'LineWidth', 1.5);
xline(ax2, 62.5, 'w--', 'LineWidth', 1.5);

ylims = ylim(ax2);
patch(ax2, [0 51.5 51.5 0], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [0.7 0.9 1], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
patch(ax2, [51.5 62.5 62.5 51.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [1 0.9 0.7], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
patch(ax2, [62.5 64 64 62.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [1 0.7 0.7], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

text(ax2, 25.75, ylims(2)+0.4, 'FRACTION', 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0 0.4 0.8]);
text(ax2, 57, ylims(2)+0.4, 'EXP', 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0.8 0.4 0]);
text(ax2, 63.25, ylims(2)+0.4, 'S', 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0.8 0 0]);
hold(ax2, 'off');

% U-factor FP16 heatmap
if have_fp16_U
    ax3 = nexttile;
    imagesc(ax3, 0:15, 1:numel(detectorsWanted), S_U.TPR);
    colorbar(ax3); caxis(ax3, [0 1]); 
    colormap(ax3, flipud(hot));
    xlabel(ax3, 'Bit Index'); ylabel(ax3, 'Detector');
    yticks(ax3, 1:numel(detectorsWanted)); yticklabels(ax3, detNames);
    title(ax3, 'FP16 U-factor (16 bits)', 'FontSize', 12, 'FontWeight', 'bold');
    xlim(ax3, [0 16]);
    
    % Add clear divisions
    hold(ax3, 'on');
    xline(ax3, 9.5, 'k-', 'LineWidth', 3);
    xline(ax3, 14.5, 'k-', 'LineWidth', 3);
    xline(ax3, 9.5, 'w--', 'LineWidth', 1.5);
    xline(ax3, 14.5, 'w--', 'LineWidth', 1.5);
    
    ylims = ylim(ax3);
    patch(ax3, [0 9.5 9.5 0], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [0.7 0.9 1], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    patch(ax3, [9.5 14.5 14.5 9.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [1 0.9 0.7], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    patch(ax3, [14.5 16 16 14.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [1 0.7 0.7], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    
    text(ax3, 4.75, ylims(2)+0.4, 'FRAC', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Color', [0 0.4 0.8]);
    text(ax3, 12, ylims(2)+0.4, 'EXP', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Color', [0.8 0.4 0]);
    text(ax3, 15.25, ylims(2)+0.4, 'S', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Color', [0.8 0 0]);
    hold(ax3, 'off');
end

% U-factor FP64 heatmap
ax4 = nexttile;
imagesc(ax4, 0:63, 1:numel(detectorsWanted), S64_U.TPR);
colorbar(ax4); caxis(ax4, [0 1]); 
colormap(ax4, flipud(hot));
xlabel(ax4, 'Bit Index'); ylabel(ax4, 'Detector');
yticks(ax4, 1:numel(detectorsWanted)); yticklabels(ax4, detNames);
title(ax4, 'FP64 U-factor (64 bits)', 'FontSize', 12, 'FontWeight', 'bold');
xlim(ax4, [0 64]);

% Add clear divisions
hold(ax4, 'on');
xline(ax4, 51.5, 'k-', 'LineWidth', 3);
xline(ax4, 62.5, 'k-', 'LineWidth', 3);
xline(ax4, 51.5, 'w--', 'LineWidth', 1.5);
xline(ax4, 62.5, 'w--', 'LineWidth', 1.5);

ylims = ylim(ax4);
patch(ax4, [0 51.5 51.5 0], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [0.7 0.9 1], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
patch(ax4, [51.5 62.5 62.5 51.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [1 0.9 0.7], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
patch(ax4, [62.5 64 64 62.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [1 0.7 0.7], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

text(ax4, 25.75, ylims(2)+0.4, 'FRACTION', 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0 0.4 0.8]);
text(ax4, 57, ylims(2)+0.4, 'EXP', 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0.8 0.4 0]);
text(ax4, 63.25, ylims(2)+0.4, 'S', 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0.8 0 0]);
hold(ax4, 'off');

maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
    %% OPTION 4: Normalized Comparison (Aligned Bit Classes)
    fprintf('Creating Option 4: Normalized Comparison...\n');
    fig_norm = figure('Position', [100 100 1600 600], 'Name', 'FP16_vs_FP64_Normalized', 'Color', 'w');
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    % L-factor normalized
    nexttile; hold on; grid on;
    title('L-factor: Normalized by Bit Class', 'FontSize', 13);
    
    for i = 1:numel(detectorsWanted)
        % FP16 normalized positions
        if have_fp16_L
            fp16_frac_x = linspace(0, 0.9, 10);
            fp16_exp_x = linspace(1, 1.9, 5);
            fp16_sign_x = 3;
            
            plot(fp16_frac_x, S_L.TPR(i,1:10), 'o-', 'LineWidth', 2, 'MarkerSize', 6, ...
                'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'DisplayName', sprintf('%s FP16', detNames{i}));
            plot(fp16_exp_x, S_L.TPR(i,11:15), 'o-', 'LineWidth', 2, 'MarkerSize', 6, ...
                'Color', detCols(i,:), 'HandleVisibility', 'off');
            plot(fp16_sign_x, S_L.TPR(i,16), 'o', 'MarkerSize', 10, 'LineWidth', 2, ...
                'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'HandleVisibility', 'off');
        end
        
        % FP64 normalized positions
        fp64_frac_x = linspace(0, 0.9, 52);
        fp64_exp_x = linspace(1, 1.9, 11);
        fp64_sign_x = 3;
        
        plot(fp64_frac_x, S64_L.TPR(i,1:52), 's-', 'LineWidth', 1.5, 'MarkerSize', 3, ...
            'Color', detCols(i,:)*0.6 + [0.4 0.4 0.4], 'DisplayName', sprintf('%s FP64', detNames{i}));
        plot(fp64_exp_x, S64_L.TPR(i,53:63), 's-', 'LineWidth', 1.5, 'MarkerSize', 3, ...
            'Color', detCols(i,:)*0.6 + [0.4 0.4 0.4], 'HandleVisibility', 'off');
        plot(fp64_sign_x, S64_L.TPR(i,64), 's', 'MarkerSize', 8, 'LineWidth', 2, ...
            'Color', detCols(i,:)*0.6 + [0.4 0.4 0.4], 'HandleVisibility', 'off');
    end
    
    xlim([-0.2 3.5]); ylim([0 1]);
    xticks([0.45 1.45 3]); xticklabels({'Fraction', 'Exponent', 'Sign'});
    ylabel('TPR'); legend('Location', 'best', 'FontSize', 8);
    xline(0.95, '--k', 'Alpha', 0.3); xline(2.5, '--k', 'Alpha', 0.3);
    
    % U-factor normalized
    nexttile; hold on; grid on;
    title('U-factor: Normalized by Bit Class', 'FontSize', 13);
    
    for i = 1:numel(detectorsWanted)
        % FP16 normalized positions
        if have_fp16_U
            fp16_frac_x = linspace(0, 0.9, 10);
            fp16_exp_x = linspace(1, 1.9, 5);
            fp16_sign_x = 3;
            
            plot(fp16_frac_x, S_U.TPR(i,1:10), 'o-', 'LineWidth', 2, 'MarkerSize', 6, ...
                'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'DisplayName', sprintf('%s FP16', detNames{i}));
            plot(fp16_exp_x, S_U.TPR(i,11:15), 'o-', 'LineWidth', 2, 'MarkerSize', 6, ...
                'Color', detCols(i,:), 'HandleVisibility', 'off');
            plot(fp16_sign_x, S_U.TPR(i,16), 'o', 'MarkerSize', 10, 'LineWidth', 2, ...
                'Color', detCols(i,:), 'MarkerFaceColor', detCols(i,:), 'HandleVisibility', 'off');
        end
        
        % FP64 normalized positions
        fp64_frac_x = linspace(0, 0.9, 52);
        fp64_exp_x = linspace(1, 1.9, 11);
        fp64_sign_x = 3;
        
        plot(fp64_frac_x, S64_U.TPR(i,1:52), 's-', 'LineWidth', 1.5, 'MarkerSize', 3, ...
            'Color', detCols(i,:)*0.6 + [0.4 0.4 0.4], 'DisplayName', sprintf('%s FP64', detNames{i}));
        plot(fp64_exp_x, S64_U.TPR(i,53:63), 's-', 'LineWidth', 1.5, 'MarkerSize', 3, ...
            'Color', detCols(i,:)*0.6 + [0.4 0.4 0.4], 'HandleVisibility', 'off');
        plot(fp64_sign_x, S64_U.TPR(i,64), 's', 'MarkerSize', 8, 'LineWidth', 2, ...
            'Color', detCols(i,:)*0.6 + [0.4 0.4 0.4], 'HandleVisibility', 'off');
    end
    
    xlim([-0.2 3.5]); ylim([0 1]);
    xticks([0.45 1.45 3]); xticklabels({'Fraction', 'Exponent', 'Sign'});
    ylabel('TPR'); legend('Location', 'best', 'FontSize', 8);
    xline(0.95, '--k', 'Alpha', 0.3); xline(2.5, '--k', 'Alpha', 0.3);
    
    maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);
    
    fprintf('✓ All comparison visualizations generated and saved!\n');
    fprintf('  - FP16_vs_FP64_SideBySide.png\n');
    fprintf('  - FP16_vs_FP64_Classes.png\n');
    fprintf('  - FP16_vs_FP64_Heatmap.png\n');
    fprintf('  - FP16_vs_FP64_Normalized.png\n\n');
    
    clear S64_L S64_U Scls64_L Scls64_U bitIdxVec64;

end

% ============================ Global pooled ==============================
fprintf('\n=================================================================\n');
fprintf('  GLOBAL POOLED SUMMARY\n');
fprintf('=================================================================\n');

figure('Name','Pooled summary','Color','w');
G = exec_finalize(EXEC);
tl = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile; hold on; title('TPR pooled');
for i=1:numel(detNames)
    bar(i, G.TPR(i), 0.7, 'FaceColor', detCols(i,:));
    errorbar(i, G.TPR(i), G.TPR(i)-G.TPR_lo(i), G.TPR_hi(i)-G.TPR(i), 'k', 'LineWidth',1, 'HandleVisibility','off');
end
xlim([0.5 numel(detNames)+0.5]); ylim([0 1]); xticks(1:numel(detNames)); xticklabels(detNames); grid on;

nexttile; hold on; title('FPR pooled');
for i=1:numel(detNames)
    bar(i, G.FPR(i), 0.7, 'FaceColor', detCols(i,:));
    errorbar(i, G.FPR(i), G.FPR(i)-G.FPR_lo(i), G.FPR_hi(i)-G.FPR(i), 'k', 'LineWidth',1, 'HandleVisibility','off');
end
xlim([0.5 numel(detNames)+0.5]); ylim([0 1]); xticks(1:numel(detNames)); xticklabels(detNames); grid on;

nexttile; hold on; title('Refactorization pooled');
for i=1:numel(detNames)
    bar(i, G.Refac(i), 0.7, 'FaceColor', detCols(i,:));
    errorbar(i, G.Refac(i), G.Refac(i)-G.Refac_lo(i), G.Refac_hi(i)-G.Refac(i), 'k', 'LineWidth',1, 'HandleVisibility','off');
end
xlim([0.5 numel(detNames)+0.5]); ylim([0 1]); xticks(1:numel(detNames)); xticklabels(detNames); grid on;
maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI);

% ============================ Save outputs ===============================
if saveOutputs
    if saveCSV
        if exist('G','var'), save_all_exec_csv(G, detNames, outDir); end
    end
    if saveJSONMeta
        meta = struct();
        meta.time = datestr(now,'yyyy-mm-dd HH:MM:SS');
        meta.profile = PROFILE; meta.sizePreset = SIZE_PRESET; meta.seedMult = SEED_MULT;
        meta.detectors = string(detectorsWanted);
        meta.base = struct('m',m,'n',n,'wb',wb,'pinject',pinject,'q',q);
        meta.baseComparison = struct('nMatrices', nSeedsBaseComp, 'sizes', sizesToTest);
        meta.features = 'FP16_and_FP64_bit_analysis';
        meta.fp64enabled = doFP64BitSweeps;
        try
            fid = fopen(fullfile(outDir, 'meta.json'), 'w');
            fprintf(fid, '%s', jsonencode(meta));
            fclose(fid);
        catch
        end
    end
end

fprintf('\n=================================================================\n');
fprintf('  EXPERIMENT COMPLETE! ✓\n');
fprintf('=================================================================\n');
fprintf('Total wall-clock time: %.1f seconds (%.1f minutes)\n', toc(ticTotal), toc(ticTotal)/60);
fprintf('Output: %s\n', outDir);
fprintf('Profile: %s | Matrix: %dx%d\n', PROFILE, m, n);
fprintf('=================================================================\n\n');

% =========================================================================
% ==================== CORE FUNCTIONS ====================================
% =========================================================================

function S = sweep_bit_index_all_detectors(bitIdxVec, target, precision, A0, m, n, wb, useGPU, detList, aRel, aRat, aHy, aCrossCheck, q, blo, bhi, maskZeroRows, gammaMask, seeds, injCfgBase)
S.TPR = zeros(numel(detList), numel(bitIdxVec));
for i=1:numel(detList)
    det = detList{i}; TPR = zeros(numel(seeds), numel(bitIdxVec));
    for b = 1:numel(bitIdxVec)
        bidx = bitIdxVec(b);
        injCfg = injCfgBase; 
        injCfg.mode = 'bitflip'; 
        injCfg.target = target;
        injCfg.precision = precision;
        injCfg.bit.fixedIndices = bidx; 
        injCfg.bit.class = 'any'; 
        injCfg.bit.kFlips = 1;
        for s = 1:numel(seeds)
            injEvents = make_injection_schedule(m, n, wb, 1.0, seeds(s), injCfg);
            a = pick_alpha(det, aRel, aRat, aHy, aCrossCheck);
            R = run_one_detector(A0, m, n, wb, useGPU, det, a, aRel, q, blo, bhi, maskZeroRows, gammaMask, injEvents, false, false, injCfg);
            inj = [R.panel.injected]; flg = [R.panel.flagged];
            TPR(s,b) = safe_div(sum(inj & flg), sum(inj));
        end
    end
    S.TPR(i,:) = mean(TPR, 1, 'omitnan');
end
end

function S = sweep_bit_class_all_detectors(classes, target, precision, A0, m, n, wb, useGPU, detList, aRel, aRat, aHy, aCrossCheck, q, blo, bhi, maskZeroRows, gammaMask, seeds, injCfgBase)
S.TPR = zeros(numel(detList), numel(classes));
for i=1:numel(detList)
    det = detList{i}; TPR = zeros(numel(seeds), numel(classes));
    for c = 1:numel(classes)
        injCfg = injCfgBase; 
        injCfg.mode = 'bitflip'; 
        injCfg.target = target;
        injCfg.precision = precision;
        injCfg.bit.fixedIndices = []; 
        injCfg.bit.class = classes{c}; 
        injCfg.bit.kFlips = 1;
        for s = 1:numel(seeds)
            injEvents = make_injection_schedule(m, n, wb, 1.0, seeds(s), injCfg);
            a = pick_alpha(det, aRel, aRat, aHy, aCrossCheck);
            R = run_one_detector(A0, m, n, wb, useGPU, det, a, aRel, q, blo, bhi, maskZeroRows, gammaMask, injEvents, false, false, injCfg);
            inj = [R.panel.injected]; flg = [R.panel.flagged];
            TPR(s,c) = safe_div(sum(inj & flg), sum(inj));
        end
    end
    S.TPR(i,:) = mean(TPR, 1, 'omitnan');
end
end

function R = run_one_detector(A0, m, n, wb, useGPU, detector, ...
    alpha, alphaRelMultGlobal, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, ...
    injEvents, verbose, make3D, injCfg)

A = A0;
nb = n / wb;
panel(nb) = struct('injected',false,'flagged',false,'metric',NaN,'tau',NaN,'normMetric',NaN,'ok_npv',true);

for k = 1:nb
    row0 = (k-1)*wb + 1;    col0 = row0;
    rows_panel = row0:m;    cols_panel = col0:col0+wb-1;
    mP = numel(rows_panel);

    Pcpu = A(rows_panel, cols_panel);
    s16  = max(abs(Pcpu(:)));
    if s16 > 0, Pscaled = Pcpu / s16; else, Pscaled = Pcpu; end
    Plo = half(Pscaled);
    u_round = 2^-10;

    [Lh_h, Uh_h, piv_h, ok_half] = lu_half_pp(Plo);
    A(rows_panel,:) = A(rows_panel(piv_h), :);

    [Ld, Ud, ok] = npv(A(rows_panel, cols_panel));

    injected = false;
    if injEvents(k).inj
        injected = true;
        target = injCfg.target;
        if strcmp(target, 'either'), target = tern(rand < 0.5, 'L', 'U'); end
        ii = min(max(injEvents(k).ii,1), mP);
        jj = min(max(injEvents(k).jj,1), wb);
        
        if target=='U'
            ii = min(ii, wb);
            if injCfg.enforceUUpperTri, jj = max(jj, ii); end
        else
            if injCfg.enforceLLowerTri && ii <= wb, jj = min(jj, ii); end
        end
        
        switch injCfg.mode
            case 'signflip'
                if target=='L', Lh_h(ii,jj) = -Lh_h(ii,jj); else, Uh_h(ii,jj) = -Uh_h(ii,jj); end
            case 'bitflip'
                bitIdxVec = resolve_bit_indices(injCfg);
                if target=='L'
                    Lh_h(ii,jj) = flip_bits_scalar(Lh_h(ii,jj), bitIdxVec, injCfg.precision);
                else
                    Uh_h(ii,jj) = flip_bits_scalar(Uh_h(ii,jj), bitIdxVec, injCfg.precision);
                end
        end
    end

    Lh_wb = double(Lh_h(:,1:wb));
    Uh    = double(Uh_h) * s16;
    Uh_wb = Uh(1:wb,1:wb);
    
    v16 = sum(Uh_wb, 2);    c16 = Lh_wb * v16;
    v64 = sum(Ud,  2);      c64 = Ld * v64;

    flag=false; metric=NaN; tau=NaN; normMetric=NaN;
    
    switch lower(detector)
        case 'relative'
            num    = norm(c16 - c64, inf);
            denom  = max(norm(c64, inf), realmin);
            metric = num / denom;
            tau    = alpha * u_round;
            flag   = (~ok) || isnan(metric) || isinf(metric) || (metric > tau);
            normMetric = metric / max(tau, realmin);

        case 'ratio'
            d = sort(abs(c16 - c64), 'descend');
            if isempty(d), d1=0; d2=0; else, d1=d(1); d2=d(min(2,numel(d))); end
            denom = max(d2, realmin);
            metric = d1 / denom;
            tau    = alpha;
            flag   = (~ok) || isnan(metric) || isinf(metric) || (metric > tau);
            normMetric = metric / max(tau, realmin);

        case 'hybrid'
            [score, ~] = hybrid_score(Lh_wb, Uh_wb, Ld, Ud, wb, mP, u_round, q, alpha, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask);
            flag = (~ok) || (score > 1);
            metric = score; tau = 1; normMetric = score;

        case 'crosscheck'
            Ulow = Uh_wb; Uref = double(Ud(1:wb,1:wb)); e = ones(wb,1);
            
            col_rel = norm(Ulow*e - Uref*e, inf) / max(norm(Uref*e, inf), realmin);
            row_rel = norm((e.'*Ulow).' - (e.'*Uref).', inf) / max(norm((e.'*Uref).', inf), realmin);
            
            Llow = Lh_wb; Lref = Ld;
            L_rel = norm(Llow - Lref, 'fro') / max(norm(Lref, 'fro'), realmin);
            
            metric = max([row_rel, col_rel, L_rel]);
            tau    = alpha * u_round;
            flag   = (~ok) || isnan(metric) || isinf(metric) || (metric > tau);
            normMetric = metric / max(tau, realmin);

        case 'ensemble'
            num = norm(c16 - c64, inf);
            denom = max(norm(c64, inf), realmin);
            rel_metric = num / denom;
            tau_rel = alphaRelMultGlobal * u_round;
            s_rel = rel_metric / max(tau_rel, realmin);
            
            [s_hyb, ~] = hybrid_score(Lh_wb, Uh_wb, Ld, Ud, wb, mP, u_round, q, alpha, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask);
            
            score = max(s_rel, s_hyb);
            flag  = (~ok) || (score > 1);
            metric = score; tau = 1; normMetric = score;

        otherwise
            error('Unknown detector: %s', detector);
    end

    if flag
        [Ldf, Udf, p2] = lu(A(rows_panel, cols_panel), 'vector');
        A(rows_panel,:) = A(rows_panel(p2), :);
        Ld = Ldf(:, 1:wb); Ud = Udf(1:wb, 1:wb);
    end

    L11 = Ld(1:wb, 1:wb); U11 = Ud;
    A(row0:row0+wb-1, col0:col0+wb-1) = tril(L11, -1) + triu(U11, 0);
    
    if k < nb
        rows_A21 = (row0+wb):m; cols_A12 = (col0+wb):n;
        L21 = A(rows_A21, cols_panel) / U11;
        A(rows_A21, cols_panel) = L21;
        A12 = A(row0:row0+wb-1, cols_A12);
        U12 = L11 \ A12;
        A(row0:row0+wb-1, cols_A12) = U12;
        A(rows_A21, cols_A12) = A(rows_A21, cols_A12) - L21 * U12;
    end

    panel(k).injected   = injected;
    panel(k).flagged    = flag;
    panel(k).metric     = metric;
    panel(k).tau        = tau;
    panel(k).normMetric = normMetric;
    panel(k).ok_npv     = ok;
end

R.detector = detector;
R.panel    = panel;
end

function [score, details] = hybrid_score(Lh_wb, Uh_wb, Ld, Ud, wb, mP, u_round, q, alpha, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask)
Ld_top = double(Ld(1:wb, 1:wb));
Ud_wb  = double(Ud(1:wb, 1:wb));
Lh_top = double(Lh_wb(1:wb, 1:wb));

hasBottom = (mP > wb);
if hasBottom
    Ld_bot = double(Ld(wb+1:mP, 1:wb));
    Lh_bot = double(Lh_wb(wb+1:mP, 1:wb));
else
    Ld_bot = zeros(0, wb);
    Lh_bot = zeros(0, wb);
end

beta_lo = beta_lo_mult * u_round;
beta_hi = beta_hi_mult * u_round;
score = 0;
tiny64 = realmin('double');

for t = 1:q
    w = rademacher(wb);
    u64 = Ud_wb * w;
    u16 = Uh_wb * w;

    cU16 = Ld_top * u16;
    cU64 = Ld_top * u64;
    [rU, relU] = ratio_and_rel_fixed(cU16, cU64, tiny64, maskZeroRows, gammaMask);
    sU_hi  = relU / max(beta_hi, tiny64);
    sU_and = (rU / max(alpha, tiny64)) * (relU / max(beta_lo, tiny64));
    score  = max(score, max(sU_hi, sU_and));

    cL11_16 = Lh_top * u64;
    cL11_64 = Ld_top * u64;
    [rL11, relL11] = ratio_and_rel_fixed(cL11_16, cL11_64, tiny64, maskZeroRows, gammaMask);
    s11_hi  = relL11 / max(beta_hi, tiny64);
    s11_and = (rL11 / max(alpha, tiny64)) * (relL11 / max(beta_lo, tiny64));
    score   = max(score, max(s11_hi, s11_and));

    if hasBottom
        cL21_16 = Lh_bot * u64;
        cL21_64 = Ld_bot * u64;
        [rL21, relL21] = ratio_and_rel_fixed(cL21_16, cL21_64, tiny64, maskZeroRows, gammaMask);
        s21_hi  = relL21 / max(beta_hi, tiny64);
        s21_and = (rL21 / max(alpha, tiny64)) * (relL21 / max(beta_lo, tiny64));
        score   = max(score, max(s21_hi, s21_and));
    end
end
details = [];
end

function [r, rel] = ratio_and_rel_fixed(c16, c64, tiny, maskZeroRows, gammaMask)
c16 = double(c16(:));
c64 = double(c64(:));
diff = abs(c16 - c64);

if maskZeroRows
    scale = max(abs(c64));
    thr = gammaMask * max(scale, tiny);
    mask = (abs(c64) <= thr) & (diff <= thr * 100);
    diff(mask) = 0;
end

d = sort(diff, 'descend');
if isempty(d) || d(1) <= tiny
    r = 1;
elseif numel(d) >= 2
    r = d(1) / max(d(2), tiny);
else
    r = 1;
end

rel = max(diff) / max(max(abs(c64)), tiny);
end

function EXEC = exec_init(detNames)
EXEC.detNames = detNames;
nd = numel(detNames);
EXEC.TP = zeros(1,nd); EXEC.FP = zeros(1,nd);
EXEC.FN = zeros(1,nd); EXEC.TN = zeros(1,nd);
EXEC.REF = zeros(1,nd);
EXEC.FPclean = zeros(1,nd); EXEC.Nclean = zeros(1,nd);
EXEC.nRuns = 0;
end

function G = exec_finalize(EXEC)
TP=EXEC.TP; FP=EXEC.FP; FN=EXEC.FN; TN=EXEC.TN; REF=EXEC.REF;
P = TP + FN; N = FP + TN; T = P + N; Nclean = EXEC.Nclean; FPc = EXEC.FPclean;
alpha=0.05;
G.TPR = TP ./ max(P,1); [G.TPR_lo, G.TPR_hi] = binom_wilson_ci(TP, P, alpha);
G.FPR = FP ./ max(N,1); [G.FPR_lo, G.FPR_hi] = binom_wilson_ci(FP, N, alpha);
G.Refac = REF ./ max(T,1); [G.Refac_lo, G.Refac_hi] = binom_wilson_ci(REF, T, alpha);
G.FPRpf = FPc ./ max(Nclean,1); [G.FPRpf_lo, G.FPRpf_hi] = binom_wilson_ci(FPc, Nclean, alpha);
G.nRuns = EXEC.nRuns;
end

function alphaVal = pick_alpha(detName, aRel, aRat, aHy, aCross)
switch detName
    case 'relative',  alphaVal = aRel;
    case 'ratio',     alphaVal = aRat;
    case 'hybrid',    alphaVal = aHy;
    case 'crosscheck', alphaVal = aCross;
    case 'ensemble',  alphaVal = aHy;
    otherwise, error('Unknown detector %s', detName);
end
end

function alphaStar = calibrate_ratio_alpha(A0, m, n, wb, useGPU, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, targetFPR, nRuns, injCfg, alphaRelMult0)
injEventsNull = make_injection_schedule(m, n, wb, 0.0, 999, injCfg);
metrics = [];
for s = 1:nRuns
    R = run_one_detector(A0, m, n, wb, useGPU, 'ratio', 1.2, alphaRelMult0, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injEventsNull, false, false, injCfg);
    mvec = [R.panel.metric]; metrics = [metrics, mvec(~isnan(mvec))];
end
if isempty(metrics), alphaStar = 1.2; else, alphaStar = quantile(metrics, 1 - targetFPR); end
end

function alphaRelStar = calibrate_relative_alpha_mult(A0, m, n, wb, useGPU, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, targetFPR, nRuns, injCfg, alphaRelMult0)
injEventsNull = make_injection_schedule(m, n, wb, 0.0, 1337, injCfg);
u = 2^-10; metrics = [];
for s = 1:nRuns
    R = run_one_detector(A0, m, n, wb, useGPU, 'relative', 1.0, alphaRelMult0, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injEventsNull, false, false, injCfg);
    mvec = [R.panel.metric]; metrics = [metrics, mvec(~isnan(mvec))];
end
if isempty(metrics), alphaRelStar = 16;
else
    tauStar = quantile(metrics, 1 - targetFPR);
    alphaRelStar = max(tauStar / u, 1.0);
end
end

function alphaCrossCheckStar = calibrate_crosscheck_alpha_mult(A0, m, n, wb, useGPU, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, targetFPR, nRuns, injCfg, alphaRelMult0)
injEventsNull = make_injection_schedule(m, n, wb, 0.0, 1777, injCfg);
u = 2^-10; metrics = [];
for s = 1:nRuns
    R = run_one_detector(A0, m, n, wb, useGPU, 'crosscheck', 1.0, alphaRelMult0, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injEventsNull, false, false, injCfg);
    mvec = [R.panel.metric]; metrics = [metrics, mvec(~isnan(mvec))];
end
if isempty(metrics), alphaCrossCheckStar = 16;
else
    tauStar = quantile(metrics, 1 - targetFPR);
    alphaCrossCheckStar = max(tauStar / u, 1.0);
end
end

function [betaLoStar, betaHiStar, sStar, baseFPR] = calibrate_hybrid_beta_scale(A0, m, n, wb, useGPU, alphaHybridRatio, beta_lo_mult, beta_hi_mult, q, maskZeroRows, gammaMask, targetFPR, nRuns, injCfg, alphaRelMult0)
injEventsNull = make_injection_schedule(m, n, wb, 0.0, 2024, injCfg);
scores = []; flags = 0; total = 0;
for s = 1:nRuns
    R = run_one_detector(A0, m, n, wb, useGPU, 'hybrid', alphaHybridRatio, alphaRelMult0, q, beta_lo_mult, beta_hi_mult, maskZeroRows, gammaMask, injEventsNull, false, false, injCfg);
    svec = [R.panel.normMetric];
    scores = [scores, svec(~isnan(svec))];
    flags  = flags + sum(svec > 1); total = total + numel(svec);
end
if isempty(scores), sStar = 1.0; baseFPR = 0.0;
else, sStar = quantile(scores, 1 - targetFPR); baseFPR = flags / max(total, 1); end
betaLoStar = beta_lo_mult * max(sStar, realmin);
betaHiStar = beta_hi_mult * max(sStar, realmin);
end

function injEvents = make_injection_schedule(m, n, wb, pinject, seed, injCfg)
rng(seed); nb = n / wb;
injEvents(nb) = struct('inj', false, 'ii', 1, 'jj', 1);
for k = 1:nb
    mP = m - (k-1)*wb;
    injEvents(k).inj = (rand < pinject);
    injEvents(k).ii  = randi(mP);
    injEvents(k).jj  = randi(wb);
end
end

function bitIdxVec = resolve_bit_indices(injCfg)
if isfield(injCfg, 'precision')
    precision = injCfg.precision;
else
    precision = 'fp16';
end

if ~isempty(injCfg.bit.fixedIndices)
    bitIdxVec = unique(injCfg.bit.fixedIndices(:)).';
    if strcmp(precision, 'fp16')
        assert(all(bitIdxVec>=0 & bitIdxVec<=15));
    else
        assert(all(bitIdxVec>=0 & bitIdxVec<=63));
    end
else
    if strcmp(precision, 'fp16')
        switch injCfg.bit.class
            case 'frac', pool = 0:9;
            case 'exp',  pool = 10:14;
            case 'sign', pool = 15;
            case 'any',  pool = 0:15;
        end
    else
        switch injCfg.bit.class
            case 'frac', pool = 0:51;
            case 'exp',  pool = 52:62;
            case 'sign', pool = 63;
            case 'any',  pool = 0:63;
        end
    end
    
    k = max(1, min(numel(pool), injCfg.bit.kFlips));
    if numel(pool) == 1
        bitIdxVec = pool;
    else
        idx = randperm(numel(pool), k);
        bitIdxVec = sort(pool(idx));
    end
end
end

function y = flip_bits_scalar(x, bitIdxVec, precision)
if strcmp(precision, 'fp16')
    u = typecast(x, 'uint16'); 
    mask = uint16(0);
    for b = bitIdxVec
        mask = bitor(mask, bitshift(uint16(1), b)); 
    end
    u2 = bitxor(u, mask); 
    y = typecast(u2, 'half');
else
    u = typecast(double(x), 'uint64');
    mask = uint64(0);
    for b = bitIdxVec
        mask = bitor(mask, bitshift(uint64(1), b));
    end
    u2 = bitxor(u, mask);
    y = typecast(u2, 'double');
end
end

function v = safe_div(a,b)
if b<=0, v = NaN; else, v = a/b; end
end

function tf = tern(c,a,b)
if c, tf = a; else, tf = b; end
end

function [Lh, Uh, piv, ok] = lu_half_pp(Ah)
[mP, wb] = size(Ah);
Rh = Ah; piv = double(1:mP).'; ok = true;
oneH = half(1); tinyH = half(realmin('half'));
for k = 1:wb
    imax = k; maxv = abs(Rh(k,k));
    for i = k+1:mP
        ai = abs(Rh(i,k));
        if ai > maxv, maxv = ai; imax = i; end
    end
    if imax ~= k
        tmp = Rh(k,:); Rh(k,:) = Rh(imax,:); Rh(imax,:) = tmp;
        tmpi = piv(k); piv(k) = piv(imax); piv(imax) = tmpi;
    end
    pivval = Rh(k,k);
    if abs(pivval) <= tinyH || ~isfinite(double(pivval)), ok = false; continue; end
    for i = k+1:mP
        Rh(i,k) = Rh(i,k) / pivval;
        if k < wb
            lik = Rh(i,k);
            for j = k+1:wb
                Rh(i,j) = Rh(i,j) - lik * Rh(k,j);
            end
        end
    end
end
Uh = triu_half(Rh(1:wb, 1:wb), 0);
Lh = tril_half(Rh(:, 1:wb), -1);
for k = 1:wb, Lh(k,k) = oneH; end
end

function H = triu_half(Ah, k)
if nargin < 2, k = 0; end
[m, n] = size(Ah);
H = half(zeros(m,n,'like',Ah));
for i = 1:m
    jstart = max(1, i + k);
    if jstart <= n, H(i, jstart:n) = Ah(i, jstart:n); end
end
end

function H = tril_half(Ah, k)
if nargin < 2, k = 0; end
[m, n] = size(Ah);
H = half(zeros(m,n,'like',Ah));
for i = 1:m
    jend = min(n, i + k);
    if jend >= 1, H(i, 1:jend) = Ah(i, 1:jend); end
end
end

function [L,U,ok] = npv(A)
[mP, wb] = size(A);
U = zeros(wb,wb,'like',A); L = eye(mP,wb,'like',A); ok = true;
R = A; tiny = cast(realmin('double'), 'like', U);
for k = 1:wb
    U(k,k:wb) = R(k,k:wb);
    if abs(U(k,k)) <= tiny, ok = false; return; end
    if k < mP
        L(k+1:mP,k) = R(k+1:mP,k) ./ U(k,k);
        R(k+1:mP,k+1:wb) = R(k+1:mP,k+1:wb) - L(k+1:mP,k)*U(k,k+1:wb);
    end
end
end

function w = rademacher(n)
w = double(2 * randi([0,1], n, 1) - 1);
end

function decorate_bitaxis_fp16()
xline(9.5,'k--','frac→exp'); xline(14.5,'k--','exp→sign'); xticks(0:15);
end

function decorate_bitaxis_fp64()
xline(51.5,'k--','frac→exp'); xline(62.5,'k--','exp→sign'); 
xticks([0 10 20 30 40 51 52 62 63]);
end

function plot_panel_timeline(runs, detNames)
figure('Name','Panel Timeline (First Run)','Color','w');
nb = numel(runs{1}.panel);
vals = zeros(numel(runs), nb);
for i = 1:numel(runs)
    inj = [runs{i}.panel.injected]; flg = [runs{i}.panel.flagged];
    vals(i, :) = 0;
    vals(i, ~inj & flg) = 1;
    vals(i, inj  & ~flg) = 2;
    vals(i, inj  & flg) = 3;
end
imagesc(vals); axis tight;
colormap([0.85 0.85 0.85; 1.00 0.60 0.00; 0.80 0.10 0.10; 0.10 0.60 0.10]);
caxis([-0.5 3.5]);
colorbar('Ticks',0:3,'TickLabels',{'OK','FP','Miss','TP'});

% Fix: Use integer panel indices only
xticks(1:nb);  
xticklabels(arrayfun(@num2str, 1:nb, 'UniformOutput', false));
xlabel('Panel index'); 

ylabel('Detector');
yticks(1:numel(detNames)); 
yticklabels(detNames);
title('Per-panel Outcomes (Sample Run)'); 
grid on;
end

function plot_metric_separation_pooled(allRuns, detNames, detCols, doLogY, logFloor, logYMaxQ, showMinorGrid, limitTicks, maxTicks, annotate, captionBottom)
haveBoxchart = (exist('boxchart','class')==8) || (exist('boxchart','file')==2);
figure('Name','Metric Separation (Pooled)','Color','w');
tiledlayout(3,1,'Padding','compact','TileSpacing','compact');

nd = numel(allRuns);
nRuns = numel(allRuns{1});

axTop = nexttile; hold(axTop,'on');
for i = 1:nd
    pooledData = [];
    for r = 1:nRuns
        nm = [allRuns{i}(r).panel.normMetric];
        inj = [allRuns{i}(r).panel.injected];
        pooledData = [pooledData, nm(~inj)];
    end
    if isempty(pooledData), pooledData = NaN; end
    
    if haveBoxchart
        boxchart(axTop, i * ones(1,numel(pooledData)), pooledData, 'BoxFaceColor', detCols(i,:));
    else
        boxplot(axTop, pooledData, i*ones(size(pooledData)), 'Positions', i, 'Colors', detCols(i,:), 'Symbol','.');
    end
end
yline(axTop,1,'r--','Threshold');
xlim(axTop,[0.5 nd+0.5]); xticks(axTop,1:nd); xticklabels(axTop,detNames);
ylabel(axTop,'Score'); title(axTop,sprintf('Non-injected panels (n=%d runs pooled)', nRuns));
grid(axTop,'on'); hold(axTop,'off');

axMid = nexttile; hold(axMid,'on');
allPos = [];
for i = 1:nd
    pooledData = [];
    for r = 1:nRuns
        nm = [allRuns{i}(r).panel.normMetric];
        inj = [allRuns{i}(r).panel.injected];
        pooledData = [pooledData, nm(inj)];
    end
    if isempty(pooledData), pooledData = NaN; end
    
    dp = pooledData(isfinite(pooledData) & pooledData > 0);
    allPos = [allPos, dp];
    
    plotData = pooledData;
    maskBad = ~isfinite(plotData) | (plotData <= 0);
    plotData(maskBad) = logFloor;
    
    if haveBoxchart
        boxchart(axMid, i * ones(1,numel(plotData)), plotData, 'BoxFaceColor', detCols(i,:));
    else
        boxplot(axMid, plotData, i*ones(size(plotData)), 'Positions', i, 'Colors', detCols(i,:), 'Symbol','.');
    end
end
yline(axMid,1,'r--','Threshold');
xlim(axMid,[0.5 nd+0.5]); xticks(axMid,1:nd); xticklabels(axMid,detNames);
ylabel(axMid,'Score'); title(axMid,sprintf('Injected panels (n=%d runs pooled)', nRuns));
grid(axMid,'on');

if doLogY
    set(axMid, 'YScale', 'log');
    if ~isempty(allPos)
        ylow = max(min(allPos), logFloor);
        ymax = quantile(allPos, logYMaxQ);
        if ~(isfinite(ymax) && ymax > ylow), ymax = 10 * ylow; end
        ylim(axMid,[ylow, ymax]);
    end
    if showMinorGrid, set(axMid, 'YMinorGrid','on'); end
end
hold(axMid,'off');

axCap = nexttile; axis(axCap,'off');
if annotate
    text(axCap, 0, 0.95, ['Score s: flag if s>1. Pooled across ', num2str(nRuns), ' independent matrices.'], ...
        'Units','normalized', 'VerticalAlignment','top', 'FontSize',8);
end
end

function delays = detection_delays(panel)
inj = [panel.injected]; flg = [panel.flagged];
idxInj = find(inj); idxFlg = find(flg);
delays = [];
for k = idxInj
    j = idxFlg(find(idxFlg>=k, 1, 'first'));
    if ~isempty(j), delays(end+1) = j - k; end
end
if isempty(delays), delays = 0; end
end

function [fpr_pf, Nclean, FPclean] = compute_propagation_free_FPR_single(panel)
inj = [panel.injected]; flg = [panel.flagged];
isClean = true(size(inj)); contaminated = false;
for k=1:numel(inj)
    isClean(k) = ~contaminated;
    if inj(k) && ~flg(k), contaminated = true; end
    if flg(k), contaminated = false; end
end
mask = ~inj & isClean;
Nclean = sum(mask); FPclean = sum(mask & flg);
fpr_pf = safe_div(FPclean, Nclean);
end

function shaded_CI(x, lo, hi, color, alpha)
if nargin < 5, alpha = 0.15; end
x = x(:).'; lo = lo(:).'; hi = hi(:).';
X = [x, fliplr(x)]; Y = [lo, fliplr(hi)];
patch('XData',X,'YData',Y,'FaceColor',color,'FaceAlpha',alpha,'EdgeColor','none', 'HandleVisibility','off');
end

function ensure_dir(d)
if ~exist(d,'dir'), mkdir(d); end
end

function maybe_export_and_close(outDir, saveOutputs, saveFigures, figFormats, figDPI)
if saveOutputs && saveFigures
    figs = findobj('Type','figure');
    for k = 1:numel(figs)
        fh = figs(k); 
        nm = get(fh, 'Name');
        if isempty(nm), nm = sprintf('Figure_%d', double(fh.Number)); end
        nm = regexprep(nm, '[^\w\-]', '_');
        base = fullfile(outDir, nm);
        
        for f = 1:numel(figFormats)
            fmt = lower(figFormats{f});
            try
                switch fmt
                    case 'png'
                        exportgraphics(fh, [base '.png'], 'Resolution', figDPI);
                    case 'eps'
                        exportgraphics(fh, [base '.eps'], 'ContentType', 'vector');
                    case 'pdf'
                        exportgraphics(fh, [base '.pdf'], 'ContentType', 'vector');
                    case 'svg'
                        print(fh, [base '.svg'], '-dsvg', '-vector');
                    otherwise
                        warning('Unsupported format: %s', fmt);
                end
            catch ME
                warning('Failed to export %s as %s: %s', nm, fmt, ME.message);
            end
        end
    end
end
close all;
end

function [lo, hi] = binom_wilson_ci(k, n, alpha)
if nargin < 3, alpha = 0.05; end
z = sqrt(2) * erfcinv(alpha);
k = double(k); n = double(n);
phat = zeros(size(k)); m = (n>0);
phat(m) = k(m)./n(m);
z2 = z*z; den = 1 + z2./n;
center = (phat + z2./(2*n)) ./ den;
half = (z .* sqrt((phat.*(1-phat))./n + z2./(4*n.^2))) ./ den;
lo = max(0, center - half); hi = min(1, center + half);
lo(~m) = NaN; hi(~m) = NaN;
end

function save_all_exec_csv(G, detNames, outDir)
T = table(detNames(:), G.TPR(:), G.TPR_lo(:), G.TPR_hi(:), G.FPR(:), G.FPR_lo(:), G.FPR_hi(:), ...
    G.Refac(:), G.Refac_lo(:), G.Refac_hi(:), G.FPRpf(:), G.FPRpf_lo(:), G.FPRpf_hi(:), ...
    'VariableNames', {'detector','TPR','TPR_lo','TPR_hi','FPR','FPR_lo','FPR_hi', ...
        'Refac','Refac_lo','Refac_hi','FPRpf','FPRpf_lo','FPRpf_hi'});
writetable(T, fullfile(outDir, 'all_exec_pooled.csv'));
end

function PS = sweep_pinject_all_detectors_extended(pinjectVec, nSeedsVec, seedBase, ...
    A0, m, n, wb, useGPU, detList, aRel, aRat, aHy, aCrossCheck, q, blo, bhi, maskZeroRows, gammaMask, injCfg, doPropFree)
nd = numel(detList); np = numel(pinjectVec);
PS.TPR_mean=zeros(nd,np); PS.TPR_lo=PS.TPR_mean; PS.TPR_hi=PS.TPR_mean;
PS.FPR_mean=zeros(nd,np); PS.FPR_lo=PS.FPR_mean; PS.FPR_hi=PS.FPR_mean;
PS.FPRpf_mean=zeros(nd,np); PS.FPRpf_lo=PS.FPRpf_mean; PS.FPRpf_hi=PS.FPRpf_mean;
alpha = 0.05;
for j = 1:np
    pinj = pinjectVec(j); nSeeds = nSeedsVec(j);
    fprintf('  pinject=%.2f (%d seeds)...', pinj, nSeeds);
    TP=zeros(nd,1); FP=zeros(nd,1); FN=zeros(nd,1); TN=zeros(nd,1); FPpf=zeros(nd,1); Npf=zeros(nd,1);
    for s = 1:nSeeds
        injEvents = make_injection_schedule(m, n, wb, pinj, seedBase + 1000*j + s, injCfg);
        runs = cell(1, nd);
        for i=1:nd
            a = pick_alpha(detList{i}, aRel, aRat, aHy, aCrossCheck);
            runs{i} = run_one_detector(A0, m, n, wb, useGPU, detList{i}, a, aRel, q, blo, bhi, maskZeroRows, gammaMask, injEvents, false, false, injCfg);
        end
        for i=1:nd
            p = runs{i}.panel; inj=[p.injected]; flg=[p.flagged];
            TP(i) = TP(i) + sum(inj & flg); FP(i) = FP(i) + sum(~inj & flg);
            FN(i) = FN(i) + sum(inj & ~flg); TN(i) = TN(i) + sum(~inj & ~flg);
            if doPropFree
                [~, Nclean, FPclean] = compute_propagation_free_FPR_single(p);
                FPpf(i)= FPpf(i) + FPclean; Npf(i) = Npf(i) + Nclean;
            end
        end
    end
    P = TP + FN; N = FP + TN;
    PS.TPR_mean(:,j) = TP ./ max(P,1); PS.FPR_mean(:,j) = FP ./ max(N,1);
    [PS.TPR_lo(:,j), PS.TPR_hi(:,j)] = binom_wilson_ci(TP, P, alpha);
    [PS.FPR_lo(:,j), PS.FPR_hi(:,j)] = binom_wilson_ci(FP, N, alpha);
    if doPropFree
        PS.FPRpf_mean(:,j) = FPpf ./ max(Npf,1);
        [PS.FPRpf_lo(:,j), PS.FPRpf_hi(:,j)] = binom_wilson_ci(FPpf, Npf, alpha);
    end
    fprintf(' done.\n');
end
end

function SS = size_sweep_with_CIs(sizesVec, wb, A0_ref, seedA, useGPU, ...
    detList, q, beta_lo_base, beta_hi_base, maskZeroRows, gammaMask, ...
    targetFPR, nNullRuns, pinject, nSeeds, seedStart, injCfg, guardGE1, alphaRelMult0)
nd = numel(detList); ns = numel(sizesVec);
SS.TPR_mean=zeros(nd,ns); SS.TPR_lo=SS.TPR_mean; SS.TPR_hi=SS.TPR_mean;
SS.FPR_mean=zeros(nd,ns); SS.FPR_lo=SS.FPR_mean; SS.FPR_hi=SS.FPR_mean;
SS.Refac_mean=zeros(nd,ns); SS.Refac_lo=SS.Refac_mean; SS.Refac_hi=SS.Refac_mean;
alphaCI = 0.05;
for t = 1:ns
    n = sizesVec(t); m = n; assert(mod(n,wb)==0);
    rng(seedA + t); A0 = randn(m,n,'double');
    fprintf('Size n=%d: calibrating...', n);
    
    alphaRatio = calibrate_ratio_alpha(A0, m, n, wb, useGPU, q, beta_lo_base, beta_hi_base, maskZeroRows, gammaMask, targetFPR, nNullRuns, injCfg, alphaRelMult0);
    alphaRel   = calibrate_relative_alpha_mult(A0, m, n, wb, useGPU, q, beta_lo_base, beta_hi_base, maskZeroRows, gammaMask, targetFPR, nNullRuns, injCfg, alphaRelMult0);
    alphaCrossCheck  = calibrate_crosscheck_alpha_mult(A0, m, n, wb, useGPU, q, beta_lo_base, beta_hi_base, maskZeroRows, gammaMask, targetFPR, nNullRuns, injCfg, alphaRelMult0);
    [~, ~, sStar, ~] = calibrate_hybrid_beta_scale(A0, m, n, wb, useGPU, alphaRatio, beta_lo_base, beta_hi_base, q, maskZeroRows, gammaMask, targetFPR, nNullRuns, injCfg, alphaRelMult0);
    sUse = sStar; if guardGE1, sUse = max(1, sStar); end
    bLo = beta_lo_base * sUse; bHi = beta_hi_base * sUse;
    fprintf(' running %d seeds...', nSeeds);

    TP=zeros(nd,1); FP=zeros(nd,1); FN=zeros(nd,1); TN=zeros(nd,1); REF=zeros(nd,1);
    for s = 1:nSeeds
        injEvents = make_injection_schedule(m, n, wb, pinject, seedStart + 1000*t + s, injCfg);
        runs = cell(1, nd);
        for i=1:nd
            det = detList{i};
            a = pick_alpha(det, alphaRel, alphaRatio, alphaRatio, alphaCrossCheck);
            runs{i} = run_one_detector(A0, m, n, wb, useGPU, det, a, alphaRel, q, bLo, bHi, maskZeroRows, gammaMask, injEvents, false, false, injCfg);
        end
        for i=1:nd
            p = runs{i}.panel; inj=[p.injected]; flg=[p.flagged];
            TP(i) = TP(i) + sum(inj & flg); FP(i) = FP(i) + sum(~inj & flg);
            FN(i) = FN(i) + sum(inj & ~flg); TN(i) = TN(i) + sum(~inj & ~flg);
            REF(i)= REF(i)+ sum(flg);
        end
    end
    P = TP + FN; N = FP + TN;
    for i=1:nd
        SS.TPR_mean(i,t)   = TP(i) / max(P(i),1);
        SS.FPR_mean(i,t)   = FP(i) / max(N(i),1);
        SS.Refac_mean(i,t) = REF(i)/ max(P(i)+N(i),1);
    end
    [SS.TPR_lo(:,t), SS.TPR_hi(:,t)] = binom_wilson_ci(TP, P, alphaCI);
    [SS.FPR_lo(:,t), SS.FPR_hi(:,t)] = binom_wilson_ci(FP, N, alphaCI);
    [SS.Refac_lo(:,t), SS.Refac_hi(:,t)] = binom_wilson_ci(REF, P+N, alphaCI);
    fprintf(' done.\n');
    
    clear A0 runs injEvents;
end
end
