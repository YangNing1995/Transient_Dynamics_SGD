% Fig4 revised: Analytical model results
%   Panel A: Effective loss landscape (unchanged)
%   Panel B: Simulation P_flat(y) with y_freeze markers (no NESS overlay)
%   Panel C: Paradox plot — static prediction vs actual outcome
% =========================================================================

%% Paths and settings
script_dir = fileparts(mfilename('fullpath'));
repo_root  = fullfile(script_dir, '..', '..');
out_dir    = fullfile(repo_root, 'PRX_revison', 'Figures_revision', 'Fig4_analytic_model');
export_res = 600;

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% --- Model parameters ---
x1 = 0.8;  x2 = 0.65;  x0 = 1;  f0 = 1;
y_b = 30;  y_f = 20;   y_d = 100;  L_d = 1;

% --- Component functions ---
g1 = @(y) 2*x1.*y./(y + y_b);
g2 = @(y) 2*x2.*y./(y + y_b);
f1 = @(y) f0.*(x1.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
f2 = @(y) f0.*(x2.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
L0 = @(y) L_d.*exp(-y/y_d) + x0^2/f0;
L  = @(x, y) arrayfun(@(xx) ...
        L0(y) + (xx>=0)*xx.*(xx - g1(y))./f1(y) + ...
        (xx<0)*xx.*(xx + g2(y))./f2(y), x);

gamma = x1^2 / x2^2;   % flatness ratio (~1.515)
P_eq  = 1 / (1 + 1/gamma);  % equilibrium P_flat (~0.60)

%% ===== Panel A: Effective loss landscape (unchanged) ====================
y_fixed = 40;
x_vals  = linspace(-1.0, 1.2, 500);
L_vals  = L(x_vals, y_fixed);

f1_val = f1(y_fixed);  f2_val = f2(y_fixed);
L_eff  = zeros(size(x_vals));
for k = 1:length(x_vals)
    xx = x_vals(k);
    if xx >= 0
        r = sqrt(f1_val / f2_val);
    else
        r = sqrt(f2_val / f1_val);
    end
    L_eff(k) = r * L(xx, y_fixed) + (1 - r) * L0(y_fixed);
end

figA = figure('Position', [200 200 600 400]);
hold on;
plot(x_vals, L_vals,              'b-', 'LineWidth', 2.5, ...
    'DisplayName', '$\mathcal{L}(x,y)$');
plot(x_vals, L_eff,               'r-', 'LineWidth', 2.5, ...
    'DisplayName', '$\mathcal{L}_\mathrm{eff}(x,y)$');
plot(x_vals, L_eff - L_vals+1.3,  'k-', 'LineWidth', 2.5, ...
    'DisplayName', '$\mathcal{L}_\mathrm{SGD}(x,y)$');
xlabel('$x$', 'FontSize', 18, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('Loss', 'FontSize', 18);
xlim([-0.8 1.2]);
legend('Location', 'best', 'Interpreter', 'latex'); legend boxoff;
box off; grid off; axis off;
set(gca, 'FontSize', 14, 'LineWidth', 1.2, 'FontName', 'Times New Roman');
save_panel(figA, fullfile(out_dir, 'Fig4A_effective_loss.png'), export_res);


%% ===== Panel B: Simulation P_flat(y) + y_freeze markers ================
%  No NESS overlay — just the dynamics and where they freeze.

% --- Load simulation data ------------------------------------------------
file_list = {
    fullfile(repo_root, 'Two_valleys_model', 'P_right_eta0.010_sigma0.100.mat'), ...
    fullfile(repo_root, 'Two_valleys_model', 'P_right_eta0.010_sigma0.050.mat'), ...
    fullfile(repo_root, 'Two_valleys_model', 'P_right_eta0.010_sigma0.010.mat')
};

n_files    = numel(file_list);
y_data     = cell(n_files, 1);
P_sim_data = cell(n_files, 1);
DS_vals    = zeros(n_files, 1);
yf_est     = zeros(n_files, 1);   % estimated y_freeze
Pf_est     = zeros(n_files, 1);   % P_flat at y_freeze

for i = 1:n_files
    if ~exist(file_list{i}, 'file')
        warning('File not found: %s', file_list{i});
        continue
    end
    d = load(file_list{i});
    DS_vals(i)    = d.learning_rate * d.noise_strength;
    y_data{i}     = d.mean_y;
    P_sim_data{i} = d.P_right_simulation;

    % --- Estimate y_freeze: y where P_flat reaches 90% of its total rise --
    yv = d.mean_y;
    Pv = d.P_right_simulation;
    N  = length(Pv);
    win    = max(50, floor(N/20));
    Ps     = movmean(Pv, win);
    n_tail = max(10, floor(N/10));
    P_end   = mean(Ps(end-n_tail+1 : end));
    P_start = Ps(1);

    if abs(P_end - P_start) < 0.005
        idx = N;
    else
        target = P_start + 0.90 * (P_end - P_start);
        idx = find(Ps >= target, 1, 'first');
        if isempty(idx), idx = N; end
    end
    yf_est(i) = yv(idx);
    Pf_est(i) = Ps(idx);
end

% --- Colours & labels ----------------------------------------------------
clr = [0.00 0.45 0.74;    % blue   — largest noise
       0.85 0.33 0.10;    % orange — medium noise
       0.47 0.67 0.19];   % green  — smallest noise

ds_lab = {'$\Delta_S=10^{-3}$', ...
          '$\Delta_S=5{\times}10^{-4}$', ...
          '$\Delta_S=10^{-4}$'};

% --- Draw ----------------------------------------------------------------
figB = figure('Units','points','PaperUnits','points','Position',[100 100 500 400]);
hold on;

valid = find(~cellfun(@isempty, y_data));

% 1) Simulation curves (solid)
for i = reshape(valid, 1, [])
    plot(y_data{i}, P_sim_data{i}, '-', ...
        'Color', [clr(i,:) 0.6], 'LineWidth', 2.0, ...
        'DisplayName', ds_lab{i});
end

% 2) Equilibrium reference line
plot([1 20], [P_eq P_eq], '-.', ...
    'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, ...
    'HandleVisibility', 'off');
text(1.15, P_eq - 0.03, sprintf('$P_\\mathrm{flat}^\\mathrm{eq}=%.2f$', P_eq), ...
    'Interpreter', 'latex', 'FontSize', 14, 'Color', [0.4 0.4 0.4]);

% 3) y_freeze markers (pentagram) + vertical dashed guides
for i = reshape(valid, 1, [])
    % Vertical guide from bottom axis to the freeze point
    plot([yf_est(i) yf_est(i)], [0.5 Pf_est(i)], '--', ...
        'Color', [clr(i,:) 0.5], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
    % Star marker at the freeze point
    hv = 'off';
    if i == valid(1), hv = 'on'; end   % one legend entry for all stars
    plot(yf_est(i), Pf_est(i), 'p', ...
        'MarkerSize', 18, 'MarkerFaceColor', clr(i,:), ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.8, ...
        'HandleVisibility', hv, 'DisplayName', '$y_\mathrm{freeze}$');
end

xlabel('$y$', 'Interpreter', 'LaTeX', 'FontSize', 22);
ylabel('$P_\mathrm{flat}$', 'Interpreter', 'LaTeX', 'FontSize', 22);
ylim([0.5 1.02]);  xlim([1 20]);  yticks([0.5 0.75 1]);
legend('Location', 'southeast', 'Interpreter', 'LaTeX', ...
       'Box', 'on', 'FontSize', 13);
grid on;  box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 20, 'XScale', 'log');
save_panel(figB, fullfile(out_dir, 'Fig4B_Pflat_vs_y.png'), export_res);


%% ===== Panel C: Paradox plot ============================================
%  Static (fixed-y) prediction DECREASES with Delta_S,
%  but the actual simulation outcome INCREASES with Delta_S.
%  This contrast is the paradox resolved by transient freezing.

% --- 1. Compute NESS at one representative fixed y ----------------------
y_ref = 4;                           % chosen so transition falls mid-range
DS_smooth = logspace(-4.5, -1.5, 300);
P_ness_curve = zeros(size(DS_smooth));
for jd = 1:numel(DS_smooth)
    rl = NESS_escape_rate('left',  y_ref, f1, f2, g1, g2, L, DS_smooth(jd));
    rr = NESS_escape_rate('right', y_ref, f1, f2, g1, g2, L, DS_smooth(jd));
    P_ness_curve(jd) = rl / (rr + rl);
end

% --- 2. Load simulation final P_flat from phase diagram ------------------
pd = load(fullfile(repo_root, 'Two_valleys_model', ...
          'result_PhsDgm_Ld1_yd100_x10.8_x20.65.mat'));

% --- 3. Draw -------------------------------------------------------------
figC = figure('Units','points','PaperUnits','points','Position',[100 100 500 400]);
hold on;

% (a) NESS curve — static prediction (thick dashed)
plot(DS_smooth, P_ness_curve, '--', ...
    'Color', [0.3 0.3 0.3], 'LineWidth', 3, ...
    'DisplayName', sprintf('Steady-state (fixed $y{=}%d$)', y_ref));

% (b) Equilibrium reference
plot(DS_smooth([1 end]), [P_eq P_eq], ':', ...
    'Color', [0.65 0.65 0.65], 'LineWidth', 1.5, ...
    'DisplayName', sprintf('$P_\\mathrm{flat}^\\mathrm{eq}=%.2f$', P_eq));

% (c) Simulation outcome — lines with markers for three eta values
eta_indices = [1, 5, 10];
sim_clr = [0.12 0.47 0.71;   % blue
           0.20 0.62 0.17;   % green
           0.84 0.15 0.16];  % red
mk = {'o','s','^'};

for ie = 1:numel(eta_indices)
    idx   = eta_indices(ie);
    eta_v = pd.learning_rate_list(idx);
    DS_line = eta_v .* pd.noise_strength_list;
    Pf_line = pd.final_position(idx, :);
    dv_line = pd.divergence_marker(idx, :);
    Pf_line(dv_line > 0.1) = NaN;

    plot(DS_line, Pf_line, ['-' mk{ie}], ...
        'Color', sim_clr(ie,:), ...
        'MarkerFaceColor', sim_clr(ie,:), ...
        'MarkerEdgeColor', 'w', ...
        'MarkerSize', 8, 'LineWidth', 1.8, ...
        'DisplayName', sprintf('Sim. ($\\eta{=}%.2g$)', eta_v));
end

xlabel('$\Delta_S = \eta\sigma$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$P_\mathrm{flat}$',       'Interpreter', 'latex', 'FontSize', 20);
ylim([0.45 1.05]);  yticks([0.5 0.75 1]);
xlim([5e-5 2e-2]);
set(gca, 'XScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 18);
legend('Location', 'southeast', 'Interpreter', 'LaTeX', ...
       'FontSize', 11, 'Box', 'on');
grid on;  box on;
save_panel(figC, fullfile(out_dir, 'Fig4C_paradox.png'), export_res);

fprintf('\nExported revised Fig4 panels to:\n  %s\n', out_dir);


%% ===== Helper functions =================================================

function save_panel(fig, output_file, resolution)
    set(fig, 'Color', 'w', 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
    set(findall(fig, 'Type', 'axes'), 'Color', 'w');
    drawnow;
    print(fig, output_file, '-dpng', sprintf('-r%d', resolution));
    fprintf('  Saved %s\n', output_file);
end

function rate = NESS_escape_rate(x_start, y_trajectory, f1, f2, g1, g2, Loss, Delta_s)
    rate = zeros(1, length(y_trajectory));
    for t_idx = 1:length(y_trajectory)
        y_val = y_trajectory(t_idx);
        if x_start == "right"
            f_val     = f1(y_val);
            L_min     = Loss(g1(y_val)/2, y_val);
            L_barrier = Loss(0, y_val);
            rate(t_idx) = 1 / ((pi/2) * f_val * ...
                erfi(sqrt((L_barrier - L_min) / (2*Delta_s/f_val))));
        else
            f_val     = f2(y_val);
            L_min     = Loss(-g2(y_val)/2, y_val);
            L_barrier = Loss(0, y_val);
            rate(t_idx) = 1 / ((pi/2) * f_val * ...
                erfi(sqrt((L_barrier - L_min) / (2*Delta_s/f_val))));
        end
    end
end
