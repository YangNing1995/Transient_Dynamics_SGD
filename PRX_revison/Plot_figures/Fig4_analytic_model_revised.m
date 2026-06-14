% Fig4 revised: Analytical model results
%   Panel A: Effective loss landscape (unchanged)
%   Panel B: Simulation P_flat(y) with y_freeze markers (paper definition)
%   Panel C: Paradox plot — NESS band vs simulation outcome
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
%  Uses new data with actual y_freeze (y at last valley hop).

sim_data = load(fullfile(repo_root, 'Two_valleys_model', 'Fig4_simulation_data.mat'));

% --- Colours & labels (ordered: largest to smallest noise) ---------------
%  panel_b_sigmas = [0.01, 0.05, 0.1] in the data file
%  Display order: sigma=0.1 (blue), 0.05 (orange), 0.01 (green)
display_order = [3, 2, 1];  % indices into panel_b_sigmas

clr = [0.00 0.45 0.74;    % blue   — largest noise  (sigma=0.1)
       0.85 0.33 0.10;    % orange — medium noise   (sigma=0.05)
       0.47 0.67 0.19];   % green  — smallest noise (sigma=0.01)

ds_lab = {'$\Delta_S=10^{-3}$', ...
          '$\Delta_S=5{\times}10^{-4}$', ...
          '$\Delta_S=10^{-4}$'};

figB = figure('Units','points','PaperUnits','points','Position',[100 100 500 400]);
hold on;

% 1) Simulation curves (solid)
for ci = 1:3
    di = display_order(ci);  % index into panel_b data
    my = sim_data.pb_mean_y{di};
    pr = sim_data.pb_P_right{di};
    plot(my, pr, '-', ...
        'Color', [clr(ci,:) 0.6], 'LineWidth', 2.0, ...
        'DisplayName', ds_lab{ci});
end

% 2) Equilibrium reference line
plot([1 20], [P_eq P_eq], '-.', ...
    'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, ...
    'HandleVisibility', 'off');
text(1.15, P_eq - 0.03, sprintf('$P_\\mathrm{flat}^\\mathrm{eq}=%.2f$', P_eq), ...
    'Interpreter', 'latex', 'FontSize', 14, 'Color', [0.4 0.4 0.4]);

% 3) y_freeze markers (cross) + vertical dashed guides
for ci = 1:3
    di = display_order(ci);
    yf = sim_data.pb_y_freeze(di);
    % Find P_flat at y_freeze from the trajectory
    my = sim_data.pb_mean_y{di};
    pr = sim_data.pb_P_right{di};
    [~, idx_yf] = min(abs(my - yf));
    pf = pr(idx_yf);

    % Vertical guide
    plot([yf yf], [0.5 pf], '--', ...
        'Color', [clr(ci,:) 0.5], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
    % Cross marker
    hv = 'off';
    if ci == 1, hv = 'on'; end
    plot(yf, pf, 'x', ...
        'MarkerSize', 14, 'LineWidth', 2.5, ...
        'Color', 'k', ...
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


%% ===== Panel C: Contour of P_flat^ss(Delta_S, y) + y_freeze trajectory ==
%  Color map shows NESS P_flat on the (Delta_S, y) plane.
%  y_freeze(Delta_S) trajectory overlaid — climbs to higher y with noise,
%  tracking high-P_flat contours even as each fixed-y slice drops.

% --- 1. Compute NESS P_flat on a 2D grid --------------------------------
DS_grid = logspace(-4, log10(6e-3), 240);
y_grid  = linspace(0.01, 15, 240);
P_ness_grid = zeros(numel(y_grid), numel(DS_grid));

for iy = 1:numel(y_grid)
    for jd = 1:numel(DS_grid)
        rl = NESS_escape_rate('left',  y_grid(iy), f1, f2, g1, g2, L, DS_grid(jd));
        rr = NESS_escape_rate('right', y_grid(iy), f1, f2, g1, g2, L, DS_grid(jd));
        P_ness_grid(iy, jd) = rl / (rr + rl);
    end
end

% The exact erfi expression overflows in the low-noise, high-barrier corner.
% Use the asymptotic NESS form there, which is the expression used in the
% transient freezing estimate.
[DS_mesh_for_calc, y_mesh_for_calc] = meshgrid(DS_grid, y_grid);
DeltaL_mesh = (x0^2/f0) .* (y_mesh_for_calc.^2 ./ (y_mesh_for_calc + y_f).^2);
f1_mesh = f1(y_mesh_for_calc);
f2_mesh = f2(y_mesh_for_calc);
P_ness_asym = (1 + gamma^(-1/2) .* ...
    exp(DeltaL_mesh .* (f2_mesh - f1_mesh) ./ (2 .* DS_mesh_for_calc))).^(-1);
bad_ness = ~isfinite(P_ness_grid) | P_ness_grid <= 0 | P_ness_grid >= 1;
P_ness_grid(bad_ness) = P_ness_asym(bad_ness);

% --- 2. y_freeze trajectory ---------------------------------------------
eps_freeze = 0.01;
Phi_scale = 3000;
Phi_freeze = Phi_scale * 2*(x1^2 - x2^2)/(27*y_b) * (L_d/y_d + 8*x0^2/(27*y_f));
y_freeze_func = @(DS) y_b .* sqrt(DS .* log(DS ./ (eps_freeze^2 * Phi_freeze^2))) ./ ...
    (x2 - sqrt(DS .* log(DS ./ (eps_freeze^2 * Phi_freeze^2))));
yf_line = y_freeze_func(DS_grid);
valid_yf = isreal(yf_line) & yf_line > min(y_grid) & yf_line < max(y_grid);
yf_line(~valid_yf) = NaN;

% --- 3. Draw -------------------------------------------------------------
figC = figure('Units','points','PaperUnits','points','Position',[100 100 520 400]);

logDS_grid = log10(DS_grid);
imagesc(logDS_grid, y_grid, P_ness_grid);
set(gca, 'YDir', 'normal');
hold on;

colormap(parula(256));
cb = colorbar;
ylabel(cb, '$P_\mathrm{flat}^\mathrm{ss}$', 'Interpreter', 'latex', 'FontSize', 18);
clim([0.65 0.98]);
cb.Ticks = [0.65 0.75 0.85 0.95];

valid_patch = isfinite(yf_line);
line_start = linspace(min(logDS_grid), max(logDS_grid) - 0.18, 18);
for ih = 1:numel(line_start)
    x0_h = line_start(ih);
    x1_h = min(max(logDS_grid), x0_h + 0.28);
    x_seg = linspace(x0_h, x1_h, 40);
    y_boundary = interp1(logDS_grid(valid_patch), yf_line(valid_patch), x_seg, 'linear', NaN);
    y_seg = y_boundary + linspace(0.25, 3.0, numel(x_seg));
    inside = isfinite(y_boundary) & y_seg <= max(y_grid);
    if nnz(inside) > 1
        plot(x_seg(inside), y_seg(inside), '-', ...
            'Color', [0 0 0], 'LineWidth', 0.8, ...
            'HandleVisibility', 'off');
    end
end

plot(logDS_grid, yf_line, '-', 'Color', 'w', 'LineWidth', 4.5, ...
    'HandleVisibility', 'off');
plot(logDS_grid, yf_line, '-', 'Color', 'k', 'LineWidth', 2.5, ...
    'DisplayName', '$y_\mathrm{freeze}(\Delta_S)$');

y_ref = 5;
plot(logDS_grid([1 end]), [y_ref y_ref], '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.8, ...
    'DisplayName', sprintf('Fixed $y{=}%d$', y_ref));

xlabel('$\Delta_S = \eta\sigma$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$y$',                     'Interpreter', 'latex', 'FontSize', 20);
xlim([min(logDS_grid) max(logDS_grid)]);
ylim([0 15]);
xticks([-4 -3.5 -3 -2.5]);
xticklabels({'$10^{-4}$', '$10^{-3.5}$', '$10^{-3}$', '$10^{-2.5}$'});
yticks([0 4 8 12]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 18);
set(gca, 'TickLabelInterpreter', 'latex');
axis square;
legend('Location', 'northwest', 'Interpreter', 'LaTeX', 'FontSize', 13, ...
       'Box', 'on', 'TextColor', 'k');
box on;
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

function rate = NESS_escape_rate(x_start, y_val, f1, f2, g1, g2, Loss, Delta_s)
    if x_start == "right"
        f_val     = f1(y_val);
        L_min     = Loss(g1(y_val)/2, y_val);
        L_barrier = Loss(0, y_val);
    else
        f_val     = f2(y_val);
        L_min     = Loss(-g2(y_val)/2, y_val);
        L_barrier = Loss(0, y_val);
    end
    rate = 1 / ((pi/2) * f_val * erfi(sqrt((L_barrier - L_min) / (2*Delta_s/f_val))));
end
