% Fig4 PREVIEW: draw with available data
%   Panel B: old trajectory files + cross markers (90% estimate for now)
%   Panel C: 6 completed sim points + analytical NESS band
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

g1 = @(y) 2*x1.*y./(y + y_b);
g2 = @(y) 2*x2.*y./(y + y_b);
f1 = @(y) f0.*(x1.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
f2 = @(y) f0.*(x2.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
L0 = @(y) L_d.*exp(-y/y_d) + x0^2/f0;
L  = @(x, y) arrayfun(@(xx) ...
        L0(y) + (xx>=0)*xx.*(xx - g1(y))./f1(y) + ...
        (xx<0)*xx.*(xx + g2(y))./f2(y), x);

gamma = x1^2 / x2^2;
P_eq  = 1 / (1 + 1/gamma);

%% ===== Panel B: New simulation data with paper-correct y_freeze ========
sim_data = load(fullfile(repo_root, 'Two_valleys_model', 'Fig4_simulation_data.mat'));

% display_order: indices into panel_b_sigmas = [0.01, 0.05, 0.1]
%   → display as sigma=0.1 (blue), 0.05 (orange), 0.01 (green)
display_order = [3, 2, 1];

y_data     = cell(3, 1);
P_sim_data = cell(3, 1);
yf_est     = zeros(3, 1);
Pf_est     = zeros(3, 1);

for ci = 1:3
    di = display_order(ci);
    y_data{ci}     = sim_data.pb_mean_y{di};
    P_sim_data{ci} = sim_data.pb_P_right{di};
    yf_est(ci)     = sim_data.pb_y_freeze(di);
    % P_flat at y_freeze
    [~, idx_yf] = min(abs(y_data{ci} - yf_est(ci)));
    Ps_smooth = movmean(P_sim_data{ci}, max(50, floor(length(P_sim_data{ci})/20)));
    Pf_est(ci) = Ps_smooth(idx_yf);
end

clr = [0.00 0.45 0.74;  0.85 0.33 0.10;  0.47 0.67 0.19];
ds_lab = {'$\Delta_S=10^{-3}$','$\Delta_S=5{\times}10^{-4}$','$\Delta_S=10^{-4}$'};

figB = figure('Units','points','PaperUnits','points','Position',[100 100 500 400]);
hold on;

for i = 1:3
    plot(y_data{i}, P_sim_data{i}, '-', ...
        'Color', [clr(i,:) 0.6], 'LineWidth', 2.0, ...
        'DisplayName', ds_lab{i});
end

plot([1 20], [P_eq P_eq], '-.', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, ...
    'HandleVisibility', 'off');
text(1.15, P_eq - 0.03, sprintf('$P_\\mathrm{flat}^\\mathrm{eq}=%.2f$', P_eq), ...
    'Interpreter', 'latex', 'FontSize', 14, 'Color', [0.4 0.4 0.4]);

for i = 1:3
    plot([yf_est(i) yf_est(i)], [0.5 Pf_est(i)], '--', ...
        'Color', [clr(i,:) 0.5], 'LineWidth', 1.2, 'HandleVisibility', 'off');
    hv = 'off'; if i == 1, hv = 'on'; end
    plot(yf_est(i), Pf_est(i), 'x', ...
        'MarkerSize', 14, 'LineWidth', 2.5, 'Color', 'k', ...
        'HandleVisibility', hv, 'DisplayName', '$y_\mathrm{freeze}$');
end

xlabel('$y$', 'Interpreter', 'LaTeX', 'FontSize', 22);
ylabel('$P_\mathrm{flat}$', 'Interpreter', 'LaTeX', 'FontSize', 22);
ylim([0.5 1.02]); xlim([1 20]); yticks([0.5 0.75 1]);
legend('Location', 'southeast', 'Interpreter', 'LaTeX', 'Box', 'on', 'FontSize', 13);
grid on; box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 20, 'XScale', 'log');
save_panel(figB, fullfile(out_dir, 'Fig4B_Pflat_vs_y.png'), export_res);


%% ===== Panel C: Contour of P_flat^ss(Delta_S, y) + y_freeze trajectory ==
%  Color map shows NESS P_flat on the (Delta_S, y) plane.
%  y_freeze(Delta_S) trajectory overlaid — climbs to higher y with noise,
%  tracking high-P_flat contours even as each fixed-y slice drops.

% --- 1. Compute NESS P_flat on a 2D grid --------------------------------
DS_grid = logspace(-4.5, -1.5, 200);
y_grid  = linspace(0.3, 10, 200);
P_ness_grid = zeros(numel(y_grid), numel(DS_grid));

for iy = 1:numel(y_grid)
    for jd = 1:numel(DS_grid)
        rl = NESS_escape_rate('left',  y_grid(iy), f1, f2, g1, g2, L, DS_grid(jd));
        rr = NESS_escape_rate('right', y_grid(iy), f1, f2, g1, g2, L, DS_grid(jd));
        P_ness_grid(iy, jd) = rl / (rr + rl);
    end
end

% --- 2. y_freeze trajectory ---------------------------------------------
eps_freeze = 0.01;
C_freeze   = y_b * sqrt(2*log(1/eps_freeze)/(x1*x2));
y_freeze_func = @(DS) C_freeze .* sqrt(DS);
yf_line = y_freeze_func(DS_grid);

% Clip trajectory to y_grid range
yf_line(yf_line > max(y_grid)) = NaN;

% P_flat along trajectory (Eq. 12)
epsilon_tr = 100;
Phi = (2*(x1^2 - x2^2)) / (27*y_b) * (L_d/y_d + 8*x0^2/(27*y_f));
Base_line  = sqrt(DS_grid) ./ (epsilon_tr * Phi);
Ptr_line   = (1 + gamma^(-0.5) .* Base_line.^(1-gamma)).^(-1);

% --- 3. Draw -------------------------------------------------------------
figC = figure('Units','points','PaperUnits','points','Position',[100 100 520 400]);

% Filled contour — P_flat^ss landscape
contour_levels = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98];
[~, hcf] = contourf(DS_grid, y_grid, P_ness_grid, contour_levels, ...
    'LineColor', [0.6 0.6 0.6]);
set(hcf, 'HandleVisibility', 'off');
hold on;
set(gca, 'XScale', 'log');

colormap(flipud(parula(256)));
cb = colorbar;
ylabel(cb, '$P_\mathrm{flat}^\mathrm{ss}$', 'Interpreter', 'latex', 'FontSize', 18);
clim([0.55 1.0]);
cb.Ticks = [0.6 0.7 0.8 0.9 1.0];

% Contour line labels
[C_lines, h_lines] = contour(DS_grid, y_grid, P_ness_grid, ...
    [0.7, 0.8, 0.9, 0.95], 'LineColor', [0.9 0.9 0.9], 'LineWidth', 0.8);
clabel(C_lines, h_lines, 'FontSize', 10, 'Color', 'w', ...
    'FontName', 'Times New Roman');
set(h_lines, 'HandleVisibility', 'off');

% y_freeze trajectory — thick black line with white outline
plot(DS_grid, yf_line, '-', 'Color', 'w', 'LineWidth', 4.5, ...
    'HandleVisibility', 'off');
plot(DS_grid, yf_line, '-', 'Color', 'k', 'LineWidth', 2.5, ...
    'DisplayName', '$y_\mathrm{freeze}(\Delta_S)$');

% Fixed-y reference line for contrast (e.g., y=2)
y_ref = 2;
plot(DS_grid([1 end]), [y_ref y_ref], '--', 'Color', 'w', 'LineWidth', 2.5, ...
    'HandleVisibility', 'off');
plot(DS_grid([1 end]), [y_ref y_ref], '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.8, ...
    'DisplayName', sprintf('Fixed $y{=}%d$', y_ref));

xlabel('$\Delta_S = \eta\sigma$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$y$',                     'Interpreter', 'latex', 'FontSize', 20);
xlim([3e-5 6e-3]);
ylim([0.3 8]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 18);
lg = legend('Location', 'northwest', 'Interpreter', 'LaTeX', 'FontSize', 13, ...
       'Box', 'on', 'TextColor', 'k');
box on;
save_panel(figC, fullfile(out_dir, 'Fig4C_paradox.png'), export_res);

fprintf('\nPreview exported to:\n  %s\n', out_dir);


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
