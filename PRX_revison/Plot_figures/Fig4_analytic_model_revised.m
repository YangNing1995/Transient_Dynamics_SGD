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

dg1dy = @(y) (2*x1*y_b) ./ ((y + y_b).^2);
dg2dy = @(y) (2*x2*y_b) ./ ((y + y_b).^2);
d2g1dy2 = @(y) -(4*x1*y_b) ./ ((y + y_b).^3);
d2g2dy2 = @(y) -(4*x2*y_b) ./ ((y + y_b).^3);
df1dy = @(y) (2*f0*(x1^2/x0^2) .* (y + y_f) * (y_b - y_f)) ./ ((y + y_b).^3);
df2dy = @(y) (2*f0*(x2^2/x0^2) .* (y + y_f) * (y_b - y_f)) ./ ((y + y_b).^3);
dL0dy = @(y) -(L_d/y_d) .* exp(-y/y_d);
d2L0dy2 = @(y) (L_d/y_d^2) .* exp(-y/y_d);
dLdx = @(x, y) (x >= 0) .* ((2*x - g1(y)) ./ f1(y)) + ...
               (x < 0)  .* ((2*x + g2(y)) ./ f2(y));
dLdy = @(x, y) dL0dy(y) - ...
               (x >= 0) .* (x.*(x-g1(y)).*df1dy(y)./(f1(y).^2) + x.*dg1dy(y)./f1(y)) - ...
               (x < 0)  .* (x.*(x+g2(y)).*df2dy(y)./(f2(y).^2) - x.*dg2dy(y)./f2(y));
H11 = @(x, y) (x >= 0) .* (2 ./ f1(y)) + ...
              (x < 0)  .* (2 ./ f2(y));
H12 = @(x, y) (x >= 0) .* (-dg1dy(y)./f1(y) - (2*x - g1(y)) .* df1dy(y) ./ (f1(y).^2)) + ...
              (x < 0)  .* (dg2dy(y)./f2(y) - (2*x + g2(y)) .* df2dy(y) ./ (f2(y).^2));
H22 = @(x, y) (x >= 0) .* (-x .* d2g1dy2(y)./ f1(y) + ...
                           2*x .* dg1dy(y) .* df1dy(y) ./ (f1(y).^2) + ...
                           2*x .* (x - g1(y)) .* (df1dy(y).^2) ./ (f1(y).^3) + ...
                           d2L0dy2(y)) + ...
              (x < 0)  .* (x .* d2g2dy2(y)./ f2(y) - ...
                           2*x .* dg2dy(y) .* df2dy(y) ./ (f2(y).^2) + ...
                           2*x .* (x + g2(y)) .* (df2dy(y).^2) ./ (f2(y).^3) + ...
                           d2L0dy2(y));
Hessian = @(x, y) [H11(x, y), H12(x, y); H12(x, y), H22(x, y)];

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


%% ===== Panel B: P_flat(y) and representative hopping trajectories =======
%  Top: ensemble P_flat(y) from the original P_right simulation files.
%  Bottom: a few freshly simulated trajectories on the two-valley landscape.

file_list = {
    fullfile(repo_root, 'Two_valleys_model', 'P_right_eta0.010_sigma0.100.mat'), ...
    fullfile(repo_root, 'Two_valleys_model', 'P_right_eta0.010_sigma0.050.mat'), ...
    fullfile(repo_root, 'Two_valleys_model', 'P_right_eta0.010_sigma0.010.mat')
};

clr = [0.00 0.45 0.74;    % blue   — largest noise
       0.85 0.33 0.10;    % orange — medium noise
       0.47 0.67 0.19];   % green  — smallest noise

ds_lab = {'$\Delta_S=10^{-3}$', ...
          '$\Delta_S=5{\times}10^{-4}$', ...
          '$\Delta_S=10^{-4}$'};

figB = figure('Units','points','PaperUnits','points','Position',[100 100 520 400]);

axB1 = axes('Parent', figB, 'Position', [0.15 0.57 0.78 0.35]);
hold(axB1, 'on');
y_contour = linspace(1, 10, 180);
x_contour = linspace(-0.38, 0.55, 180);
[Yc, Xc] = meshgrid(y_contour, x_contour);
Lc = arrayfun(@(xx, yy) L(xx, yy), Xc, Yc);
contour(axB1, Yc, Xc, Lc, 14, 'LineColor', [0.8 0.8 0.8], 'LineWidth', 0.55, ...
    'HandleVisibility', 'off');
plot(axB1, [1 10], [0 0], 'k--', 'LineWidth', 0.9, 'HandleVisibility', 'off');
plot(axB1, y_contour, g1(y_contour)/2, '-', 'Color', [0.35 0.35 0.35], ...
    'LineWidth', 0.9, 'HandleVisibility', 'off');
plot(axB1, y_contour, -g2(y_contour)/2, '-', 'Color', [0.35 0.35 0.35], ...
    'LineWidth', 0.9, 'HandleVisibility', 'off');

rng(11);
traj_sigmas = [0.1, 0.05, 0.01];
pool_size = 18;
n_select_each = 2;
traj_iterations = 52000;
traj_eta = 0.01;
for ci = 1:numel(traj_sigmas)
    traj_pool = cell(pool_size, 1);
    last_cross_y = nan(pool_size, 1);
    for ri = 1:pool_size
        init_x = -0.01;
        if mod(ri, 2) == 0
            init_x = 0.01;
        end
        traj = sgd_2d(L, dLdx, dLdy, Hessian, [init_x, 1], ...
            traj_eta, traj_iterations, traj_sigmas(ci));
        traj_pool{ri} = traj;
        last_cross_y(ri) = last_crossing_y(traj, 10);
    end
    valid = find(isfinite(last_cross_y));
    if numel(valid) >= n_select_each
        [~, order] = sort(last_cross_y(valid), 'ascend');
        pick_local = unique(round(linspace(max(1, round(0.35*numel(order))), numel(order), n_select_each)));
        selected = valid(order(pick_local));
    else
        selected = 1:min(n_select_each, pool_size);
    end
    for jj = 1:numel(selected)
        traj = traj_pool{selected(jj)};
        keep = traj(:,2) >= 1 & traj(:,2) <= 10 & traj(:,1) >= -0.42 & traj(:,1) <= 0.58;
        idx = find(keep);
        idx = idx(1:18:end);
        if numel(idx) < 5
            continue;
        end
        x_smooth = smoothdata(traj(idx,1), 'movmean', 31);
        plot(axB1, traj(idx,2), x_smooth, '-', ...
            'Color', [clr(ci,:) 0.78], 'LineWidth', 1.35, ...
            'HandleVisibility', 'off');
    end
end

ylabel(axB1, '$x$', 'Interpreter', 'LaTeX', 'FontSize', 20);
xlim(axB1, [1 10]);
ylim(axB1, [-0.38 0.55]);
yticks(axB1, [-0.25 0 0.25 0.5]);
box(axB1, 'on');
set(axB1, 'FontName', 'Times New Roman', 'FontSize', 18, 'XTickLabel', []);

axB2 = axes('Parent', figB, 'Position', [0.15 0.13 0.78 0.35]);
hold(axB2, 'on');
for ci = 1:numel(file_list)
    data = load(file_list{ci});
    plot(axB2, data.mean_y, data.P_right_simulation, '-', ...
        'Color', [clr(ci,:) 0.72], 'LineWidth', 2.2, ...
        'DisplayName', ds_lab{ci});
end
plot(axB2, [1 10], [P_eq P_eq], '-.', ...
    'Color', [0.45 0.45 0.45], 'LineWidth', 1.2, ...
    'HandleVisibility', 'off');
text(axB2, 1.18, P_eq - 0.055, sprintf('$P_\\mathrm{flat}^\\mathrm{eq}=%.2f$', P_eq), ...
    'Interpreter', 'latex', 'FontSize', 14, 'Color', [0.35 0.35 0.35], ...
    'BackgroundColor', 'w', 'Margin', 1.5);
xlabel(axB2, '$y$', 'Interpreter', 'LaTeX', 'FontSize', 20);
ylabel(axB2, '$P_\mathrm{flat}$', 'Interpreter', 'LaTeX', 'FontSize', 20);
xlim(axB2, [1 10]);
ylim(axB2, [0.48 1.02]);
yticks(axB2, [0.5 0.75 1]);
legend(axB2, 'Location', 'southeast', 'Interpreter', 'LaTeX', ...
       'Box', 'on', 'FontSize', 12);
grid(axB2, 'on');
box(axB2, 'on');
set(axB2, 'FontName', 'Times New Roman', 'FontSize', 18);

save_panel(figB, fullfile(out_dir, 'Fig4B_Pflat_and_trajectories.png'), export_res);


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
cb.Label.String = '$P_{\rm flat}^{\rm ss}$';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0;
cb.Label.Units = 'normalized';
cb.Label.Position = [0.5, 1.1, 0];
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

% plot(logDS_grid, yf_line, '-', 'Color', 'w', 'LineWidth', 4.5, ...
%     'HandleVisibility', 'off');
plot(logDS_grid, yf_line, '-', 'Color', [0.85 0.2 0.2], 'LineWidth', 2.5, ...
    'DisplayName', '$y_\mathrm{freeze}(\Delta_S)$');

y_ref = 5;
plot(logDS_grid([1 end]), [y_ref y_ref], '--', 'Color', 'k', 'LineWidth', 1.8, ...
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

function yf = last_crossing_y(trajectory, y_max)
    x = trajectory(:, 1);
    y = trajectory(:, 2);
    valid = y <= y_max;
    crossing_idx = find(valid(1:end-1) & valid(2:end) & x(1:end-1).*x(2:end) < 0);
    if isempty(crossing_idx)
        yf = NaN;
    else
        yf = y(crossing_idx(end));
    end
end

function trajectory = sgd_2d(Loss, grad_x, grad_y, Hessian, initial_pos, learning_rate, iterations, noise_strength)
    trajectory = zeros(iterations+1, 2);
    trajectory(1, :) = initial_pos;
    x = initial_pos(1);
    y = initial_pos(2);

    for i = 1:iterations
        dx = grad_x(x, y);
        dy = grad_y(x, y);

        H = Hessian(x, y);
        H = (H + H') / 2;
        [V, D] = eig(H);
        D_positive = max(D, 0);
        Cov = 2 * noise_strength * V * D_positive * V';
        Cov = (Cov + Cov') / 2 + 1e-8 * eye(2);

        [Vn, Dn] = eig(Cov);
        noise = (Vn * sqrt(max(Dn, 0)) * randn(2, 1))';

        x = x - learning_rate * (dx + noise(1));
        y = y - learning_rate * (dy + noise(2));
        trajectory(i+1, :) = [x, y];
    end
end
