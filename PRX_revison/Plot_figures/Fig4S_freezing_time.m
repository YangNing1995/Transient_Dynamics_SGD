% Fig S4: Jaccard threshold robustness — heatmaps of eta * <t_f>
% Reads CSV results from recompute_freezing_thresholds.py and plots one
% heatmap per threshold on the same 6x6 grid as main text Fig. 2.
% -------------------------------------------------------------------------

clear; clc; close all;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
prx_root = fileparts(script_dir);
repo_root = fileparts(prx_root);

plot_path_candidates = {
    fullfile(prx_root, 'Plot_figures')
    fullfile(repo_root, 'Plot_figures')
};
for path_idx = 1:numel(plot_path_candidates)
    if exist(plot_path_candidates{path_idx}, 'dir')
        addpath(plot_path_candidates{path_idx});
    end
end

csv_root = fullfile(repo_root, 'freezing_time_training_lr_r1-5_ce', 'threshold_robustness');
save_dir = fullfile(prx_root, 'Figures_SI', 'Fig4S_freezing_time');
fig2_summary = fullfile(prx_root, 'Figures_revision', 'Fig2_freezing_control', ...
    'Fig2_freezing_control_summary.csv');
if ~exist(csv_root, 'dir')
    error('CSV input directory not found: %s', csv_root);
end
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% Full grid (for CSV data loading)
bs_list_full = [1000, 500, 200, 100, 50, 20, 10];
lr_list_full = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
num_bs_full  = length(bs_list_full);
num_lr_full  = length(lr_list_full);

%% Display grid (aligned with main text Fig. 2, no B=1000 or lr=0.001)
bs_display = [500, 200, 100, 50, 20, 10];
lr_display = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
bs_idx = 2:7;
lr_idx = 2:7;

%% Divergence mask from main text Fig. 2 (n_converged_final = NaN → diverged)
T_fig2 = readtable(fig2_summary);
diverge_mask = false(length(bs_display), length(lr_display));
for i = 1:length(bs_display)
    for j = 1:length(lr_display)
        idx = T_fig2.batch_size == bs_display(i) & ...
              abs(T_fig2.learning_rate - lr_display(j)) < 1e-12;
        if any(idx)
            diverge_mask(i, j) = isnan(T_fig2.n_converged_final(find(idx, 1)));
        else
            diverge_mask(i, j) = true;
        end
    end
end

%% Thresholds
threshold_list = [0.90, 0.95, 0.99];
eta_tf_jaccard09 = [];  % will store Jaccard S_th=0.90 for correlation plot

for th_idx = 1:length(threshold_list)
    th = threshold_list(th_idx);
    csv_file = fullfile(csv_root, sprintf('freezing_time_jaccard%.2f.csv', th));
    if ~exist(csv_file, 'file')
        error('CSV input file not found: %s', csv_file);
    end

    % --- Read CSV ---
    T = readtable(csv_file);

    % --- Compute mean eta*t_f per (BS, LR) on full grid ---
    eta_tf_full = NaN(num_bs_full, num_lr_full);

    for i = 1:num_bs_full
        for j = 1:num_lr_full
            mask = (T.batch_size == bs_list_full(i)) & ...
                   (abs(T.learning_rate - lr_list_full(j)) < 1e-8);
            if any(mask)
                vals = T.eta_tf(mask);
                eta_tf_full(i, j) = mean(vals, 'omitnan');
            end
        end
    end

    % --- Slice to display grid and apply divergence mask ---
    eta_tf_mean = eta_tf_full(bs_idx, lr_idx);
    eta_tf_mean(diverge_mask) = NaN;

    % Save Jaccard S_th=0.90 for correlation plot
    if th == 0.90
        eta_tf_jaccard09 = eta_tf_mean;
    end

    % --- Plot heatmap ---
    draw_heatmap(eta_tf_mean, sprintf('Jaccard ($S_\\mathrm{th}=%.2f$)', th), ...
        bs_display, lr_display, save_dir, ...
        sprintf('Fig4S_jaccard_threshold_%.2f', th));
end

%% ===== Pred099 (prediction agreement >= 0.99) =====
pred099_dir = fullfile(repo_root, 'freezing_time_training_lr_r1-5_ce', 'pred099');
eta_tf_pred099 = NaN(length(bs_display), length(lr_display));
for i = 1:length(bs_display)
    for j = 1:length(lr_display)
        csv_path = fullfile(pred099_dir, ...
            sprintf('bs%d_lr%g', bs_display(i), lr_display(j)), ...
            'freezing_time_summary.csv');
        if exist(csv_path, 'file')
            T_pred = readtable(csv_path);
            eta_tf_pred099(i, j) = mean(T_pred.eta_tf, 'omitnan');
        end
    end
end
eta_tf_pred099(diverge_mask) = NaN;

draw_heatmap(eta_tf_pred099, 'Pred.\ agreement ($\geq 0.99$)', ...
    bs_display, lr_display, save_dir, 'Fig4S_pred099');

%% ===== Loss proxy (train loss <= 0.1) =====
data_dir = fullfile(repo_root, '..', 'Data', 'Train_with_different_hyperparas');
loss_threshold = 0.1;
num_repeats = 20;
eta_tf_loss = NaN(length(bs_display), length(lr_display));
for i = 1:length(bs_display)
    bs = bs_display(i);
    for j = 1:length(lr_display)
        lr = lr_display(j);
        tf_vals = [];
        for k = 1:num_repeats
            mat_path = fullfile(data_dir, ...
                sprintf('bs%d_lr%g', bs, lr), ...
                sprintf('save_metrics_repeat%d.mat', k));
            if exist(mat_path, 'file')
                S = load(mat_path, 'train_loss', 'save_iterations');
                idx_freeze = find(S.train_loss <= loss_threshold, 1, 'first');
                if ~isempty(idx_freeze)
                    tf_vals(end+1) = lr * S.save_iterations(idx_freeze); %#ok<AGROW>
                end
            end
        end
        if ~isempty(tf_vals)
            eta_tf_loss(i, j) = mean(tf_vals);
        end
    end
end
eta_tf_loss(diverge_mask) = NaN;

draw_heatmap(eta_tf_loss, 'Loss proxy ($\mathcal{L} \leq 0.1$)', ...
    bs_display, lr_display, save_dir, 'Fig4S_loss_proxy');

%% ===== Correlation scatter plot =====
fig = figure('Units', 'points', 'PaperUnits', 'points', ...
    'Position', [100 100 450 400]);

c_pred = [0.85 0.37 0.01];   % orange
c_loss = [0.00 0.45 0.74];   % blue
valid_all = ~diverge_mask;

hold on;
% Pred099 vs Jaccard
vp = valid_all & ~isnan(eta_tf_pred099) & ~isnan(eta_tf_jaccard09);
x_pred = eta_tf_jaccard09(vp);  y_pred = eta_tf_pred099(vp);
R2_pred = corr(x_pred(:), y_pred(:))^2;
scatter(x_pred, y_pred, 80, c_pred, 'filled', 'MarkerFaceAlpha', 0.7);

% Loss proxy vs Jaccard
vl = valid_all & ~isnan(eta_tf_loss) & ~isnan(eta_tf_jaccard09);
x_loss = eta_tf_jaccard09(vl);  y_loss = eta_tf_loss(vl);
R2_loss = corr(x_loss(:), y_loss(:))^2;
scatter(x_loss, y_loss, 80, c_loss, 'filled', 'MarkerFaceAlpha', 0.7);

% Diagonal reference
plot([0 100], [0 100], 'k--', 'LineWidth', 1);
hold off;

xlabel('Jaccard $\eta\langle t_f \rangle$', 'Interpreter', 'latex');
ylabel('Other method $\eta\langle t_f \rangle$', 'Interpreter', 'latex');
legend({sprintf('Pred.\\ agreement ($R^2=%.2f$)', R2_pred), ...
        sprintf('Loss proxy ($R^2=%.2f$)', R2_loss)}, ...
    'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 14);
xlim([0 100]); ylim([0 100]);
axis square;
set(gca, 'Fontname', 'Times New Roman', 'FontSize', 18);

set(fig, 'Color', 'w', 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
drawnow;
print(fig, fullfile(save_dir, 'Fig4S_correlation.png'), '-dpng', '-r600');
close(fig);
fprintf('Saved: Fig4S_correlation\n');

fprintf('\nAll Fig S4 panels saved to:\n  %s\n', save_dir);

%% ========== Local function ==========
function draw_heatmap(Data, title_str, bs_display, lr_display, save_dir, filename)
    fig = figure('Units', 'points', 'PaperUnits', 'points', ...
        'Position', [100 100 400 400]);
    imagesc(Data);
    clim([0 100]);
    set(gca, 'YDir', 'normal');
    axis square;
    colormap(viridis);

    cb = colorbar;
    cb.Label.String = '$\eta\langle t_f \rangle$';
    cb.Label.Units = 'normalized';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 18;
    cb.Label.Rotation = 0;
    cb.Label.Position = [0.5, 1.1, 0];
    cb.Ticks = [0 50 100];

    xlabel('Learning rate $\eta$', 'Interpreter', 'latex');
    ylabel('Batch size $B$', 'Interpreter', 'latex');
    xticks(1:length(lr_display));
    yticks(1:length(bs_display));
    xticklabels(compose('%g', lr_display));
    yticklabels(string(bs_display));
    set(gca, 'Fontname', 'Times New Roman', 'FontSize', 18);

    title(title_str, 'Interpreter', 'latex', 'FontSize', 20);

    % Gray overlay for NaN (divergent) cells
    hold on;
    [ny, nx] = size(Data);
    for ii = 1:ny
        for jj = 1:nx
            if isnan(Data(ii, jj))
                patch([jj-0.5 jj+0.5 jj+0.5 jj-0.5], ...
                      [ii-0.5 ii-0.5 ii+0.5 ii+0.5], ...
                      [0.70 0.70 0.70], 'EdgeColor', 'none');
            end
        end
    end
    hold off;

    set(fig, 'Color', 'w', 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
    drawnow;
    print(fig, fullfile(save_dir, [filename '.png']), '-dpng', '-r600');
    close(fig);
    fprintf('Saved: %s\n', filename);
end
