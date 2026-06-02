% FigS3: Jaccard threshold robustness — heatmaps of eta * <t_f>
% Reads CSV results from recompute_freezing_thresholds.py and plots one
% heatmap per threshold, matching the style of the original Fig S3.
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
save_dir = fullfile(script_dir, 'FigS3_threshold_robustness');
if ~exist(csv_root, 'dir')
    error('CSV input directory not found: %s', csv_root);
end
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% Grid (same as reference Fig 3 / Fig S3)
bs_list  = [1000, 500, 200, 100, 50, 20, 10];
lr_list  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
num_bs   = length(bs_list);
num_lr   = length(lr_list);

%% Thresholds
threshold_list = [0.90, 0.95, 0.99];

for th_idx = 1:length(threshold_list)
    th = threshold_list(th_idx);
    csv_file = fullfile(csv_root, sprintf('freezing_time_jaccard%.2f.csv', th));
    if ~exist(csv_file, 'file')
        error('CSV input file not found: %s', csv_file);
    end

    % --- Read CSV ---
    T = readtable(csv_file);

    % --- Compute mean eta*t_f per (BS, LR) condition ---
    eta_tf_mean = NaN(num_bs, num_lr);

    for i = 1:num_bs
        for j = 1:num_lr
            mask = (T.batch_size == bs_list(i)) & ...
                   (abs(T.learning_rate - lr_list(j)) < 1e-8);
            if any(mask)
                vals = T.eta_tf(mask);
                eta_tf_mean(i, j) = mean(vals, 'omitnan');
            end
        end
    end

    % --- Color limits from valid (non-NaN) entries ---
    valid = eta_tf_mean(~isnan(eta_tf_mean));
    clim_lo = min(valid);
    clim_hi = max(valid);

    % --- Plot ---
    fig = figure('Units', 'points', 'PaperUnits', 'points', ...
                 'Position', [100 + 50*th_idx, 100, 400, 400]);

    imagesc(eta_tf_mean, [clim_lo clim_hi]);
    set(gca, 'YDir', 'normal');
    axis square;
    if exist('viridis', 'file')
        colormap(viridis());
    else
        warning('viridis.m not found on MATLAB path; using parula instead.');
        colormap(parula);
    end

    % Colorbar
    cb = colorbar;
    cb.Label.String = '$\eta\langle t_f \rangle$';
    cb.Label.Units  = 'normalized';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 18;
    cb.Label.Rotation = 0;
    cb.Label.Position = [0.5, 1.1, 0];

    % Axis labels
    xlabel('Learning rate $\eta$', 'Interpreter', 'latex');
    set(gca, 'XTick', 1:num_lr, 'XTickLabel', string(lr_list));
    ylabel('Batch size $B$', 'Interpreter', 'latex');
    set(gca, 'YTick', 1:num_bs, 'YTickLabel', string(bs_list));
    set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 18);

    % Title
    title(sprintf('$S_\\mathrm{th}=%.2f$', th), ...
          'Interpreter', 'latex', 'FontSize', 20);

    % Gray overlay for missing data (NaN cells)
    hold on;
    for i = 1:num_bs
        for j = 1:num_lr
            if isnan(eta_tf_mean(i, j))
                patch([j-0.5 j+0.5 j+0.5 j-0.5], ...
                      [i-0.5 i-0.5 i+0.5 i+0.5], ...
                      [0.5 0.5 0.5], 'EdgeColor', 'none');
            end
        end
    end
    hold off;

    % --- Save ---
    fig_name = sprintf('FigS3_jaccard_threshold_%.2f', th);
    saveas(fig, fullfile(save_dir, [fig_name '.fig']));
    saveas(fig, fullfile(save_dir, [fig_name '.png']));
    print(fig, fullfile(save_dir, [fig_name '.pdf']), '-dpdf', '-bestfit');

    fprintf('Saved: %s (clim=[%.3f, %.3f])\n', fig_name, clim_lo, clim_hi);
end

fprintf('\nAll figures saved to: %s\n', save_dir);
