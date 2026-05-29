% Fig. 2 heatmaps exported as four separate panels.
% The script intentionally keeps the plotting layer simple: it reads the
% precomputed Fig2 summary CSV and draws one heatmap per figure.

clear; clc;

%% Paths and settings
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);  % for viridis.m

repo_root = fullfile(script_dir, '..', '..');
out_dir = fullfile(repo_root, 'PRX_revison', 'Figures_revision', 'Fig2_freezing_control');
summary_csv = fullfile(out_dir, 'Fig2_freezing_control_summary.csv');
data_root = fullfile(repo_root, '..', 'Data', 'Train_with_different_hyperparas');
train_acc_threshold = 0.999;
n_repeats = 20;

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
if ~exist(summary_csv, 'file')
    error('Missing summary file: %s\nRun the data-summary generation once before plotting.', summary_csv);
end

% x-axis: learning rate increases to the right.
% y-axis: batch size decreases upward, so the upper-right corner is high noise.
lr_display = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
bs_display = [500, 200, 100, 50, 20, 10];
export_resolution = 600;

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

T = readtable(summary_csv);

%% Convert table columns to heatmap matrices
Freezing_time = make_matrix(T, 'mean_eta_tf', bs_display, lr_display);
Solution_spread = log10(make_matrix(T, 'rms_radius', bs_display, lr_display));
Final_flatness = make_mean_hessian_flatness(data_root, out_dir, bs_display, lr_display, ...
    train_acc_threshold, n_repeats);
Best_accuracy = make_matrix(T, 'max_test_acc', bs_display, lr_display);

% Mark the two divergent high-noise conditions in panel A as unavailable,
% consistent with the gray cells in the final-solution panels.
N_converged_final = make_matrix(T, 'n_converged_final', bs_display, lr_display);
Freezing_time(isnan(N_converged_final)) = NaN;

%% Draw and export panels separately
plot_one_heatmap(Freezing_time, bs_display, lr_display, ...
    '$\eta \langle t_f \rangle$', ...
    fullfile(out_dir, 'Fig2A_freezing_coordinate.png'), ...
    export_resolution, [0, 100], [0, 50, 100], false);

plot_one_heatmap(Solution_spread, bs_display, lr_display, ...
    '$r_\theta$', ...
    fullfile(out_dir, 'Fig2B_solution_spread.png'), ...
    export_resolution, log10([0.03, 3]), log10([0.03, 0.1, 0.3, 1, 3]), false, ...
    {'0.03', '0.1', '0.3', '1', '3'});

plot_one_heatmap(Final_flatness, bs_display, lr_display, ...
    '$\langle F\rangle$', ...
    fullfile(out_dir, 'Fig2C_final_flatness.png'), ...
    export_resolution, [1.5, 5.5], [2, 3, 4, 5], false);

plot_one_heatmap(Best_accuracy, bs_display, lr_display, ...
    '$ \max(Acc_\mathrm{test})$', ...
    fullfile(out_dir, 'Fig2D_best_accuracy.png'), ...
    export_resolution, [0.90, 0.93], [0.90, 0.91, 0.92, 0.93], false);

fprintf('Exported Fig2 panels to %s\n', out_dir);


function M = make_matrix(T, value_name, bs_display, lr_display)
    M = nan(numel(bs_display), numel(lr_display));
    for i = 1:numel(bs_display)
        for j = 1:numel(lr_display)
            idx = T.batch_size == bs_display(i) & abs(T.learning_rate - lr_display(j)) < 1e-12;
            if any(idx)
                M(i, j) = T.(value_name)(find(idx, 1));
            end
        end
    end
end


function M = make_mean_hessian_flatness(data_root, out_dir, bs_display, lr_display, train_acc_threshold, n_repeats)
    cache_path = fullfile(out_dir, 'hessian_flatness_summary.csv');
    M = nan(numel(bs_display), numel(lr_display));
    if exist(cache_path, 'file')
        T = readtable(cache_path);
        for i = 1:numel(bs_display)
            for j = 1:numel(lr_display)
                idx = T.batch_size == bs_display(i) & abs(T.learning_rate - lr_display(j)) < 1e-12;
                if any(idx)
                    M(i, j) = T.mean_hessian_flatness(find(idx, 1));
                end
            end
        end
        return;
    end

    rows = [];
    for i = 1:numel(bs_display)
        for j = 1:numel(lr_display)
            bs = bs_display(i);
            lr = lr_display(j);
            vals = [];
            for repeat = 1:n_repeats
                metrics_path = fullfile(data_root, sprintf('bs%d_lr%g', bs, lr), ...
                    sprintf('save_metrics_repeat%d.mat', repeat));
                hessian_path = fullfile(data_root, sprintf('bs%d_lr%g', bs, lr), ...
                    sprintf('save_hessian_repeat%d.mat', repeat));
                if ~exist(metrics_path, 'file') || ~exist(hessian_path, 'file')
                    continue;
                end
                S_metrics = load(metrics_path, 'train_accuracy');
                if S_metrics.train_accuracy(end) < train_acc_threshold
                    continue;
                end
                S_hessian = load(hessian_path, 'Hessian');
                eigvals = real(S_hessian.Hessian(1:10, end));
                if all(eigvals > 0)
                    vals(end + 1) = prod(eigvals)^(-1/10); %#ok<AGROW>
                end
            end
            M(i, j) = mean(vals, 'omitnan');
            rows = [rows; bs, lr, numel(vals), M(i, j)]; %#ok<AGROW>
        end
    end

    T = array2table(rows, 'VariableNames', ...
        {'batch_size', 'learning_rate', 'n_converged', 'mean_hessian_flatness'});
    writetable(T, cache_path);
end


function plot_one_heatmap(Data, bs_display, lr_display, cbar_text, output_file, resolution, clim_values, cbar_ticks, show_values, cbar_tick_labels)
    if nargin < 10
        cbar_tick_labels = [];
    end

    fig = figure('unit', 'points', 'PaperUnits', 'points', 'position', [100 100 400 400]);
    imagesc(Data);
    set(gca, 'YDir', 'normal');
    axis square;
    colormap(viridis);

    if ~isempty(clim_values)
        clim(clim_values);
        cmin = clim_values(1);
        cmax = clim_values(2);
    else
        valid_values = Data(isfinite(Data));
        cmin = min(valid_values);
        cmax = max(valid_values);
    end

    cb = colorbar;
    cb.Label.String = cbar_text;
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 18;
    cb.Label.Rotation = 0;
    cb.Label.Units = 'normalized';
    cb.Label.Position = [0.5, 1.1, 0];
    set_colorbar_ticks(cb, cmin, cmax, cbar_ticks, cbar_tick_labels);

    xlabel('Learning rate $\eta$', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('Batch size $B$', 'Interpreter', 'latex', 'FontSize', 18);
    xticks(1:numel(lr_display));
    yticks(1:numel(bs_display));
    xticklabels(compose('%g', lr_display));
    yticklabels(string(bs_display));
    set(gca, 'Fontname', 'Times New Roman', 'FontSize', 18, 'LineWidth', 1.0);

    hold on;
    [num_y, num_x] = size(Data);
    for i = 1:num_y
        for j = 1:num_x
            if isnan(Data(i, j))
                patch([j - 0.5, j + 0.5, j + 0.5, j - 0.5], ...
                      [i - 0.5, i - 0.5, i + 0.5, i + 0.5], ...
                      [0.70, 0.70, 0.70], 'EdgeColor', 'none');
                if show_values
                    text(j, i, '--', 'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'middle', 'FontSize', 12);
                end
            elseif show_values
                text(j, i, sprintf('%.2g', Data(i, j)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'FontSize', 12, 'Color', 'k');
            end
        end
    end
    hold off;

    save_fig2_png(fig, output_file, resolution);
    close(fig);
end


function set_colorbar_ticks(cb, cmin, cmax, tick_values, tick_labels)
    if isempty(tick_values)
        tick_values = linspace(cmin, cmax, 4);
    end
    cb.Ticks = tick_values;
    if nargin >= 5 && ~isempty(tick_labels)
        cb.TickLabels = tick_labels;
    else
        cb.TickLabels = strip_trailing_zeros(tick_values);
    end
end


function labels = strip_trailing_zeros(values)
    labels = strings(size(values));
    for i = 1:numel(values)
        label = sprintf('%.3g', values(i));
        if contains(label, '.')
            label = regexprep(label, '0+$', '');
            label = regexprep(label, '\.$', '');
        end
        labels(i) = label;
    end
end


function save_fig2_png(fig_handle, output_file, resolution)
    % Same export style as Fig1_escaping_behavior.m: keep the full canvas so
    % manual assembly in Illustrator/PowerPoint has predictable dimensions.
    set(fig_handle, 'Color', 'w', 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
    set(findall(fig_handle, 'Type', 'axes'), 'Color', 'w');
    drawnow;
    print(fig_handle, output_file, '-dpng', sprintf('-r%d', resolution));
end
