% Figure 1 for the PRX revision: continuation readout of transient freezing.
% The script builds one four-panel figure from the original continuation data:
% A: PCA visualization of the SGD trajectory and deterministic continuations.
% B: endpoint test loss and neuron-wise flatness versus continuation time.
% C: pairwise endpoint Jaccard similarity showing the late stable block.
% D: final-basin stability score and operational definition of t_f.

clear; close all; clc;

%% Paths and options
script_dir = fileparts(mfilename('fullpath'));
revision_dir = fileparts(script_dir);
output_dir = fullfile(revision_dir, 'Figures_revision', 'Fig1_transient_freezing');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
addpath(script_dir);

train_data_root = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Train_with_different_hyperparas\';
continue_data_root = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Continue_training\';

bs = 50;
lr = 0.05;
realization_num = 2;
num_solutions = 51;
num_iterations = 101;
continue_iterations_list = linspace(0, 1000, num_solutions);
base_iterations_list = linspace(0, 100/lr, num_iterations);
continuation_stride = continue_iterations_list(2) - continue_iterations_list(1);

% Similarity threshold used only to visualize the basin-stability readout.
basin_similarity_threshold = 0.95;

%% Load original SGD trajectory
base_dir = fullfile(train_data_root, ['bs', num2str(bs), '_lr', num2str(lr)]);
base_metrics = load(fullfile(base_dir, ['save_metrics_repeat', num2str(realization_num), '.mat']));
positions = find(ismember(base_metrics.save_iterations, base_iterations_list));

Train_loss_base = base_metrics.train_loss(1, positions);
Test_loss_base = base_metrics.test_loss(1, positions);
Weights_base = base_metrics.weight_all(positions, :);

%% Load continuation trajectories
continue_dir = fullfile(continue_data_root, ['bs', num2str(bs), '_lr', num2str(lr), '_repeat', num2str(realization_num), '_ct']);
flatness_file = fullfile(continue_dir, 'save_neuron_wise_flatness_continue_all.mat');
if ~exist(flatness_file, 'file')
    error('Missing neuron-wise flatness file: %s', flatness_file);
end

Train_loss_all = zeros(num_solutions, num_iterations);
Test_loss_all = zeros(num_solutions, num_iterations);
Test_acc_all = zeros(num_solutions, num_iterations);
Wrong_indices_all = cell(num_solutions, num_iterations);
Weights_all = zeros(num_solutions, num_iterations, size(Weights_base, 2));

for i = 1:num_solutions
    tc = continue_iterations_list(i);
    metrics = load(fullfile(continue_dir, ['save_metrics_ct', num2str(tc), '.mat']));
    Train_loss_all(i, :) = metrics.train_loss;
    Test_loss_all(i, :) = metrics.test_loss;
    Test_acc_all(i, :) = metrics.test_accuracy;
    Weights_all(i, :, :) = metrics.weight_all;

    if iscell(metrics.wrong_indices)
        Wrong_indices_all(i, :) = metrics.wrong_indices;
    else
        Wrong_indices_all(i, :) = num2cell(metrics.wrong_indices, 2)';
    end
end

base_flatness = load(fullfile(continue_dir, 'save_neuron_wise_flatness_base.mat'), 'Flatness');
Flatness_base = squeeze(base_flatness.Flatness);
continuation_flatness = load(flatness_file, 'Flatness_all');
Flatness_all = continuation_flatness.Flatness_all;

%% Extract endpoints at the common global time t = 2000
Flatness_end = zeros(num_solutions, 1);
Test_loss_end = zeros(num_solutions, 1);
Test_acc_end = zeros(num_solutions, 1);
Wrong_indices_end = cell(num_solutions, 1);

for i = 1:num_solutions
    endpoint_idx = num_iterations - i + 1;
    Flatness_end(i) = Flatness_all(i, endpoint_idx);
    Test_loss_end(i) = Test_loss_all(i, endpoint_idx);
    Test_acc_end(i) = Test_acc_all(i, endpoint_idx);
    Wrong_indices_end{i} = Wrong_indices_all{i, endpoint_idx};
end

%% Pairwise endpoint similarity and continuation-based freezing time
similarity_matrix = zeros(num_solutions, num_solutions);
for i = 1:num_solutions
    for j = 1:num_solutions
        similarity_matrix(i, j) = jaccard_similarity(Wrong_indices_end{i}, Wrong_indices_end{j});
    end
end

final_basin_similarity = similarity_matrix(:, end);
stable_tail = false(num_solutions, 1);
for i = 1:num_solutions
    stable_tail(i) = all(final_basin_similarity(i:end) >= basin_similarity_threshold);
end
freezing_index = find(stable_tail, 1, 'first');
if isempty(freezing_index)
    warning('No stable tail found at threshold %.2f; using the historical t_f = 440.', basin_similarity_threshold);
    freezing_index = round(440 / continuation_stride) + 1;
end
t_freeze = continue_iterations_list(freezing_index);

%% PCA projection for trajectory visualization
selected_tc = [0, 20, 40, 80, 160, 320, t_freeze, 640, 1000];
selected_tc = unique(selected_tc(selected_tc >= 0 & selected_tc <= 1000), 'stable');
solution_list = round(selected_tc / continuation_stride) + 1;
solution_list = solution_list(solution_list >= 1 & solution_list <= num_solutions);

Weights_all_reshaped = reshape(Weights_all, [num_solutions * num_iterations, size(Weights_base, 2)]);
[coeff, score, ~, ~, ~, mu] = pca(Weights_all_reshaped);
Weights_pca = reshape(score, [num_solutions, num_iterations, size(Weights_base, 2)]);
Weights_base_pca = (Weights_base - mu) * coeff;

%% Figure style
blue = [68, 119, 179] / 255;
orange = [221, 132, 82] / 255;
red = [204, 51, 68] / 255;
gray = [0.45, 0.45, 0.45];
branch_colors = turbo(max(numel(solution_list), 3));

fig = figure('Units', 'points', 'PaperUnits', 'points', ...
    'Position', [80, 60, 980, 760], 'Color', 'w');
tiled = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
axis_font = 11;
title_font = 12;
label_font = 12;

%% A. Continuation protocol in PCA-loss space
axA = nexttile(tiled, 1);
h_sgd = plot3(axA, Weights_base_pca(:, 1), Weights_base_pca(:, 2), Train_loss_base, '-', ...
    'Color', blue, 'LineWidth', 2.2, 'DisplayName', 'SGD');
hold(axA, 'on');
h_cont = gobjects(1, 1);
for k = 1:numel(solution_list)
    sol = solution_list(k);
    h_branch = plot3(axA, squeeze(Weights_pca(sol, :, 1)), squeeze(Weights_pca(sol, :, 2)), Train_loss_all(sol, :), ...
        '-o', 'Color', branch_colors(k, :), 'LineWidth', 0.8, 'MarkerSize', 2.8, ...
        'MarkerFaceColor', branch_colors(k, :), 'MarkerEdgeColor', branch_colors(k, :));
    if k == 1
        h_cont = h_branch;
    end
end
h_init = scatter3(axA, Weights_base_pca(1, 1), Weights_base_pca(1, 2), Train_loss_base(1), ...
    60, 'k', 'p', 'filled', 'DisplayName', 'init');
h_final = scatter3(axA, Weights_base_pca(end, 1), Weights_base_pca(end, 2), Train_loss_base(end), ...
    50, blue, 'o', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'final');
view(axA, -148, 21);
grid(axA, 'on'); box(axA, 'on');
set(axA, 'ZScale', 'log', 'FontName', 'Times New Roman', 'FontSize', axis_font, 'LineWidth', 0.8);
xlabel(axA, '$\theta_{\rm pc1}$', 'Interpreter', 'latex', 'FontSize', label_font);
ylabel(axA, '$\theta_{\rm pc2}$', 'Interpreter', 'latex', 'FontSize', label_font);
zlabel(axA, '$\mathcal{L}_{\rm train}$', 'Interpreter', 'latex', 'FontSize', label_font);
title(axA, '\bf A  Continuation readout', 'Interpreter', 'latex', 'FontSize', title_font);
legend(axA, [h_sgd, h_cont, h_init, h_final], ...
    {'SGD reference', 'GD continuations', 'initial', 'final'}, ...
    'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9, 'Box', 'off');

%% B. Endpoint quality versus continuation time
axB = nexttile(tiled, 2);
yyaxis(axB, 'left');
plot(axB, continue_iterations_list, Test_loss_end, '-o', 'Color', blue, ...
    'LineWidth', 1.8, 'MarkerSize', 4.2, 'MarkerFaceColor', blue);
ylabel(axB, '$\mathcal{L}_{\rm test}\!\left[\theta_{\rm GD}^{(t_c)}(2000-t_c)\right]$', ...
    'Interpreter', 'latex', 'Color', blue, 'FontSize', label_font);
axB.YAxis(1).Color = blue;

yyaxis(axB, 'right');
plot(axB, continue_iterations_list, Flatness_end, '-s', 'Color', orange, ...
    'LineWidth', 1.8, 'MarkerSize', 4.2, 'MarkerFaceColor', orange);
ylabel(axB, '$F_{\rm nw}\!\left[\theta_{\rm GD}^{(t_c)}(2000-t_c)\right]$', ...
    'Interpreter', 'latex', 'Color', orange, 'FontSize', label_font);
axB.YAxis(2).Color = orange;

xline(axB, t_freeze, '--', '$t_f$', 'Color', red, 'LineWidth', 1.6, ...
    'Interpreter', 'latex', 'LabelVerticalAlignment', 'bottom');
grid(axB, 'on'); box(axB, 'on');
set(axB, 'FontName', 'Times New Roman', 'FontSize', axis_font, 'LineWidth', 0.8);
xlabel(axB, 'continuation time $t_c$', 'Interpreter', 'latex', 'FontSize', label_font);
title(axB, '\bf B  Continuation endpoint quality', 'Interpreter', 'latex', 'FontSize', title_font);
xlim(axB, [0, 1000]);

%% C. Endpoint similarity matrix
axC = nexttile(tiled, 3);
imagesc(axC, similarity_matrix);
axis(axC, 'square');
if exist('viridis', 'file')
    colormap(axC, viridis);
else
    colormap(axC, parula);
end
cb = colorbar(axC);
cb.Label.String = 'Jaccard similarity';
cb.Label.Interpreter = 'latex';
cb.FontName = 'Times New Roman';
cb.FontSize = axis_font;
hold(axC, 'on');
rectangle(axC, 'Position', [freezing_index - 0.5, freezing_index - 0.5, ...
    num_solutions - freezing_index + 1, num_solutions - freezing_index + 1], ...
    'EdgeColor', red, 'LineWidth', 2.0);
set(axC, 'YDir', 'normal', 'FontName', 'Times New Roman', 'FontSize', axis_font, 'LineWidth', 0.8);
tick_positions = 1:10:num_solutions;
tick_labels = continue_iterations_list(tick_positions);
set(axC, 'XTick', tick_positions, 'XTickLabel', tick_labels, ...
    'YTick', tick_positions, 'YTickLabel', tick_labels);
xlabel(axC, '$t_c$', 'Interpreter', 'latex', 'FontSize', label_font);
ylabel(axC, '$t_c$', 'Interpreter', 'latex', 'FontSize', label_font);
title(axC, '\bf C  Endpoint similarity matrix', 'Interpreter', 'latex', 'FontSize', title_font);

%% D. Operational definition of t_f
axD = nexttile(tiled, 4);
plot(axD, continue_iterations_list, final_basin_similarity, '-o', 'Color', gray, ...
    'LineWidth', 1.8, 'MarkerSize', 4.2, 'MarkerFaceColor', [0.85, 0.85, 0.85]);
hold(axD, 'on');
yline(axD, basin_similarity_threshold, ':', 'basin-stability threshold', ...
    'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.4, 'Interpreter', 'latex');
xline(axD, t_freeze, '--', ['$t_f=', num2str(t_freeze), '$'], 'Color', red, ...
    'LineWidth', 1.6, 'Interpreter', 'latex', 'LabelVerticalAlignment', 'bottom');

stable_y = 0.055 * ones(size(continue_iterations_list));
scatter(axD, continue_iterations_list(~stable_tail), stable_y(~stable_tail), 28, [0.78, 0.78, 0.78], 'filled');
scatter(axD, continue_iterations_list(stable_tail), stable_y(stable_tail), 32, red, 'filled');
text(axD, 40, 0.17, '$t_f=\min\{t_i: b(t_j)=b_\ast\;\forall j\geq i\}$', ...
    'Interpreter', 'latex', 'FontSize', 11, 'BackgroundColor', 'w', 'Margin', 3);
text(axD, 40, 0.075, 'stable tail', 'Interpreter', 'latex', 'FontSize', 10, 'Color', red);
grid(axD, 'on'); box(axD, 'on');
set(axD, 'FontName', 'Times New Roman', 'FontSize', axis_font, 'LineWidth', 0.8);
xlabel(axD, 'continuation time $t_c$', 'Interpreter', 'latex', 'FontSize', label_font);
ylabel(axD, '$Sim_J\left[b(t_c), b_\ast\right]$', 'Interpreter', 'latex', 'FontSize', label_font);
title(axD, '\bf D  Operational definition of $t_f$', 'Interpreter', 'latex', 'FontSize', title_font);
xlim(axD, [0, 1000]);
ylim(axD, [0, 1.03]);

%% Save outputs
exportgraphics(fig, fullfile(output_dir, 'Fig1_transient_freezing.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(output_dir, 'Fig1_transient_freezing.pdf'), 'ContentType', 'vector');
savefig(fig, fullfile(output_dir, 'Fig1_transient_freezing.fig'));

fprintf('Saved Figure 1 to:\n  %s\n', output_dir);
fprintf('Continuation-based t_f = %.0f iterations at similarity threshold %.2f.\n', t_freeze, basin_similarity_threshold);

%% Local helper
function similarity = jaccard_similarity(list1, list2)
    set1 = unique(list1);
    set2 = unique(list2);
    union_set = union(set1, set2);
    if isempty(union_set)
        similarity = 1;
    else
        similarity = numel(intersect(set1, set2)) / numel(union_set);
    end
end
