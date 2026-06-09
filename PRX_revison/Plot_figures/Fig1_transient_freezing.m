% Results of continuing training with GD at different moments along a single SGD trajectory
% This version uses the original Hessian flatness definition from
% Fig4_4S_escaping_behavior.m.
% =========================================================================

%% Load original SGD trajectory
Data_dir = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Train_with_different_hyperparas\';
bs = 50;
lr = 0.05;
realization_num = 2;

% Load metrics and Hessian spectra for the specific run.
load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_metrics_repeat', num2str(realization_num), '.mat'));
load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_hessian_repeat', num2str(realization_num), '.mat'));

% Process iterations
max_iteration = 100/lr;
iterations_list = linspace(0, max_iteration, 101);
Positions = find(ismember(save_iterations, iterations_list));

% Extract base metrics
Train_loss_base = train_loss(1, Positions);
Test_loss_base = test_loss(1, Positions);
Train_acc_base = train_accuracy(1, Positions);
Test_acc_base = test_accuracy(1, Positions);
Wrong_indices_base = wrong_indices(1, Positions);
Weights_base = weight_all(Positions, :);
Hessian_base = Hessian';

%% Load various GD trajectories 
% (Start points at 0, 20, 40...1000 Iterations, 51 points total)
% (Each trajectory samples 101 time points from 0:20:2000)
Data_dir = strcat('E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Continue_training\', ...
    'bs', num2str(bs), '_lr', num2str(lr), '_repeat', num2str(realization_num), '_ct/');

num_solutions = 51;
num_iterations = 101;
continue_iterations_list = linspace(0, 1000, num_solutions);

% Pre-allocate arrays
Train_loss_all = zeros(num_solutions, num_iterations);
Test_loss_all = zeros(num_solutions, num_iterations);
Train_acc_all = zeros(num_solutions, num_iterations);
Test_acc_all = zeros(num_solutions, num_iterations);
Wrong_indices_all = cell(num_solutions, num_iterations);
Weights_all = zeros(num_solutions, num_iterations, 2500);
Hessian_all = zeros(num_solutions, num_iterations, 2500);

% Load data loop
for i = 1:num_solutions
    load(strcat(Data_dir, 'save_metrics_ct', num2str(20*(i-1)), '.mat'));
    load(strcat(Data_dir, 'save_hessian_ct', num2str(20*(i-1)), '.mat'));
    
    Train_loss_all(i, :) = train_loss;
    Test_loss_all(i, :) = test_loss;
    Train_acc_all(i, :) = train_accuracy;
    Test_acc_all(i, :) = test_accuracy;
    Weights_all(i, :, :) = weight_all;
    Hessian_all(i, :, :) = Hessian';
    
    if iscell(wrong_indices)
        Wrong_indices_all(i, :) = wrong_indices;
    else
        Wrong_indices_all(i, :) = num2cell(wrong_indices, 2)';
    end
end

% Calculate flatness: inverse geometric mean of the top 10 Hessian eigenvalues.
Flatness_base = prod(Hessian_base(:, 1:10), 2).^(-1/10);
Flatness_all = prod(Hessian_all(:, :, 1:10), 3).^(-1/10);

% Extract end-point metrics for each solution
Flatness_end = zeros(num_solutions, 1);
Test_acc_end = zeros(num_solutions, 1);
Test_loss_end = zeros(num_solutions, 1);

for i = 1:num_solutions
    Flatness_end(i, 1) = Flatness_all(i, end-i+1);
    Test_acc_end(i, 1) = Test_acc_all(i, end-i+1);
    Test_loss_end(i, 1) = Test_loss_all(i, end-i+1);
end

Flatness_values = [Flatness_base(:); Flatness_all(:)];
Flatness_values = Flatness_values(isfinite(Flatness_values));
Flatness_ylim = [floor(min(Flatness_values)*10)/10, ceil(max(Flatness_values)*10)/10];
Flatness_end_ylim = [floor(min(Flatness_end)*10)/10, ceil(max(Flatness_end)*10)/10];

fig1_export_dir = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Transient_Dynamics_SGD\PRX_revison\Figures_revision\Fig1_transient_freezing';
fig1_export_resolution = 600;
fig1_axis_font = 24;
fig1_label_font = fig1_axis_font;
if ~exist(fig1_export_dir, 'dir')
    mkdir(fig1_export_dir);
end

%% A. Loss landscape & weight dynamics
solution_list = [1, 2, 3, 5, 9, 17, 33];
Weights_all_reshaped = reshape(Weights_all, [num_solutions*num_iterations, 2500]);

% PCA Projection ('score' is the projection in the PCA coordinate system)
[coeff, score, latent, ~, ~, mu] = pca(Weights_all_reshaped); 
Weights_pca = reshape(score, [num_solutions, num_iterations, 2500]);
Weights_base_pca = (Weights_base - mu) * coeff;

% Color definition
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;

% 2D + Loss Visualization
fig_A = figure('unit','points','PaperUnits','points', 'position', [100 100 600 600]);
landscape_solution_handles = gobjects(length(solution_list), 1);
landscape_solution_labels = cell(length(solution_list), 1);
plot3(Weights_base_pca(:, 1), Weights_base_pca(:, 2), Train_loss_base, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD reference');
hold on 
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    landscape_solution_handles(i) = plot3(Weights_pca(sol, :, 1), Weights_pca(sol, :, 2), Train_loss_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label);
    landscape_solution_labels{i} = legend_label;
end
% Mark start points of each continuation on SGD curve
for i = 1:length(solution_list)
    sol = solution_list(i);
    scatter3(Weights_base_pca(sol, 1), Weights_base_pca(sol, 2), Train_loss_base(sol), ...
        150, landscape_solution_handles(i).Color, 'x', 'LineWidth', 2.5, 'HandleVisibility', 'off');
end

% Settings
grid on
box on
view(-148, 21)
hx = xlabel('$\theta_\mathrm{pc1}$', 'Interpreter', 'latex','Units','normalized', 'FontSize', fig1_label_font);
hy = ylabel('$\theta_\mathrm{pc2}$', 'Interpreter', 'latex','Units','normalized', 'FontSize', fig1_label_font);
zlabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex', 'FontSize', fig1_label_font)
zlim([min(Train_loss_all, [], 'all'), max(Train_loss_all, [], 'all')])
set(gca, 'Fontname', 'Times New Roman', 'Zscale', 'log', 'Fontsize', fig1_axis_font);

% Adjust label positions
set(hx, 'Position', [ 0.1956    0.0083         0],'Units','normalized');   % Move closer in y direction
set(hy, 'Position', [  0.8044    -0.01         0],'Units','normalized');   % Move closer in x direction
save_fig1_png(fig_A, fullfile(fig1_export_dir, 'Fig1A_loss_landscape.png'), fig1_export_resolution);

% Standalone two-row legend for panel A
fig_A_legend = figure('unit','points','PaperUnits','points', 'position', [100 100 600 120]);
legend_axes = axes('Position', [0 0 1 1], 'Visible', 'off');
hold(legend_axes, 'on')
legend_handles = gobjects(10, 1);
legend_labels = cell(10, 1);

legend_handles(1) = plot(legend_axes, NaN, NaN, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'MarkerSize', 10);
legend_labels{1} = 'SGD reference';
for i = 1:4
    legend_handles(i + 1) = plot(legend_axes, NaN, NaN, ...
        'LineStyle', landscape_solution_handles(i).LineStyle, ...
        'Marker', landscape_solution_handles(i).Marker, ...
        'color', landscape_solution_handles(i).Color, ...
        'LineWidth', landscape_solution_handles(i).LineWidth, ...
        'MarkerSize', landscape_solution_handles(i).MarkerSize);
    legend_labels{i + 1} = landscape_solution_labels{i};
end
legend_handles(6) = plot(legend_axes, NaN, NaN, 'x', 'Color', 'k', 'MarkerSize', 15, 'LineWidth', 3, 'LineStyle', 'none');
legend_labels{6} = 'Continuation point';
for i = 5:length(solution_list)
    legend_handles(i + 2) = plot(legend_axes, NaN, NaN, ...
        'LineStyle', landscape_solution_handles(i).LineStyle, ...
        'Marker', landscape_solution_handles(i).Marker, ...
        'color', landscape_solution_handles(i).Color, ...
        'LineWidth', landscape_solution_handles(i).LineWidth, ...
        'MarkerSize', landscape_solution_handles(i).MarkerSize);
    legend_labels{i + 2} = landscape_solution_labels{i};
end
legend_handles(10) = plot(legend_axes, NaN, NaN, 'LineStyle', 'none', 'Marker', 'none');
legend_labels{10} = '';
lgd = legend(legend_axes, legend_handles, legend_labels, ...
    'Fontname', 'Times New Roman', 'Fontsize', 16, 'location', 'north', ...
    'Orientation', 'horizontal', 'NumColumns', 5, ...
    'Interpreter', 'latex');
set(lgd, 'Box', 'on');
save_fig1_png(fig_A_legend, fullfile(fig1_export_dir, 'Fig1A_legend.png'), fig1_export_resolution);

%% Train loss
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
semilogy(iterations_list, Train_loss_base,'-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD reference');
hold on
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    semilogy(iterations_list + continue_iterations_list(sol), Train_loss_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end
legend('Fontname', 'Times New Roman', 'Fontsize', 18, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
xlim([0 2000])
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex')
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);

%% B. Test loss
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
fig_B = figure('unit','points','PaperUnits','points', 'position', [100 100 600 450]);
semilogy(iterations_list, Test_loss_base,'-', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD reference');
hold on
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    semilogy(iterations_list + continue_iterations_list(sol), Test_loss_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end
% Mark start points of each continuation on SGD curve
for i = 1:length(solution_list)
    sol = solution_list(i);
    scatter(continue_iterations_list(sol), Test_loss_base(sol), 150, ...
        landscape_solution_handles(i).Color, 'x', 'LineWidth', 2.5, 'HandleVisibility', 'off');
end
% legend('Fontname', 'Times New Roman', 'Fontsize', 18, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
xlim([0 1500])
xlabel('Iteration $t$', 'Interpreter', 'latex', 'FontSize', fig1_label_font)
ylabel('$\mathcal{L}_\mathrm{test}$', 'Interpreter', 'latex', 'FontSize', fig1_label_font)
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', fig1_axis_font);
save_fig1_png(fig_B, fullfile(fig1_export_dir, 'Fig1B_test_loss.png'), fig1_export_resolution);

%% Train accuracy
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(iterations_list, Train_acc_base,'-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD reference');
hold on
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    plot(iterations_list + continue_iterations_list(sol), Train_acc_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end
legend('Fontname', 'Times New Roman', 'Fontsize', 18, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
xlim([0 2000])
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$Acc_\mathrm{train}$', 'Interpreter', 'latex')
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);

%% Test accuracy
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(iterations_list, Test_acc_base,'-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD reference');
hold on
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    plot(iterations_list + continue_iterations_list(sol), Test_acc_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end
legend('Fontname', 'Times New Roman', 'Fontsize', 18, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
xlim([0 2000])
ylim([0.8 0.95])
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$Acc_\mathrm{test}$', 'Interpreter', 'latex')
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);

%% Test loss new (Combined figure)
% Combined figure: Main plot + inset subplot
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; ...
              204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])

% -------- Main plot: Test loss trajectory --------
semilogy(iterations_list, Test_loss_base, '-', 'color', color_seq(2,:), ...
    'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
hold on
for i = 1:length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    semilogy(iterations_list + continue_iterations_list(sol), ...
        Test_loss_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, ...
        'DisplayName', legend_label)
end

xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$\mathcal{L}_\mathrm{test}$', 'Interpreter', 'latex')
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlim([0 2000])
yticks([0.5 1 2])

% -------- Inset plot: Final test loss vs t_c --------
% Position: [x, y, width, height]
axes('Position', [0.45, 0.45, 0.45, 0.45]); 
plot(continue_iterations_list(1:2:end), Test_loss_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('$\mathcal{L}_\mathrm{test}$', 'Interpreter', 'latex')
xlim([0 1000])
yticks([0.44 0.46 0.48])
box on  % Add border to inset

%% C. Flatness
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
fig_C = figure('unit','points','PaperUnits','points', 'position', [100 100 600 450]);
plot(iterations_list, Flatness_base,'-', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
hold on
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    plot(iterations_list + continue_iterations_list(sol), Flatness_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end
% Mark start points of each continuation on SGD curve
for i = 1:length(solution_list)
    sol = solution_list(i);
    scatter(continue_iterations_list(sol), Flatness_base(sol), 150, ...
        landscape_solution_handles(i).Color, 'x', 'LineWidth', 2.5, 'HandleVisibility', 'off');
end
% legend('Fontname', 'Times New Roman', 'Fontsize', 18, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
xlim([0 1500])
xlabel('Iteration $t$', 'Interpreter', 'latex', 'FontSize', fig1_label_font)
ylabel('Flatness $F$', 'Interpreter', 'latex', 'FontSize', fig1_label_font)
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', fig1_axis_font);
save_fig1_png(fig_C, fullfile(fig1_export_dir, 'Fig1C_flatness.png'), fig1_export_resolution);

%% Flatness new (Combined figure)
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; ...
              204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])

% -------- Main plot: flatness trajectory --------
plot(iterations_list,  Flatness_base, '-', 'color', color_seq(2,:), ...
    'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
hold on
for i = 1:length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    plot(iterations_list + continue_iterations_list(sol), ...
        Flatness_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, ...
        'DisplayName', legend_label)
end

xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('Flatness $F$', 'Interpreter', 'latex')
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlim([0 2000])
ylim(Flatness_ylim)

% -------- Inset plot: final flatness vs t_c --------
axes('Position', [0.18, 0.45, 0.45, 0.45]); % [x, y, width, height]
plot(continue_iterations_list(1:2:end), Flatness_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('Flatness $F$', 'Interpreter', 'latex')
xlim([0 1000])
ylim(Flatness_end_ylim)
box on  % Add border to inset

%% Flatness vs Test loss/accuracy (Linear Regression)
% Calculate metrics at iteration 2000 for all solutions
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
Flatness_end = zeros(num_solutions, 1);
Train_acc_end = zeros(num_solutions, 1);
Train_loss_end = zeros(num_solutions, 1);
Test_acc_end = zeros(num_solutions, 1);
Test_loss_end = zeros(num_solutions, 1);

for i = 1:num_solutions
    Flatness_end(i, 1) = Flatness_all(i, end-i+1);
    Train_acc_end(i, 1) = Train_acc_all(i, end-i+1);
    Train_loss_end(i, 1) = Train_loss_all(i, end-i+1);
    Test_acc_end(i, 1) = Test_acc_all(i, end-i+1);
    Test_loss_end(i, 1) = Test_loss_all(i, end-i+1);
end

% Plot: flatness vs test loss
figure('unit','points','PaperUnits','points', 'position', [100 100 600 600])
scatter(Test_loss_end, Flatness_end, 150, 'o', 'MarkerEdgeColor','k', 'MarkerFaceColor', color_seq(3, :), 'LineWidth', 1.0);
hold on;
model = fitlm(Test_loss_end, Flatness_end, 'linear');
h = plot(Test_loss_end, predict(model), 'color', color_seq(6, :), 'LineWidth', 2);
r2 = model.Rsquared.Ordinary;
legend_text = sprintf('Fitted line : $R^2 = %.2f$', r2);
legend(h, legend_text, 'Location', 'northeast', 'FontSize', 30, 'Interpreter', 'latex');
legend box off
xlabel('$\mathcal{L}_\mathrm{test}$', 'Interpreter', 'latex')
ylabel('Flatness $F$', 'Interpreter', 'latex')
ylim(Flatness_end_ylim)
box on
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 30);
axis square

% Plot: flatness vs test accuracy
figure('unit','points','PaperUnits','points', 'position', [100 100 600 600])
scatter(Test_acc_end, Flatness_end, 150, 'o', 'MarkerEdgeColor','k', 'MarkerFaceColor', color_seq(3, :), 'LineWidth', 1.0);
hold on;
model = fitlm(Test_acc_end, Flatness_end, 'linear');
h = plot(Test_acc_end, predict(model), 'color', color_seq(6, :), 'LineWidth', 2);
r2 = model.Rsquared.Ordinary;
legend_text = sprintf('Fitted line : $R^2 = %.2f$', r2);
legend(h, legend_text, 'Location', 'northwest', 'FontSize', 30, 'Interpreter', 'latex');
legend box off
xlabel('$Acc_\mathrm{test}$', 'Interpreter', 'latex')
ylabel('Flatness $F$', 'Interpreter', 'latex')
xlim([0.888 0.912])
ylim(Flatness_end_ylim)
box on
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 30);
axis square

%% Final train/test loss/accuracy & flatness vs t_c
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;

% Train loss vs t_c
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(continue_iterations_list(1:2:end), Train_loss_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex')
xlim([0 1000])
yticks([3.6 3.8 4.0 4.2]*1e-3) 
box on 

% Test loss vs t_c
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(continue_iterations_list(1:2:end), Test_loss_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('$\mathcal{L}_\mathrm{test}$', 'Interpreter', 'latex')
xlim([0 1000])
yticks([0.44 0.46 0.48])

% Train accuracy vs t_c
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(continue_iterations_list(1:2:end), Train_acc_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('$Acc_\mathrm{train}$', 'Interpreter', 'latex')
xlim([0 1000])
ylim([0 1.0]) 
box on 

% Test accuracy vs t_c
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(continue_iterations_list(1:2:end), Test_acc_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('$Acc_\mathrm{test}$', 'Interpreter', 'latex')
xlim([0 1000])
% yticks([0.88 0.90 0.92]) 
box on 

% flatness vs t_c
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(continue_iterations_list(1:2:end), Flatness_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('Flatness $F$', 'Interpreter', 'latex')
xlim([0 1000])
ylim(Flatness_end_ylim)
box on 


%% D. Calculate pairwise Jaccard similarity between 51 solutions after training completion
% (Ratio of intersection to union of the two sets)
axis_font = 27;
label_font = 27;

num_solutions = 51;
Wrong_indices_end = cell(num_solutions, 1);
similarity_matrix = zeros(num_solutions, num_solutions);

% Extract final wrong indices
for i = 1:num_solutions
    Wrong_indices_end{i, 1} = Wrong_indices_all{i, end-i+1};
end

% Calculate similarity matrix
for i = 1:num_solutions
    for j = 1:num_solutions
        if i ~= j  % Only calculate for different pairs
            similarity_matrix(i, j) = jaccard_similarity(Wrong_indices_end{i, 1}, Wrong_indices_end{j, 1});
        else
            similarity_matrix(i, j) = 1;  % Similarity with itself is 1
        end
    end
end

max_solution_num = 51;
fig_D = figure('unit','points','PaperUnits','points', 'position', [100 100 600 600]);
imagesc(similarity_matrix(1:max_solution_num, 1:max_solution_num))
colormap(viridis)
cb_D = colorbar;
axis square

tick_labels = 0: 200: 20*(max_solution_num-1);
set(gca, 'XTick', 1:10:max_solution_num, 'XTickLabel', tick_labels, 'YTick', 1:10:max_solution_num, 'YTickLabel', tick_labels,'YDir','normal');
xlabel('continuation time $t_c$', 'Interpreter', 'latex', 'FontSize', label_font)
ylabel('continuation time $t_c$', 'Interpreter', 'latex', 'FontSize', label_font)
set(gca,'Fontname', 'Times New Roman','Fontsize', axis_font);
set(cb_D, 'FontName', 'Times New Roman', 'FontSize', axis_font);

% ----------- Highlight the region where t_c >= t_freeze -----------
t_freeze = 420;
index = t_freeze/20+1;
hold on
% Rectangle position [x, y, w, h]. Bottom-left corner (index-0.5) to cover appropriate cells.
rectangle('Position', [index-0.5, index-0.5, max_solution_num-index+1, max_solution_num-index+1], ...
          'EdgeColor',  'r', 'LineWidth', 2);
hold off
save_fig1_png(fig_D, fullfile(fig1_export_dir, 'Fig1D_jaccard_similarity.png'), fig1_export_resolution);

%% E. Endpoint quality versus continuation time
if ~exist('t_freeze', 'var')
    t_freeze = 420;
end

repo_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
endpoint_stats_root = fullfile(repo_dir, 'fig1_endpoint_stats_bs50_lr0.05_ce_uniform_grid');
endpoint_stats = load_endpoint_stats(endpoint_stats_root);
endpoint_stats = endpoint_stats(strcmp(endpoint_stats.status, 'ok'), :);

freezing_summary_path = fullfile(repo_dir, 'freezing_time_training_lr_r1-5_ce', ...
    'jaccard09', 'bs50_lr0.05', 'freezing_time_summary.csv');
freezing_stats = readtable(freezing_summary_path);
freezing_stats = freezing_stats(strcmp(freezing_stats.status, 'ok'), :);
endpoint_tf_values = freezing_stats.tf;
endpoint_tf_mean = mean(endpoint_tf_values, 'omitnan');
endpoint_tf_sem = std(endpoint_tf_values, 'omitnan') / sqrt(sum(isfinite(endpoint_tf_values)));

endpoint_tc = unique(endpoint_stats.checkpoint_t);
endpoint_tc = sort(endpoint_tc(:));
num_endpoint_tc = numel(endpoint_tc);
endpoint_test_loss_mean = nan(num_endpoint_tc, 1);
endpoint_test_loss_sem = nan(num_endpoint_tc, 1);
endpoint_flatness_mean = nan(num_endpoint_tc, 1);
endpoint_flatness_sem = nan(num_endpoint_tc, 1);
endpoint_repeat_count = zeros(num_endpoint_tc, 1);

for i = 1:num_endpoint_tc
    tc_mask = endpoint_stats.checkpoint_t == endpoint_tc(i);
    test_loss_values = endpoint_stats.test_loss(tc_mask);
    flatness_values = endpoint_stats.hessian_flatness(tc_mask);
    endpoint_repeat_count(i) = sum(tc_mask);
    endpoint_test_loss_mean(i) = mean(test_loss_values, 'omitnan');
    endpoint_flatness_mean(i) = mean(flatness_values, 'omitnan');
    endpoint_test_loss_sem(i) = std(test_loss_values, 'omitnan') / sqrt(endpoint_repeat_count(i));
    endpoint_flatness_sem(i) = std(flatness_values, 'omitnan') / sqrt(endpoint_repeat_count(i));
end

endpoint_blue = color_seq(2, :);
endpoint_orange = color_seq(4, :);
axis_font = 30;
label_font = 30;

fig_E = figure('unit','points','PaperUnits','points', 'position', [100 100 600 600]);
endpoint_ax = gca;
yyaxis(endpoint_ax, 'left');
hold(endpoint_ax, 'on')
endpoint_loss_ylim = [0.42, 0.49];
tf_band = [endpoint_tf_mean - endpoint_tf_sem, endpoint_tf_mean + endpoint_tf_sem];
patch(endpoint_ax, [tf_band(1), tf_band(2), tf_band(2), tf_band(1)], ...
    [endpoint_loss_ylim(1), endpoint_loss_ylim(1), endpoint_loss_ylim(2), endpoint_loss_ylim(2)], ...
    [1.0, 0.0, 0.0], 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
errorbar(endpoint_ax, endpoint_tc, endpoint_test_loss_mean, endpoint_test_loss_sem, '-o', ...
    'Color', endpoint_blue, 'LineWidth', 2.2, 'MarkerSize', 5, ...
    'MarkerFaceColor', endpoint_blue, 'CapSize', 5);
ylabel(endpoint_ax, '$\mathcal{L}_{\rm test}$', ...
    'Interpreter', 'latex', 'Color', endpoint_blue, 'FontSize', label_font);
ylim(endpoint_ax, endpoint_loss_ylim);
yticks(endpoint_ax, 0.42:0.01:0.49)
xline(endpoint_ax, endpoint_tf_mean, '--', 'Color', 'r', 'LineWidth', 2, ...
    'HandleVisibility', 'off');
% text(endpoint_ax, endpoint_tf_mean + 20, endpoint_loss_ylim(2) - 0.004, ...
%     '$\langle t_f\rangle$', 'Interpreter', 'latex', 'Color', 'r', ...
%     'FontSize', 18, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

yyaxis(endpoint_ax, 'right');
errorbar(endpoint_ax, endpoint_tc, endpoint_flatness_mean, endpoint_flatness_sem, '-s', ...
    'Color', endpoint_orange, 'LineWidth', 2.2, 'MarkerSize', 5, ...
    'MarkerFaceColor', endpoint_orange, 'CapSize', 5);
ylabel(endpoint_ax, 'Flatness $F$', ...
    'Interpreter', 'latex', 'Color', endpoint_orange, 'FontSize', label_font);
hold(endpoint_ax, 'on')
grid(endpoint_ax, 'on'); 
box(endpoint_ax, 'on');
axis(endpoint_ax, 'square')
set(endpoint_ax, 'FontName', 'Times New Roman', 'FontSize', axis_font, 'LineWidth', 0.8);
endpoint_ax.YAxis(1).Color = endpoint_blue;
endpoint_ax.YAxis(2).Color = endpoint_orange;
xlabel(endpoint_ax, 'continuation time $t_c$', 'Interpreter', 'latex', 'FontSize', label_font);
xlim(endpoint_ax, [0, 1000]);
ylim(endpoint_ax, [1.8, 3.8]);
yticks(endpoint_ax, 2:0.5:3.5)
save_fig1_png(fig_E, fullfile(fig1_export_dir, 'Fig1E_endpoint_quality_vs_tc.png'), fig1_export_resolution);


%% Define Jaccard similarity
function similarity = jaccard_similarity(list1, list2)
    % Convert to sets (unique elements)
    set1 = unique(list1);
    set2 = unique(list2);
    
    % Calculate size of intersection and union
    intersection_size = numel(intersect(set1, set2));
    union_size = numel(union(set1, set2));
    
    % Calculate Jaccard similarity
    similarity = intersection_size / union_size;
end

function save_fig1_png(fig_handle, output_file, resolution)
    % Use print instead of exportgraphics so Illustrator receives the full
    % fixed-size canvas rather than a tight-cropped image.
    set(fig_handle, 'Color', 'w', 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
    set(findall(fig_handle, 'Type', 'axes'), 'Color', 'w');
    drawnow;
    print(fig_handle, output_file, '-dpng', sprintf('-r%d', resolution));
end

function endpoint_stats = load_endpoint_stats(endpoint_stats_root)
    summary_files = dir(fullfile(endpoint_stats_root, 'r*', 'endpoint_stats_summary.csv'));
    if isempty(summary_files)
        error('No endpoint_stats_summary.csv files found under %s', endpoint_stats_root);
    end

    endpoint_stats = table();
    for file_idx = 1:numel(summary_files)
        summary_path = fullfile(summary_files(file_idx).folder, summary_files(file_idx).name);
        current_table = readtable(summary_path);
        endpoint_stats = [endpoint_stats; current_table]; %#ok<AGROW>
    end
end
