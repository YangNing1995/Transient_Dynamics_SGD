% Results of continuing training with GD at different moments along a single SGD trajectory
% Neuron-wise relative sharpness version of Fig. 4.
% This script replaces the Hessian flatness used in Fig4_4S_escaping_behavior.m
% with F_nw = -log10(S_nw), loaded from save_neuron_wise_flatness_*.mat.
% =========================================================================

%% Load original SGD trajectory
Data_dir = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Train_with_different_hyperparas\';
bs = 50;
lr = 0.05;
realization_num = 2;

% Load metrics for the specific run. Flatness is loaded below from the
% neuron-wise relative sharpness calculation, not from Hessian spectra.
load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_metrics_repeat', num2str(realization_num), '.mat'));

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

%% Load various GD trajectories 
% (Start points at 0, 20, 40...1000 Iterations, 51 points total)
% (Each trajectory samples 101 time points from 0:20:2000)
Data_dir = strcat('E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Continue_training\', ...
    'bs', num2str(bs), '_lr', num2str(lr), '_repeat', num2str(realization_num), '_ct/');
if ~exist(strcat(Data_dir, 'save_neuron_wise_flatness_continue_all.mat'), 'file')
    error('Missing neuron-wise flatness results in %s', Data_dir);
end

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

% Load data loop
for i = 1:num_solutions
    load(strcat(Data_dir, 'save_metrics_ct', num2str(20*(i-1)), '.mat'));
    
    Train_loss_all(i, :) = train_loss;
    Test_loss_all(i, :) = test_loss;
    Train_acc_all(i, :) = train_accuracy;
    Test_acc_all(i, :) = test_accuracy;
    Weights_all(i, :, :) = weight_all;
    
    if iscell(wrong_indices)
        Wrong_indices_all(i, :) = wrong_indices;
    else
        Wrong_indices_all(i, :) = num2cell(wrong_indices, 2)';
    end
end

% Calculate flatness from neuron-wise relative sharpness.
% save_neuron_wise_flatness_base.mat contains the original SGD trajectory.
% save_neuron_wise_flatness_continue_all.mat contains continuation trajectories
% with rows corresponding to t_c = 0:20:1000 and columns to local GD time.
load(strcat(Data_dir, 'save_neuron_wise_flatness_base.mat'), 'Flatness');
Flatness_base = squeeze(Flatness);
load(strcat(Data_dir, 'save_neuron_wise_flatness_continue_all.mat'), 'Flatness_all');

% Extract end-point metrics for each solution
Flatness_end = zeros(num_solutions, 1);
Test_acc_end = zeros(num_solutions, 1);
Test_loss_end = zeros(num_solutions, 1);

for i = 1:num_solutions
    Flatness_end(i, 1) = Flatness_all(i, end-i+1);
    Test_acc_end(i, 1) = Test_acc_all(i, end-i+1);
    Test_loss_end(i, 1) = Test_loss_all(i, end-i+1);
end

% Use data-adaptive axis limits because neuron-wise flatness is on a
% different numerical scale from the original Hessian flatness.
Flatness_values = [Flatness_base(:); Flatness_all(:)];
Flatness_values = Flatness_values(isfinite(Flatness_values));
Flatness_ylim = [floor(min(Flatness_values)*10)/10, ceil(max(Flatness_values)*10)/10];
Flatness_end_ylim = [floor(min(Flatness_end)*100)/100, ceil(max(Flatness_end)*100)/100];

%% Loss landscape & weight dynamics
solution_list = [1, 2, 3, 5, 9, 17, 33];
Weights_all_reshaped = reshape(Weights_all, [num_solutions*num_iterations, 2500]);

% PCA Projection ('score' is the projection in the PCA coordinate system)
[coeff, score, latent, ~, ~, mu] = pca(Weights_all_reshaped); 
Weights_pca = reshape(score, [num_solutions, num_iterations, 2500]);
Weights_base_pca = (Weights_base - mu) * coeff;

% Color definition
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;

% 2D + Loss Visualization
figure('unit','points','PaperUnits','points', 'position', [100 100 600 600])
plot3(Weights_base_pca(:, 1), Weights_base_pca(:, 2), Train_loss_base, '-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
hold on 
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    plot3(Weights_pca(sol, :, 1), Weights_pca(sol, :, 2), Train_loss_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end

% Plot Initial point
scatter3(Weights_base_pca(1,1), Weights_base_pca(1, 2), Train_loss_base(1), 100, 'k', 'marker', 'pentagram', 'DisplayName', 'Initial point'); 

% Settings
legend('Fontname', 'Times New Roman', 'Fontsize', 16, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
grid on
box on
view(-148, 21)
hx = xlabel('$\theta_\mathrm{pc1}$', 'Interpreter', 'latex','Units','normalized');
hy = ylabel('$\theta_\mathrm{pc2}$', 'Interpreter', 'latex','Units','normalized');
zlabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex')
zlim([min(Train_loss_all, [], 'all'), max(Train_loss_all, [], 'all')])
set(gca, 'Fontname', 'Times New Roman', 'Zscale', 'log', 'Fontsize', 24);

% Adjust label positions
set(hx, 'Position', [ 0.1956    0.0083         0],'Units','normalized');   % Move closer in y direction
set(hy, 'Position', [  0.8044    -0.01         0],'Units','normalized');   % Move closer in x direction

%% Train loss
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
semilogy(iterations_list, Train_loss_base,'-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
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

%% Test loss
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
semilogy(iterations_list, Test_loss_base,'-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
hold on
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    semilogy(iterations_list + continue_iterations_list(sol), Test_loss_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end
legend('Fontname', 'Times New Roman', 'Fontsize', 18, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
xlim([0 2000])
yticks([0.5 1 2])
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$\mathcal{L}_\mathrm{test}$', 'Interpreter', 'latex')
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);

%% Train accuracy
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(iterations_list, Train_acc_base,'-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
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
plot(iterations_list, Test_acc_base,'-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
hold on
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    plot(iterations_list + continue_iterations_list(sol), Test_acc_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end
legend('Fontname', 'Times New Roman', 'Fontsize', 18, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
xlim([0 2000])
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

%% Flatness
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(iterations_list, Flatness_base,'-', 'color', color_seq(2,:), 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'SGD trajectory');
hold on
for i = 1 : length(solution_list)
    sol = solution_list(i);
    legend_label = strcat('$t_c=', int2str(continue_iterations_list(sol)),'$');
    plot(iterations_list + continue_iterations_list(sol), Flatness_all(sol, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5, 'DisplayName', legend_label)
end
legend('Fontname', 'Times New Roman', 'Fontsize', 18, 'location', 'best', 'Orientation','vertical', 'Interpreter', 'latex')
xlim([0 2000])
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('Neuron-wise flatness $F_{\rm nw}$', 'Interpreter', 'latex')
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);

%% Flatness new (Combined figure)
color_seq = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; ...
              204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])

% -------- Main plot: neuron-wise flatness trajectory --------
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
ylabel('Neuron-wise flatness $F_{\rm nw}$', 'Interpreter', 'latex')
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlim([0 2000])
ylim(Flatness_ylim)

% -------- Inset plot: final neuron-wise flatness vs t_c --------
axes('Position', [0.18, 0.45, 0.45, 0.45]); % [x, y, width, height]
plot(continue_iterations_list(1:2:end), Flatness_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('$F_{\rm nw}$', 'Interpreter', 'latex')
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

% Plot: neuron-wise flatness vs test loss
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
ylabel('Neuron-wise flatness $F_{\rm nw}$', 'Interpreter', 'latex')
ylim(Flatness_end_ylim)
box on
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 30);
axis square

% Plot: neuron-wise flatness vs test accuracy
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
ylabel('Neuron-wise flatness $F_{\rm nw}$', 'Interpreter', 'latex')
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

% Neuron-wise flatness vs t_c
figure('unit','points','PaperUnits','points', 'position', [100 100 600 450])
plot(continue_iterations_list(1:2:end), Flatness_end(1:2:end), ...
    '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', color_seq(2,:));
grid on
axis square
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('Neuron-wise flatness $F_{\rm nw}$', 'Interpreter', 'latex')
xlim([0 1000])
ylim(Flatness_end_ylim)
box on 


%% Calculate pairwise Jaccard similarity between 51 solutions after training completion
% (Ratio of intersection to union of the two sets)
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
figure('unit','points','PaperUnits','points', 'position', [100 100 600 600])
imagesc(similarity_matrix(1:max_solution_num, 1:max_solution_num))
colormap(viridis)
colorbar
axis square

tick_labels = 0: 200: 20*(max_solution_num-1);
set(gca, 'XTick', 1:10:max_solution_num, 'XTickLabel', tick_labels, 'YTick', 1:10:max_solution_num, 'YTickLabel', tick_labels,'YDir','normal');
xlabel('$t_c$', 'Interpreter', 'latex')
ylabel('$t_c$', 'Interpreter', 'latex')
set(gca,'Fontname', 'Times New Roman','Fontsize', 30);

% ----------- Highlight the region where t_c >= t_freeze -----------
t_freeze = 440;
index = t_freeze/20+1;
hold on
% Rectangle position [x, y, w, h]. Bottom-left corner (index-0.5) to cover appropriate cells.
rectangle('Position', [index-0.5, index-0.5, max_solution_num-index+1, max_solution_num-index+1], ...
          'EdgeColor',  'r', 'LineWidth', 2);
hold off

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
