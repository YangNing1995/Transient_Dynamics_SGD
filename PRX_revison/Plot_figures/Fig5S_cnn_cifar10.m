% Analysis of all SGD trajectories starting from the same initial point
% Data in this script are simulated using a CNN trained on the FULL CIFAR-10 dataset.
% -------------------------------------------------------------------------
% Hyperparameter Combinations
% Bs = [500, 200, 100, 50, 20]
% Lr = [0.002, 0.005, 0.01, 0.02, 0.05]
% Run 10 repetitions for each hyperparameter group
% -------------------------------------------------------------------------
bs_list = [500, 200, 100, 50, 20];
lr_list = [0.002, 0.005, 0.01, 0.02, 0.05];
num_bs = length(bs_list);
num_lr = length(lr_list);
num_realizations = 10;
num_timepoints = 101;

%% Load data
Data_dir = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\save_data_cnn\';

% Pre-allocate arrays
Iteration_all = zeros(length(bs_list), length(lr_list), num_timepoints);                       % [#bs, #lr, #iteration]
Train_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);    % [#bs, #lr, #realization, #iteration]
Test_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Train_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Test_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);      % [#bs, #lr, #realization, #iteration]
Wrong_indices_all = cell(length(bs_list), length(lr_list), num_realizations, num_timepoints);  % {#bs, #lr, #realization, #iteration}
Weights_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 1280); % [#bs, #lr, #realization, #iteration, #weight]
Hessian_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 1280); % [#bs, #lr, #realization, #iteration, #weight]

for i = 1:length(bs_list)
    bs = bs_list(i);
    for j = 1:length(lr_list)
        lr = lr_list(j);
        max_iteration = 1000/lr;
        iterations_list = linspace(0, max_iteration, num_timepoints);
        for k = 1:num_realizations
            % Load data based on batch size and repetition index
            load(strcat(Data_dir, 'bs',num2str(bs),'_lr', num2str(lr), '/save_metrics_repeat', num2str(k),'.mat'));
            load(strcat(Data_dir, 'bs',num2str(bs),'_lr', num2str(lr), '/save_hessian_repeat', num2str(k),'.mat'));

            % Extract specific timepoints
            Positions = find(ismember(save_iterations, iterations_list));
            
            % Store data into arrays
            Iteration_all(i, j, :) = save_iterations(1, Positions);
            Train_loss_all(i, j, k, :) = train_loss(1, Positions);
            Test_loss_all(i, j, k, :) = test_loss(1, Positions);
            Train_acc_all(i, j, k, :) = train_accuracy(1, Positions);
            Test_acc_all(i, j, k, :) = test_accuracy(1, Positions);
            Wrong_indices_all(i, j, k, :) = wrong_indices(1, Positions);
            Weights_all(i, j, k, :, :) = weight_all(Positions, :);
            Hessian_all(i, j, k, :, :) = Hessian';
        end
    end
end

%% 3D Loss landscape & weight dynamic visualization 
% (Visualize different hyper-parameters for one realization)
Realization_index = 1;
Weights_all_onerun = reshape(Weights_all(:, :, Realization_index, :, :), [num_bs*num_lr*num_timepoints, 1280]);

% PCA Projection (score contains projections in PCA coordinate system)
[coeff, score, latent] = pca(Weights_all_onerun); 
Weights_pca = reshape(score, [num_bs, num_lr, num_timepoints, 1280]);

% 2D PCA + Loss Visualization
figure('unit', 'points', 'PaperUnits', 'points', 'position', [100 100 600 600])

% Define colors and markers
colors = [68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119]/255;
markers = {'x', 'o', '+', '*', 's'}; 

% Loop to plot trajectories
for i = 1: num_bs 
    for j = 1: num_lr 
        % Plot trajectory
        plot3(squeeze(Weights_pca(i, j, :, 1)), squeeze(Weights_pca(i, j, :, 2)), squeeze(Train_loss_all(i, j, Realization_index, :)), ...
              'color', colors(j, :), 'marker', markers{i}, 'LineWidth', 0.1, 'MarkerSize', 5)
        hold on
    end
end

% Plot initial position (using a black pentagram)
initial_position = squeeze(Weights_pca(1, 1, 1, :));  
plot3(initial_position(1), initial_position(2), squeeze(Train_loss_all(1, 1, Realization_index, 1)), ...
      'color', 'k', 'marker', 'pentagram', 'MarkerSize', 10); 

% Set view, grid, and labels
view(-230, 21)
grid on
box on

hx = xlabel('$\theta_\mathrm{pc1}$', 'Interpreter', 'latex', 'Units', 'normalized');
hy = ylabel('$\theta_\mathrm{pc2}$', 'Interpreter', 'latex', 'Units', 'normalized');
zlabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex')
zlim([min(Train_loss_all, [], 'all'), max(Train_loss_all, [], 'all')])

% Set font and axes properties
set(gca, 'Fontname', 'Times New Roman', 'Zscale', 'log', 'Fontsize', 24);

% Adjust label position closer to axes
posx = get(hx, 'Position');
posy = get(hy, 'Position');
set(hx, 'Position', posx + [0 0 0]);         % Move closer in y direction
set(hy, 'Position', posy + [-0.08 0.08 0]);  % Move closer in x direction

%% legend of loss landscape
% Legend
colors = [68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119]/255;
markers = {'x', 'o', '+', '*', 's'}; 

figure('unit','points','PaperUnits','points', 'position', [200 200 150 400]);
for j = 1 : num_lr
    % 展示颜色
    subplot(6, 2, 2*j-1);
    line([0.2 0.8], [1 1], 'color', colors(j, :), 'LineWidth', 2, 'MarkerSize', 5);
    xlim([0 1])
    axis off;
    title(['$\eta = {' num2str(lr_list(j)) '}$'], 'Interpreter', 'latex', 'Fontname', 'Times New Roman', 'Fontsize', 14);

    % 展示标记符号
    subplot(6, 2, 2*j);
    plot(0, 0, 'marker', markers{j}, 'color', 'k', 'MarkerSize', 12);  
    axis off;
    title(['$B = {' num2str(bs_list(j)) '}$'], 'Interpreter', 'latex', 'Fontname', 'Times New Roman', 'Fontsize', 14);
end
subplot(6, 2, 12);
plot(0, 0, 'marker', 'pentagram', 'color', 'k', 'MarkerSize', 12);  
axis off;
title('Initial point', 'Fontname', 'Times New Roman', 'Fontsize', 14);

%% Jaccard similarity
Realization_index = 1;
bs_list = [500, 200, 100, 50, 20];
lr_list = [0.002, 0.005, 0.01, 0.02, 0.05];

% Select required data subset
Wrong_indices_final = Wrong_indices_all(:, :, Realization_index, end); 
Wrong_indices_final_reshape = reshape(Wrong_indices_final, [num_bs*num_lr, 1]);  

% Calculate Jaccard similarity matrix
Jaccard_similarities_all = zeros(length(Wrong_indices_final_reshape), length(Wrong_indices_final_reshape));
for i = 1:length(Wrong_indices_final_reshape)
    for j = 1:length(Wrong_indices_final_reshape)
        if i ~= j  % Calculate only for different solution pairs
            Jaccard_similarities_all(i, j) = jaccard_similarity(Wrong_indices_final_reshape{i}, Wrong_indices_final_reshape{j});
        else
            Jaccard_similarities_all(i, j) = 1;  % Similarity is 1 for the same solution
        end
    end
end

% Plot Heatmap
figure('unit', 'points', 'PaperUnits', 'points', 'position', [100 100 600 600])
imagesc(Jaccard_similarities_all)
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$Sim_J$';
cb.Label.Units = 'normalized';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 16;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.07, 0]; 

% Set ticks and labels for axes
xticks(1:25)  
yticks(1:25)  

% Generate labels for axes representing (bs, lr) combinations
% Calculate all (bs, lr) combinations used in the subset
[bs_grid, lr_grid] = meshgrid(bs_list, lr_list);  % Generate combination grid

% Create label strings
labels = strings(25, 1); 
for i = 1:25
    labels(i) = ['$\eta =' num2str(lr_grid(i)) ',  B =' num2str(bs_grid(i)) '$']; 
end

% Set LaTeX interpreter and apply labels
set(gca, 'TickLabelInterpreter', 'latex');
xticklabels(labels);
yticklabels(labels);

% Set font and font size
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 14);

%% Hessian spectrum 
% Select learning-rate index and realization index
lr_index = 3;
realization_index = 1;

% Extract Hessian values with lr and realization fixed
Hessian_lr_fixed = squeeze(Hessian_all(:, lr_index, realization_index, end, :));

% Prepare figure
figure('unit','points','PaperUnits','points', 'position', [100 100 500 400])
colors = get(gca, 'ColorOrder');

num_lines = size(Hessian_lr_fixed, 1);
h = zeros(1, num_lines);

% Plot Hessian diagonal entries for each batch size B
for i = 1:num_lines
    y_mean = Hessian_lr_fixed(i, :);

    % Legend name with batch size
    legend_name = strcat('$B = ', num2str(bs_list(i)), '$');

    % Plot with log-scaled x-axis
    h(i) = semilogx(1:1280, y_mean, '-o', ...
                    'LineWidth', 1.0, ...
                    'Color', colors(i, :), ...
                    'DisplayName', legend_name);
    hold on;
end

% Axis settings
xlim([0 1280]);
xticks([1 10 100 1000])
xlabel('Index $i$', 'Interpreter', 'latex');
ylabel('$\lambda_i(\mathbf{H})$', 'Interpreter', 'latex');
box on;
grid on;

% Legend settings
legend(h, 'Location', 'best', ...
          'FontName', 'Times New Roman', ...
          'FontSize', 16, ...
          'Interpreter', 'latex');
legend box on;

% Global axis font settings
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 22);


%% Calculate basic metrics
Weights_diff = squeeze(Weights_all(:, :, :, end, :) - Weights_all(:, :, :, 1, :));
Weights_distance = sqrt(sum(Weights_diff.^2, 4));

% Flatness calculation: geometric mean of the first 10 eigenvalues of Hessian
Flatness = prod(Hessian_all(:, :, :, :, 1:10), 5).^(-1/10);
Flatness(isinf(Flatness)) = NaN;

% Calculate Means
Mean_final_train_loss = mean(Train_loss_all(:,:,:,end), 3);
Mean_final_test_loss = mean(Test_loss_all(:,:,:,end), 3);
Mean_final_train_acc = mean(Train_acc_all(:,:,:,end), 3);
Mean_final_test_acc = mean(Test_acc_all(:,:,:,end), 3);
Mean_final_Hessian = mean(Hessian_all(:,:,:,end,1), 3);
Mean_final_flatness = mean(Flatness(:,:,:,end), 3, 'omitnan');
Mean_weights_distance = mean(Weights_distance, 3);

% RMS endpoint radius r_theta (across-repeat spread)
Weights_final = squeeze(Weights_all(:, :, :, end, :));  % [bs, lr, repeat, dim]
RMS_radius = zeros(num_bs, num_lr);
for i = 1:num_bs
    for j = 1:num_lr
        W = squeeze(Weights_final(i, j, :, :));  % [repeat, dim]
        centroid = mean(W, 1);                    % [1, dim]
        RMS_radius(i, j) = sqrt(mean(sum((W - centroid).^2, 2)));
    end
end

% Calculate Convergence Probability
Convergence_probability = sum(Train_acc_all(:,:,:,end)==1, 3)/num_realizations;

% Calculate Max stats
Max_final_test_acc = max(Test_acc_all(:,:,:,end), [], 3);
Max_final_flatness = max(Flatness(:,:,:,end), [], 3);

%% ===== Panel (c): Freezing time (loss proxy, L_c = 0.1) =====
% Define the single threshold manually here
L_c = 0.1; 

% Initialize storage for the current threshold
idx_freeze_current_Lc = zeros(num_bs, num_lr, num_realizations);

% Calculate freeze time for each trajectory
for i = 1:num_bs
    for j = 1:num_lr
        for k = 1:num_realizations
            % Retrieve the loss trajectory
            traj_loss = squeeze(Train_loss_all(i, j, k, :));
            
            % Find the first index where loss drops below L_c
            idx = find(traj_loss <= L_c, 1, 'first');
            
            if ~isempty(idx)
                % Map index to actual iteration number
                idx_freeze_current_Lc(i, j, k) = Iteration_all(i, j, idx);
            else
                idx_freeze_current_Lc(i, j, k) = NaN; % Did not reach threshold
            end
        end
    end
end

% ---------------------------------------------------------------------
% Visualization: Heatmap for Current L_c
% ---------------------------------------------------------------------
% Calculate mean freeze time (ignoring NaNs)
idx_freeze_mean = mean(idx_freeze_current_Lc, 3, 'omitnan');

% Rescale by learning rate (t * eta)
idx_freeze_rescaled = idx_freeze_mean .* lr_list;

% Calculate min/max for color limits (excluding non-converged areas)
valid_mask = (Convergence_probability ~= 0);
if any(valid_mask(:))
    freeze_min = min(idx_freeze_rescaled(valid_mask));
    freeze_max = max(idx_freeze_rescaled(valid_mask));
else
    freeze_min = 0; freeze_max = 1; % Fallback if nothing converged
end

% Create Figure (Removed lc_idx dependency in position)
figure('unit','points','PaperUnits','points', 'position', [150, 100, 400, 400])

% Adjust CLim (color limits) based on data range or fix it manually
imagesc(idx_freeze_rescaled, [freeze_min freeze_max]) 

set(gca, 'YDir', 'normal');
axis square
colormap(viridis)

% Setup Colorbar
cb = colorbar;
cb.Label.String = '$\eta\langle t_f \rangle$';
cb.Label.Units = 'normalized';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 

% Axis Labels
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticks(1:5)
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

% -------- Overlay Convergence_probability information --------
[m, n] = size(Convergence_probability);
for i = 1:m
    for j = 1:n
        % If you want to gray out or shade specific areas
        if Convergence_probability(i,j) == 0
            % Draw gray rectangle (semi-transparent)
            patch([j-0.5 j+0.5 j+0.5 j-0.5], ...
                  [i-0.5 i-0.5 i+0.5 i+0.5], ...
                  [0.5 0.5 0.5], ...        % Gray RGB color
                  'EdgeColor', 'none');     % No edge color
        end
    end
end
hold off;

%% ===== Panel (d): RMS endpoint radius r_theta =====
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(log10(RMS_radius))
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$r_\theta$';
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0;
cb.Label.Position = [0.5, 1.1, 0];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticks(1:1:5)
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

% Gray overlay for divergent cells
hold on;
for i = 1:num_bs
    for j = 1:num_lr
        if Convergence_probability(i,j) == 0
            patch([j-0.5 j+0.5 j+0.5 j-0.5], ...
                  [i-0.5 i-0.5 i+0.5 i+0.5], ...
                  [0.70 0.70 0.70], 'EdgeColor', 'none');
        end
    end
end
hold off;

%% ===== Panel (e): Mean final flatness <F> =====
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Mean_final_flatness)
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\langle F \rangle$';
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0;
cb.Label.Position = [0.5, 1.1, 0];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticks(1:1:5)
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

% Gray overlay for divergent cells
hold on;
for i = 1:num_bs
    for j = 1:num_lr
        if Convergence_probability(i,j) == 0
            patch([j-0.5 j+0.5 j+0.5 j-0.5], ...
                  [i-0.5 i-0.5 i+0.5 i+0.5], ...
                  [0.70 0.70 0.70], 'EdgeColor', 'none');
        end
    end
end
hold off;

%% ===== Panel (f): Max test accuracy =====
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Max_final_test_acc)
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\max(Acc_\mathrm{test})$';
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0;
cb.Label.Position = [0.5, 1.1, 0];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticks(1:1:5)
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

% Gray overlay for divergent cells
hold on;
for i = 1:num_bs
    for j = 1:num_lr
        if Convergence_probability(i,j) == 0
            patch([j-0.5 j+0.5 j+0.5 j-0.5], ...
                  [i-0.5 i-0.5 i+0.5 i+0.5], ...
                  [0.70 0.70 0.70], 'EdgeColor', 'none');
        end
    end
end
hold off;

%% ===== Panel (g): Mean train loss =====
lr = 2;  % lr=0.005
max_iteration = 1000/(lr_list(lr));
iterations_list = linspace(0, max_iteration, num_timepoints);
Train_loss_mean = squeeze(mean(Train_loss_all(:, lr, :, :), 3));

% Create new figure
figure('unit','points','PaperUnits','points', 'position', [100 100 500 400]) 
colors = get(gca, 'ColorOrder');
num_lines = size(Train_loss_mean, 1);
h = zeros(1, num_lines);

for i = 1:num_lines
    y_mean = Train_loss_mean(i, :);
    legend_name = strcat('$B = ', num2str(bs_list(i)), '$');
    h(i) = semilogy(iterations_list, y_mean,...
                   'LineWidth', 1.5,...
                   'Color', colors(i,:),...
                   'DisplayName', legend_name); 
    hold on
end

% Horizontal line at loss=0.1
yline(0.1, '--', 'Color', 'k', 'LineWidth', 1.2)
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$\langle\mathcal{L}_\mathrm{train}\rangle$', 'Interpreter', 'latex');
yticks([1e-2 1e-1 1])
yticklabels({'10^{-2}', '10^{-1}', '10^{0}'})
box on
grid on
legend(h, 'Location', 'northeast',...
         'FontName', 'Times New Roman',...
         'FontSize', 16, 'Interpreter', 'latex');  
legend box on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 22, 'Yscale', 'log');

%% Mean flatness
lr = 2;  % lr=0.005
max_iteration = 1000/(lr_list(lr));
iterations_list = linspace(0, max_iteration, num_timepoints);
Flatness_mean = squeeze(mean(Flatness(:, lr, :, :), 3));

% Create new figure
figure('unit','points','PaperUnits','points', 'position', [100 100 500 400]) 
colors = get(gca, 'ColorOrder');
num_lines = size(Train_loss_mean, 1);
h = zeros(1, num_lines);

for i = 1:num_lines
    y_mean = Flatness_mean(i, :);
    legend_name = strcat('$B = ', num2str(bs_list(i)), '$');
    h(i) = plot(iterations_list, y_mean,...
                   'LineWidth', 1.5,...
                   'Color', colors(i,:),...
                   'DisplayName', legend_name); 
    hold on
end

% Axis settings
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$\langle F\rangle$', 'Interpreter', 'latex');
yticks([0 2 4 6])
box on
grid on
legend(h, 'Location', 'northwest',...
         'FontName', 'Times New Roman',...
         'FontSize', 16, 'Interpreter', 'latex');  
legend box on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 22);


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