% Analysis of all SGD trajectories starting from the same initial point
% -------------------------------------------------------------------------
% Hyperparameter Combinations
% Bs = [1000, 500, 200, 100, 50, 20, 10]
% Lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
% Run 20 repetitions for each hyperparameter group (except GD, i.e., Bs=1000)
% -------------------------------------------------------------------------

bs_list = [1000, 500, 200, 100, 50, 20, 10];
lr_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
num_bs = length(bs_list);
num_lr = length(lr_list);
num_realizations = 20;
num_timepoints = 101;

%% Load Data
Data_dir = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Train_with_different_hyperparas\';

% Pre-allocate arrays
Iteration_all = zeros(length(bs_list), length(lr_list), num_timepoints);                        % [#bs, #lr, #iteration]
Train_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Test_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);      % [#bs, #lr, #realization, #iteration]
Train_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);      % [#bs, #lr, #realization, #iteration]
Test_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);       % [#bs, #lr, #realization, #iteration]
Wrong_indices_all = cell(length(bs_list), length(lr_list), num_realizations, num_timepoints);   % {#bs, #lr, #realization, #iteration}
Weights_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 2500);  % [#bs, #lr, #realization, #iteration, #weight]
Hessian_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 2500);  % [#bs, #lr, #realization, #iteration, #weight]

for i = 1:length(bs_list)
    bs = bs_list(i);
    for j = 1:length(lr_list)
        lr = lr_list(j);
        max_iteration = 100/lr;
        iterations_list = linspace(0, max_iteration, num_timepoints);
        
        for k = 1:num_realizations
            % Load data based on batch size and repetition index
            if bs == 1000 && k > 1
                load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_metrics_repeat1.mat'));
                load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_hessian_repeat1.mat'));             
            else
                load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_metrics_repeat', num2str(k), '.mat'));
                load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_hessian_repeat', num2str(k), '.mat'));
            end
            
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
Realization_index = 5;
Weights_all_onerun = reshape(Weights_all(:, :, Realization_index, :, :), [num_bs*num_lr*num_timepoints, 2500]);

% PCA Projection (score contains projections in PCA coordinate system)
[coeff, score, latent] = pca(Weights_all_onerun); 
Weights_pca = reshape(score, [num_bs, num_lr, num_timepoints, 2500]);

% 2D PCA + Loss Visualization
figure('unit', 'points', 'PaperUnits', 'points', 'position', [100 100 600 600])

% Define colors and markers
colors = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
markers = {'.', 'x', 'o', '+', '*', 's', 'd'}; 

% Loop to plot trajectories
for i = 3: num_bs - 1  % Indices corresponding to Bs: [200, 100, 50, 20]
    for j = 3: num_lr - 1  % Indices corresponding to Lr: [0.005, 0.01, 0.02, 0.05]
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

%% Legend of loss landscape
% Re-define colors and markers for legend figure
colors = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
markers = {'.', 'x', 'o', '+', '*', 's', 'd'}; 

figure('unit', 'points', 'PaperUnits', 'points', 'position', [200 200 150 400]);
for j = 3 : num_lr - 1
    % Display colors (Learning Rate)
    subplot(5, 2, 2*j-5);
    line([0.2 0.8], [1 1], 'color', colors(j, :), 'LineWidth', 2, 'MarkerSize', 5);
    xlim([0 1])
    axis off;
    title(['$\eta = {' num2str(lr_list(j)) '}$'], 'Interpreter', 'latex', 'Fontname', 'Times New Roman', 'Fontsize', 14);
    
    % Display markers (Batch Size)
    subplot(5, 2, 2*j-4);
    plot(0, 0, 'marker', markers{j}, 'color', 'k', 'MarkerSize', 12);  
    axis off;
    title(['$B = {' num2str(bs_list(j)) '}$'], 'Interpreter', 'latex', 'Fontname', 'Times New Roman', 'Fontsize', 14);
end

% Display Initial Point Legend
subplot(5, 2, 10);
plot(0, 0, 'marker', 'pentagram', 'color', 'k', 'MarkerSize', 12);  
axis off;
title('Initial point', 'Fontname', 'Times New Roman', 'Fontsize', 14);

%% Jaccard similarity
Realization_index = 5;
bs_list = [1000, 500, 200, 100, 50, 20, 10];
lr_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];

% Select required data subset
Wrong_indices_final = Wrong_indices_all(3:6, 3:6, Realization_index, end); 
Wrong_indices_final_reshape = reshape(Wrong_indices_final, [4*4, 1]);  % Reshape into 16 conditions

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
colorbar

% Set ticks and labels for axes
xticks(1:16)  
yticks(1:16)  

% Generate labels for axes representing (bs, lr) combinations
% Calculate all (bs, lr) combinations used in the subset
[bs_grid, lr_grid] = meshgrid(bs_list(3:6), lr_list(3:6));  % Generate combination grid

% Create label strings
labels = strings(16, 1); 
for i = 1:16
    labels(i) = ['$\eta =' num2str(lr_grid(i)) ',  B =' num2str(bs_grid(i)) '$']; 
end

% Set LaTeX interpreter and apply labels
set(gca, 'TickLabelInterpreter', 'latex');
xticklabels(labels);
yticklabels(labels);

% Set font and font size
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 14);

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