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

%% Load data
Data_dir = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Train_with_different_hyperparas\';

% Pre-allocate arrays
Iteration_all = zeros(length(bs_list), length(lr_list), num_timepoints);                       % [#bs, #lr, #iteration]
Train_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);    % [#bs, #lr, #realization, #iteration]
Test_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Train_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Test_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);      % [#bs, #lr, #realization, #iteration]
Wrong_indices_all = cell(length(bs_list), length(lr_list), num_realizations, num_timepoints);  % {#bs, #lr, #realization, #iteration}
Weights_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 2500); % [#bs, #lr, #realization, #iteration, #weight]
Hessian_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 2500); % [#bs, #lr, #realization, #iteration, #weight]

for i = 1:length(bs_list)
    bs = bs_list(i);
    for j = 1:length(lr_list)
        lr = lr_list(j);
        max_iteration = 100/lr;
        iterations_list = linspace(0, max_iteration, num_timepoints);
        
        for k = 1:num_realizations
            % For Batch Size 1000 (GD), reuse the first repetition data for all k > 1
            if bs == 1000 && k > 1
                load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_metrics_repeat1.mat'));
                load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_hessian_repeat1.mat'));            
            else
                load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_metrics_repeat', num2str(k), '.mat'));
                load(strcat(Data_dir, 'bs', num2str(bs), '_lr', num2str(lr), '/save_hessian_repeat', num2str(k), '.mat'));
            end
            
            Positions = find(ismember(save_iterations, iterations_list));
            
            % Store data
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

% Calculate Convergence Probability
Convergence_probability = sum(Train_acc_all(:,:,:,end)==1, 3)/num_realizations;

% Calculate Min/Max stats
Min_final_train_loss = min(Train_loss_all(:,:,:,end), [], 3);
Min_final_test_loss = min(Test_loss_all(:,:,:,end), [], 3);
Max_final_train_acc = max(Train_acc_all(:,:,:,end), [], 3);
Max_final_test_acc = max(Test_acc_all(:,:,:,end), [], 3);
Max_final_Hessian = max(Hessian_all(:,:,:,end,1), [], 3);
Max_final_flatness = max(Flatness(:,:,:,end), [], 3);

%% Mean Jaccard similarity 
Wrong_indices_final = Wrong_indices_all(:, :, :, end);
Jaccard_similarities = zeros(length(bs_list), length(lr_list), num_realizations, num_realizations);

for m = 1:length(bs_list)
    for n = 1:length(lr_list)
        for i = 1:num_realizations
            for j = 1:num_realizations
                if i ~= j  % Calculate only for different solution pairs
                    Jaccard_similarities(m, n, i, j) = jaccard_similarity(Wrong_indices_final{m,n,i}, Wrong_indices_final{m,n,j});
                else
                    Jaccard_similarities(m, n, i, j) = 1;  % Similarity for the same solution is 1
                end
            end
        end
    end
end

Temp_jaccard_similarities = reshape(Jaccard_similarities, [7, 7, num_realizations^2]);
Mean_jaccard_similarities = squeeze(sum(Temp_jaccard_similarities, 3) - 20)/380;

figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Mean_jaccard_similarities)
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\langle Sim_J \rangle$';
cb.Label.Units = 'normalized';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

% -------- Overlay Convergence_probability information --------
hold on; % Hold layer
[m, n] = size(Convergence_probability);
for i = 1:m
    for j = 1:n
        % If you want to gray out or shade specific areas (where convergence prob is 0)
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

%% Mean cosine similarity 
Weights_matrix1 = permute(Weights_diff, [3 4 1 2]); 
Weights_matrix2 = permute(Weights_diff, [4 3 1 2]);

numerator = pagemtimes(Weights_matrix1, Weights_matrix2);
norm1 = sqrt(sum(Weights_matrix1.^2, 2)); % [20, 1, 7, 7]
norm2 = sqrt(sum(Weights_matrix2.^2, 1)); % [1, 20, 7, 7]
denominator = pagemtimes(norm1, norm2);
Cosine_similarities = numerator ./ denominator; 

Sum_similarities = squeeze(sum(sum(Cosine_similarities, 1), 2)); % [7, 7]
Mean_cosine_similarities = (Sum_similarities - 20) / 380;

figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Mean_cosine_similarities)
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\langle Sim_\mathrm{cos} \rangle$';
cb.Label.Units = 'normalized';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 

% Axis settings
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

%% Loss landscape & weight dynamic visualization (Bs = 50, Lr = 0.05)
bs_index = 5;
lr_index = 6;
Weights_all_replica = reshape(Weights_all(bs_index, lr_index, :, :, :), [num_realizations*num_timepoints, 2500]);
[~, score, ~] = pca(Weights_all_replica); % 'score' is the projection in the PCA coordinate system
Weights_pca = reshape(score, [num_realizations, num_timepoints, 2019]);
Train_loss_replica = reshape(Train_loss_all(bs_index, lr_index, :, :), [num_realizations, num_timepoints]);

% 2D + Loss Plot
figure('unit','points','PaperUnits','points', 'position', [100 100 600 600])
for i = 1:num_realizations
    plot3(Weights_pca(i, :, 1), Weights_pca(i, :, 2), Train_loss_replica(i, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5)
    hold on
end

% Plot initial position (using black pentagram)
initial_position_x = Weights_pca(1, 1, 1);
initial_position_y = Weights_pca(1, 1, 2);
plot3(initial_position_x, initial_position_y, squeeze(Train_loss_all(1, 1, 1, 1)), ...
      'color', 'k', 'marker', 'pentagram', 'MarkerSize', 10);  % Black pentagram marker

grid on
box on
view(45, 25)
hx = xlabel('$\theta_\mathrm{pc1}$', 'Interpreter', 'latex','Units','normalized');
hy = ylabel('$\theta_\mathrm{pc2}$', 'Interpreter', 'latex','Units','normalized');
zlabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex')
zticks([1e-2 1e-1 1])
zlim([min(Train_loss_all, [], 'all'), max(Train_loss_all, [], 'all')])

% Set font and axes
set(gca, 'Fontname', 'Times New Roman', 'Zscale', 'log', 'Fontsize', 24);
set(hx, 'Position', [ 0.1956    0.0083         0],'Units','normalized');   % Move closer in y direction
set(hy, 'Position', [  0.8044    -0.01         0],'Units','normalized');   % Move closer in x direction

%% Loss landscape & weight dynamic visualization (Bs = 50, Lr = 0.01)
bs_index = 5;
lr_index = 3;
Weights_all_replica = reshape(Weights_all(bs_index, lr_index, :, :, :), [num_realizations*num_timepoints, 2500]);
[~, score, ~] = pca(Weights_all_replica); % 'score' is the projection in the PCA coordinate system
Weights_pca = reshape(score, [num_realizations, num_timepoints, 2019]);
Train_loss_replica = reshape(Train_loss_all(bs_index, lr_index, :, :), [num_realizations, num_timepoints]);

% 2D + Loss Plot
figure('unit','points','PaperUnits','points', 'position', [100 100 600 600])
for i = 1:num_realizations
    plot3(Weights_pca(i, :, 1), Weights_pca(i, :, 2), Train_loss_replica(i, :), '-o', 'LineWidth', 0.5, 'MarkerSize', 5)
    hold on
end

% Plot initial position (using black pentagram)
initial_position_x = Weights_pca(1, 1, 1);
initial_position_y = Weights_pca(1, 1, 2);
plot3(initial_position_x, initial_position_y, squeeze(Train_loss_all(1, 1, 1, 1)), ...
      'color', 'k', 'marker', 'pentagram', 'MarkerSize', 10);  % Black pentagram marker

grid on
box on
view(45, 25)
hx = xlabel('$\theta_\mathrm{pc1}$', 'Interpreter', 'latex','Units','normalized');
hy = ylabel('$\theta_\mathrm{pc2}$', 'Interpreter', 'latex','Units','normalized');
zlabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex')
zticks([1e-2 1e-1 1])
zlim([min(Train_loss_all, [], 'all'), max(Train_loss_all, [], 'all')])

% Set font and axes
set(gca, 'Fontname', 'Times New Roman', 'Zscale', 'log', 'Fontsize', 24);
set(hx, 'Position', [ 0.1956    0.0083         0],'Units','normalized');   % Move closer in y direction
set(hy, 'Position', [  0.8044    -0.01         0],'Units','normalized');   % Move closer in x direction

%% Max final test accuracy 
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Max_final_test_acc, [0.895 0.93]) %[0.85 0.95]
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
cb.Ticks = [0.9 0.91 0.92 0.93];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

% -------- Overlay Convergence_probability information --------
hold on; % Hold layer
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

%% 1D violinplot for test accuracy
Test_acc_1d = squeeze(Test_acc_all(:, 4, :,end));  
figure('unit','points','PaperUnits','points', 'position', [100 100 250 400])
violinplot(Test_acc_1d')
xlabel('Batch size $B$', 'Interpreter','latex')
xticklabels(string(bs_list));
ylabel('$Acc_{\mathrm{test}}$', 'Interpreter','latex')
xlim([0.5, length(bs_list) + 0.5]);  % Adjust y-axis limits
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18, 'YDir', 'reverse');
view(270, 90);  % Rotate view by 90 degrees to make the plot horizontal
ax = gca;
ax.Position = [0.33, 0.2, 0.525, 0.636];  % Adjust the position of the axes to match the previous plot

%% Max final flatness heatmap
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Max_final_flatness, [1.663 5.9689]) %[1.5 6]
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\max(F)$';
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
cb.Ticks = [2, 3, 4, 5];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
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

%% 1D violinplot for flatness
Flatness_1d = squeeze(Flatness(:, 4, :, end));  
figure('unit','points','PaperUnits','points', 'position', [100 100 250 400])
violinplot(Flatness_1d')
xlabel('Batch size $B$', 'Interpreter','latex')
xticklabels(string(bs_list));
ylabel('Flatness $F$',  'Interpreter','latex')
xlim([0.5, length(bs_list) + 0.5]);  
ylim([1.3 3.7])
yticks([1.5 2.5 3.5])
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18, 'YDir', 'reverse');
view(270, 90);  % Rotate view by 90 degrees to make the plot horizontal
ax = gca;
ax.Position = [0.33, 0.2, 0.525, 0.636];  % Adjust the position of the axes to match the previous plot

%% Mean weight distance heatmap
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Mean_weights_distance) 
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\langle \|\Delta\theta\|_2 \rangle$'; 
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
cb.Ticks = [2 2.5 3];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

%% Convergence probability
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Convergence_probability) 
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$P_\mathrm{conv}$'; 
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
% cb.Ticks = [2 2.5 3];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

%% Mean final train accuracy
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Mean_final_train_acc) 
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\langle Acc_\mathrm{train} \rangle$'; 
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
% cb.Ticks = [2 2.5 3];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

%% Mean final train accuracy
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Mean_final_train_acc) 
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\langle Acc_\mathrm{train} \rangle$'; 
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
% cb.Ticks = [2 2.5 3];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

%% Mean final train loss
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Mean_final_train_loss) 
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\langle \mathcal{L}_\mathrm{train} \rangle$'; 
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
% cb.Ticks = [2 2.5 3];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18, 'ColorScale','log');

%% Mean final test loss
figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(Mean_final_test_loss) 
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\langle \mathcal{L}_\mathrm{test} \rangle$'; 
cb.Label.Interpreter = 'latex';
cb.Label.Units = 'normalized';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
% cb.Ticks = [2 2.5 3];
xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18, 'ColorScale','log');

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