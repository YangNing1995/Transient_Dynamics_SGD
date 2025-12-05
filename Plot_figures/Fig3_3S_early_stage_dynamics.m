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

% Store freeze time
idx_freeze_all = zeros(length(bs_list), length(lr_list), num_realizations);   % [#bs, #lr, #irealization]

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
            
            Positions = find(ismember(save_iterations, iterations_list));
            
            % Store metrics
            Iteration_all(i, j, :) = save_iterations(1, Positions);
            Train_loss_all(i, j, k, :) = train_loss(1, Positions);
            Test_loss_all(i, j, k, :) = test_loss(1, Positions);
            Train_acc_all(i, j, k, :) = train_accuracy(1, Positions);
            Test_acc_all(i, j, k, :) = test_accuracy(1, Positions);
            Wrong_indices_all(i, j, k, :) = wrong_indices(1, Positions);
            Weights_all(i, j, k, :, :) = weight_all(Positions, :);
            Hessian_all(i, j, k, :, :) = Hessian';
               
            % Determine freeze time (defined as the first moment when train loss drops below L_c = 0.1)
            idx_freeze = find(train_loss <= 0.1, 1, 'first'); 
            if ~isempty(idx_freeze)
                idx_freeze_all(i, j, k) = save_iterations(idx_freeze); 
            else
                idx_freeze_all(i, j, k) = nan;
            end
        end
    end
end

% Calculate Flatness and Convergence Probability
Flatness_all = prod(Hessian_all(:, :, :, :, 1:10), 5).^(-1/10);
Convergence_probability = sum(Train_acc_all(:,:,:,end) == 1, 3) / num_realizations;

%% Train loss & train acc (single trajectory)
bs = 5;
lr = 6;
num = 2;
max_iteration = 100/(lr_list(lr));
iterations_list = linspace(0, max_iteration, num_timepoints);

% Remove singleton dimensions to ensure it's a vector
train_loss = squeeze(Train_loss_all(bs, lr, num, :));
train_acc  = squeeze(Train_acc_all(bs, lr, num, :));

% Journal style color scheme
loss_color = [204 51 136]/255;   % Magenta
acc_color  = [0 153 136]/255;    % Teal/Cyan-Green

figure('unit','points','PaperUnits','points', 'position', [100 100 500 400])

% --- Left Axis (Loss) ---
yyaxis left
semilogy(iterations_list, train_loss, '-', ...
    'LineWidth', 1.8, 'Color', loss_color)
ylabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex', 'Color', loss_color)
ylim([0 max(train_loss)*1.1])
ax = gca;
ax.YColor = loss_color;   % Match left axis color with the curve

% Horizontal line at loss=0.1
yline(0.1, '--', 'Color', 'k', 'LineWidth', 1.2)

% Find intersection (first position where loss <= 0.1)
idx = find(train_loss <= 0.1, 1);
if ~isempty(idx)
    t_cross = iterations_list(idx);
    hold on
    plot(t_cross, train_loss(idx), 'o', 'MarkerSize', 7, ...
         'MarkerEdgeColor', loss_color, 'LineWidth', 2)
    xline(t_cross, '--k', 'LineWidth', 1.2)   % Keep vertical dashed line black to highlight reference
end

% --- Right Axis (Accuracy) ---
yyaxis right
plot(iterations_list, train_acc, '-', ...
    'LineWidth', 1.8, 'Color', acc_color)
ylabel('$Acc_\mathrm{train}$', 'Interpreter', 'latex', 'Color', acc_color)
ax.YColor = acc_color;   % Match right axis color with the curve

% --- Unified Formatting ---
xlabel('Iteration $t$', 'Interpreter', 'latex')
xlim([0 1500])
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 22)
grid off
box on

%% Freeze time (L_c = 0.1)
idx_freeze_mean = mean(idx_freeze_all, 3, "omitmissing");
idx_freeze_rescaled = idx_freeze_mean .* lr_list;
freeze_min = min(idx_freeze_rescaled(Convergence_probability~=0));
freeze_max = max(idx_freeze_rescaled(Convergence_probability~=0));

figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(idx_freeze_rescaled, [freeze_min freeze_max]) 
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
cb = colorbar;
cb.Label.String = '$\eta\langle t_\mathrm{freeze} \rangle$';
cb.Label.Units = 'normalized';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 18;
cb.Label.Rotation = 0; 
cb.Label.Position = [0.5, 1.1, 0]; 
xlabel('Learning rate $\eta$', 'Interpreter', 'latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter', 'latex')
yticklabels(string(bs_list));
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 18);

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

%% Mean train loss 
lr = 5; 
max_iteration = 100/(lr_list(lr));
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

xlim([0 3000])
ylim([0 4.6147]) 
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
lr = 5; 
max_iteration = 100/(lr_list(lr));
iterations_list = linspace(0, max_iteration, num_timepoints);
Flatness_mean = squeeze(mean(Flatness_all(:, lr, :, :), 3));

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
xlim([0 3000])
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$\langle F\rangle$', 'Interpreter', 'latex');
box on
grid on
legend(h, 'Location', 'northwest',...
         'FontName', 'Times New Roman',...
         'FontSize', 16, 'Interpreter', 'latex');  
legend box on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 22);

%% Analyze freeze time with different thresholds (L_c)
% Define a list of thresholds to analyze
Lc_list = [0.2, 0.05, 0.01]; 

for lc_idx = 1:length(Lc_list)
    L_c = Lc_list(lc_idx);
    
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
    
    % Rescale by learning rate (t * eta) as per your original logic
    idx_freeze_rescaled = idx_freeze_mean .* lr_list;
    freeze_min = min(idx_freeze_rescaled(Convergence_probability~=0));
    freeze_max = max(idx_freeze_rescaled(Convergence_probability~=0));

    figure('unit','points','PaperUnits','points', 'position', [100+50*lc_idx, 100, 400, 400])
    
    % Adjust CLim (color limits) based on data range or fix it manually
    imagesc(idx_freeze_rescaled, [freeze_min freeze_max]) 
    
    set(gca, 'YDir', 'normal');
    axis square
    colormap(viridis)
    
    % Setup Colorbar
    cb = colorbar;
    cb.Label.String = '$\eta\langle t_\mathrm{freeze} \rangle$';
    cb.Label.Units = 'normalized';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 18;
    cb.Label.Rotation = 0; 
    cb.Label.Position = [0.5, 1.1, 0]; 
    
    % Axis Labels
    xlabel('Learning rate $\eta$', 'Interpreter','latex')
    xticklabels(string(lr_list));
    ylabel('Batch size $B$', 'Interpreter','latex')
    yticklabels(string(bs_list));
    set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);
    
    % Add Title to the Plot
    title(['$\mathcal{L}_c=' num2str(L_c) '$'], 'Interpreter', 'latex', 'FontSize', 16);
    
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
end