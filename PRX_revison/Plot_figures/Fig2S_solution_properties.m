% Fig S2: Convergence stability and geometric properties of solutions
% Six-panel heatmap (a)-(f) on the same 6x6 grid as main text Fig. 2.
% -------------------------------------------------------------------------
clear; clc; close all;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);  % for viridis.m

Data_dir = fullfile(script_dir, '..', '..', '..', 'Data', 'Train_with_different_hyperparas');
save_dir = fullfile(script_dir, '..', 'Figures_SI', 'Fig2S_solution_properties');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% Full grid (for data loading)
bs_list_full = [1000, 500, 200, 100, 50, 20, 10];
lr_list_full = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
num_bs_full = length(bs_list_full);
num_lr_full = length(lr_list_full);
num_realizations = 20;
num_timepoints = 101;

%% Display grid (aligned with main text Fig. 2, no B=1000 or lr=0.001)
bs_display = [500, 200, 100, 50, 20, 10];
lr_display = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
bs_idx = 2:7;  % indices into full grid
lr_idx = 2:7;

%% Load data
Train_loss_all = zeros(num_bs_full, num_lr_full, num_realizations, num_timepoints);
Test_loss_all  = zeros(num_bs_full, num_lr_full, num_realizations, num_timepoints);
Train_acc_all  = zeros(num_bs_full, num_lr_full, num_realizations, num_timepoints);
Weights_all    = zeros(num_bs_full, num_lr_full, num_realizations, num_timepoints, 2500);

for i = 1:num_bs_full
    bs = bs_list_full(i);
    for j = 1:num_lr_full
        lr = lr_list_full(j);
        for k = 1:num_realizations
            if bs == 1000 && k > 1
                load(fullfile(Data_dir, sprintf('bs%d_lr%g', bs, lr), 'save_metrics_repeat1.mat'));
            else
                load(fullfile(Data_dir, sprintf('bs%d_lr%g', bs, lr), sprintf('save_metrics_repeat%d.mat', k)));
            end

            max_iteration = 100 / lr;
            iterations_list = linspace(0, max_iteration, num_timepoints);
            Positions = find(ismember(save_iterations, iterations_list));

            Train_loss_all(i, j, k, :) = train_loss(1, Positions);
            Test_loss_all(i, j, k, :)  = test_loss(1, Positions);
            Train_acc_all(i, j, k, :)  = train_accuracy(1, Positions);
            Weights_all(i, j, k, :, :) = weight_all(Positions, :);
        end
    end
end

%% Compute metrics on full grid, then slice
% Weight displacement
Weights_diff = squeeze(Weights_all(:, :, :, end, :) - Weights_all(:, :, :, 1, :));
Mean_weights_distance = mean(sqrt(sum(Weights_diff.^2, 4)), 3);

% Cosine similarity
Weights_matrix1 = permute(Weights_diff, [3 4 1 2]);
Weights_matrix2 = permute(Weights_diff, [4 3 1 2]);
numerator   = pagemtimes(Weights_matrix1, Weights_matrix2);
norm1       = sqrt(sum(Weights_matrix1.^2, 2));
norm2       = sqrt(sum(Weights_matrix2.^2, 1));
denominator = pagemtimes(norm1, norm2);
Cosine_similarities = numerator ./ denominator;
Sum_sim = squeeze(sum(sum(Cosine_similarities, 1), 2));
Mean_cosine_similarities = (Sum_sim - num_realizations) / (num_realizations^2 - num_realizations);

% Convergence probability
Convergence_probability = sum(Train_acc_all(:,:,:,end) == 1, 3) / num_realizations;

% Mean final metrics
Mean_final_train_acc  = mean(Train_acc_all(:,:,:,end), 3);
Mean_final_train_loss = mean(Train_loss_all(:,:,:,end), 3);
Mean_final_test_loss  = mean(Test_loss_all(:,:,:,end), 3);

%% Slice to display grid
P_conv_sub    = Convergence_probability(bs_idx, lr_idx);
Sim_cos_sub   = Mean_cosine_similarities(bs_idx, lr_idx);
W_dist_sub    = Mean_weights_distance(bs_idx, lr_idx);
Acc_train_sub = Mean_final_train_acc(bs_idx, lr_idx);
L_train_sub   = Mean_final_train_loss(bs_idx, lr_idx);
L_test_sub    = Mean_final_test_loss(bs_idx, lr_idx);

%% Plot panels
export_res = 600;

% (a) Convergence probability
draw_panel(P_conv_sub, '$P_\mathrm{conv}$', [0 1], 0:0.2:1, ...
    bs_display, lr_display, save_dir, export_res, 'Fig2S_a_convergence_probability', false);

% (b) Mean cosine similarity
draw_panel(Sim_cos_sub, '$\langle \mathrm{Sim}_\mathrm{cos} \rangle$', [], 0:0.2:1, ...
    bs_display, lr_display, save_dir, export_res, 'Fig2S_b_cosine_similarity', false);

% (c) Mean weight distance
draw_panel(W_dist_sub, '$\langle \|\Delta\theta\|_2 \rangle$', [], [2 2.5 3], ...
    bs_display, lr_display, save_dir, export_res, 'Fig2S_c_weight_distance', false);

% (d) Mean final training accuracy
draw_panel(Acc_train_sub, '$\langle Acc_\mathrm{train} \rangle$', [], [], ...
    bs_display, lr_display, save_dir, export_res, 'Fig2S_d_train_accuracy', false);

% (e) Mean final training loss (log scale)
draw_panel(L_train_sub, '$\langle \mathcal{L}_\mathrm{train} \rangle$', [], [], ...
    bs_display, lr_display, save_dir, export_res, 'Fig2S_e_train_loss', true);

% (f) Mean final test loss (log scale)
draw_panel(L_test_sub, '$\langle \mathcal{L}_\mathrm{test} \rangle$', [], [], ...
    bs_display, lr_display, save_dir, export_res, 'Fig2S_f_test_loss', true);

fprintf('All Fig S2 panels saved to:\n  %s\n', save_dir);

%% ========== Local function ==========
function draw_panel(Data, cbar_label, clim_vals, cbar_ticks, ...
                    bs_display, lr_display, save_dir, export_res, filename, use_log_color)

    fig = figure('Units', 'points', 'PaperUnits', 'points', ...
                 'Position', [100 100 400 400]);
    imagesc(Data);
    if ~isempty(clim_vals), clim(clim_vals); end
    set(gca, 'YDir', 'normal');
    axis square;
    colormap(viridis);
    if use_log_color, set(gca, 'ColorScale', 'log'); end

    cb = colorbar;
    cb.Label.String      = cbar_label;
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize    = 18;
    cb.Label.Rotation    = 0;
    cb.Label.Units       = 'normalized';
    cb.Label.Position    = [0.5, 1.1, 0];
    if ~isempty(cbar_ticks), cb.Ticks = cbar_ticks; end

    xlabel('Learning rate $\eta$', 'Interpreter', 'latex');
    ylabel('Batch size $B$', 'Interpreter', 'latex');
    xticks(1:length(lr_display));
    yticks(1:length(bs_display));
    xticklabels(compose('%g', lr_display));
    yticklabels(string(bs_display));
    set(gca, 'Fontname', 'Times New Roman', 'FontSize', 18);

    print(fig, fullfile(save_dir, [filename '.png']), '-dpng', sprintf('-r%d', export_res));
    close(fig);
end
