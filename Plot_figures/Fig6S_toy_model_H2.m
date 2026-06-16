% Fig S6: Phase diagrams of the two-valley toy model (Cov ~ H^2)
% Two panels: (a) convergence probability, (b) rescaled freezing time
% =========================================================================

clear; clc; close all;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);  % for viridis.m
repo_root = fileparts(script_dir);

save_dir = fullfile(repo_root, 'Figures', 'Fig6S_toy_model_H2');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% Load simulation results (Cov ~ H^2)
load(fullfile(script_dir, 'mycolormap.mat'))
load(fullfile(repo_root, 'Two_valleys_model', 'result_PhsDgm_Ld1_yd100_x10.8_x20.65_H2.mat'))

%% (a) Convergence probability to the flat valley
fig = figure('unit','points','PaperUnits','points', 'position', [100 100 400 400]);

imagesc(learning_rate_list, noise_strength_list, final_position');

colormap(mycolormap.blue_white_red_improved)

xlabel('Learning rate $\eta$', 'Interpreter', 'LaTex')
ylabel('Noise strength $\sigma$', 'Interpreter', 'LaTex')

xlim([min(learning_rate_list)/1.13, max(learning_rate_list)*1.13])
ylim([min(noise_strength_list)/1.13, max(noise_strength_list)*1.13])

text(1.07, 1.06, '$P_\mathrm{flat}$', 'Interpreter', 'LaTex', ...
    'Units', 'normalized', 'HorizontalAlignment', 'center', 'Fontsize', 20)

colorbar
axis square
box on

set(gca, 'XScale', 'log', 'YScale', 'log', 'YDir', 'normal', ...
    'Fontname', 'Times New Roman', 'Fontsize', 18);

set(fig, 'Color', 'w', 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
drawnow; print(fig, fullfile(save_dir, 'Fig6S_a_convergence_probability.png'), '-dpng', '-r600');
close(fig);
fprintf('Saved: Fig6S_a_convergence_probability\n');

%% (b) Freezing time (rescaled by learning rate)
freezing_time_rescaled = learning_rate_list' .* freezing_time_mean;

fig = figure('unit','points','PaperUnits','points', 'position', [100 100 400 400]);

imagesc(learning_rate_list, noise_strength_list, freezing_time_rescaled');

colormap(viridis)
clim([1, max(freezing_time_rescaled, [], 'all')])

xlabel('Learning rate $\eta$', 'Interpreter', 'LaTex')
ylabel('Noise strength $\sigma$', 'Interpreter', 'LaTex')

xlim([min(learning_rate_list)/1.13, max(learning_rate_list)*1.13])
ylim([min(noise_strength_list)/1.13, max(noise_strength_list)*1.13])

text(1.07, 1.06, '$\eta \langle t_f \rangle$', 'Interpreter', 'LaTex', ...
    'Units', 'normalized', 'HorizontalAlignment', 'center', 'Fontsize', 20)

colorbar
axis square
box on

set(gca, 'XScale', 'log', 'YScale', 'log', 'YDir', 'normal', ...
    'Fontname', 'Times New Roman', 'Fontsize', 18, 'ColorScale', 'log');

set(fig, 'Color', 'w', 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
drawnow; print(fig, fullfile(save_dir, 'Fig6S_b_freezing_time.png'), '-dpng', '-r600');
close(fig);
fprintf('Saved: Fig6S_b_freezing_time\n');

fprintf('\nAll Fig S6 panels saved to:\n  %s\n', save_dir);
