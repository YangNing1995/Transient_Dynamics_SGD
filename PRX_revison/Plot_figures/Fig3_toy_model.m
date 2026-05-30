% Simulation results of the two-valley toy model
% =========================================================================

%% Paths and settings
script_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(script_dir, '..', '..');
out_dir = fullfile(repo_root, 'PRX_revison', 'Figures_revision', 'Fig3_toy_model');
export_resolution = 600;

addpath(script_dir);  % for viridis.m
addpath(fullfile(repo_root, 'Plot_figures'));  % for mycolormap.mat fallback

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% -------------------------------------------------------------------------
% Model Parameters & Function Definitions
% -------------------------------------------------------------------------
x1 = 0.8; 
x2 = 0.65;
x0 = 1;
f0 = 1;
y_b = 30;
y_f = 20;
y_d = 100;
L_d = 1;

% Define component functions (Vectorized for array inputs)
g1 = @(y) 2*x1.*y./(y + y_b);
g2 = @(y) 2*x2.*y./(y + y_b);
f1 = @(y) f0.*(x1.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
f2 = @(y) f0.*(x2.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);

% Base loss drift term
L0 = @(y) L_d.*exp(-y/y_d) + x0^2/f0;  

% Define piecewise loss function L(x,y)
% Note: Using logical indexing (x>=0) for piecewise construction
L = @(x, y) L0(y) + (x>=0).*x.*(x-g1(y))./f1(y) + (x<0).*x.*(x+g2(y))./f2(y);

%% Visualization of the loss function
% Define grid range and resolution
x = linspace(-1.5, 1.5, 1000);
y = linspace(0, 100, 1000);
[X, Y] = meshgrid(x, y);  

% Calculate Loss function values on the grid
Z = L(X, Y);  

% Calculate locations of local minima paths along y
x_min_plus = 0.5 * g1(y);       % Minimum for x > 0
x_min_minus = -0.5 * g2(y);     % Minimum for x < 0
Z_min_plus = L(x_min_plus, y);
Z_min_minus = L(x_min_minus, y);

% Plot surface with contours
figB = figure('unit','points','PaperUnits','points', 'position', [100 100 400 400]);

% surfc plots both the 3D surface and the contour plot underneath
surfc(X, Y, Z, 'EdgeColor', 'none');   
hold on

% Overlay the paths of the two valleys (local minima)
plot3(x_min_plus, y, Z_min_plus, '-.', 'LineWidth', 3, 'Color', 'w')
plot3(x_min_minus, y, Z_min_minus, '-.', 'LineWidth', 3, 'Color', 'w')

% Labels and formatting
xlabel('$x$', 'Interpreter', 'LaTex');
ylabel('$y$', 'Interpreter', 'LaTex');
zlabel('$\mathcal{L}(x, y)$', 'Interpreter', 'LaTex');

axis square
grid on;
view(-145, 31) % Set view angle

% Set axes properties (Log scale for Z, reverse X direction)
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 18, ...
    'ZScale', 'log', 'XDir', 'reverse', 'ColorScale', 'log');

save_fig3_png(figB, fullfile(out_dir, 'Fig3B_loss_landscape.png'), export_resolution);

%% Convergence probability to the flat valley (varying hyperparameters)
% Load custom colormap and simulation results
% Run ../Two_valleys_model/Phase_diagram.m to save the results
load(fullfile(repo_root, 'Plot_figures', 'mycolormap.mat'))
load(fullfile(repo_root, 'Two_valleys_model', 'result_PhsDgm_Ld1_yd100_x10.8_x20.65.mat'))   % Cov ~ H
% load(fullfile(repo_root, 'Two_valleys_model', 'result_PhsDgm_Ld1_yd100_x10.8_x20.65_H2.mat')) % Cov ~ H^2

figC = figure('unit','points','PaperUnits','points', 'position', [100 100 400 400]);

% Plot heatmap of convergence probability (final_position)
imagesc(learning_rate_list, noise_strength_list, final_position');

% Apply custom colormap
colormap(mycolormap.blue_white_red_improved)
% clim([0 1]) % Optional: Fix color limits

% Axis labels
xlabel('Learning rate $\eta$', 'Interpreter', 'LaTex')
ylabel('Noise strength $\sigma$', 'Interpreter', 'LaTex')

% Set axis limits slightly wider than data range
xlim([min(learning_rate_list)/1.13, max(learning_rate_list)*1.13])
ylim([min(noise_strength_list)/1.13, max(noise_strength_list)*1.13])
% xlim([min(learning_rate_list)/1.27, max(learning_rate_list)*1.27])
% ylim([min(noise_strength_list)/1.27, max(noise_strength_list)*1.27])
% xticks([1e-3 1e-2 1e-1])
% yticks([1e-3 1e-2 1e-1])

% Add label for the colorbar variable
text(1.07, 1.06, '$P_\mathrm{flat}$', 'Interpreter', 'LaTex', ...
    'Units', 'normalized', 'HorizontalAlignment', 'center', 'Fontsize', 20)

colorbar
axis square
box on

% Set log scale for X and Y axes
set(gca, 'XScale', 'log', 'YScale', 'log', 'YDir', 'normal', ...
    'Fontname', 'Times New Roman', 'Fontsize', 18);

save_fig3_png(figC, fullfile(out_dir, 'Fig3C_flat_probability.png'), export_resolution);

%% Freezing time (varying hyperparameters)
% Rescale freezing time by learning rate: t_rescaled = eta * t_mean
freezing_time_rescaled = learning_rate_list' .* freezing_time_mean;

figD = figure('unit','points','PaperUnits','points', 'position', [100 100 400 400]);

% Plot heatmap of rescaled freezing time
imagesc(learning_rate_list, noise_strength_list, freezing_time_rescaled');

colormap(viridis)
% Set color limits from 1 to max value
clim([1, max(freezing_time_rescaled, [], 'all')])

% Axis labels
xlabel('Learning rate $\eta$', 'Interpreter', 'LaTex')
ylabel('Noise strength $\sigma$', 'Interpreter', 'LaTex')

% Set axis limits
xlim([min(learning_rate_list)/1.13, max(learning_rate_list)*1.13])
ylim([min(noise_strength_list)/1.13, max(noise_strength_list)*1.13])
% xlim([min(learning_rate_list)/1.27, max(learning_rate_list)*1.27])
% ylim([min(noise_strength_list)/1.27, max(noise_strength_list)*1.27])
% xticks([1e-3 1e-2 1e-1])
% yticks([1e-3 1e-2 1e-1])

% Add label for the colorbar variable
text(1.07, 1.06, '$\eta \langle t_f \rangle$', 'Interpreter', 'LaTex', ...
    'Units', 'normalized', 'HorizontalAlignment', 'center', 'Fontsize', 20)

colorbar
axis square
box on

% Set log scale for axes and color
set(gca, 'XScale', 'log', 'YScale', 'log', 'YDir', 'normal', ...
    'Fontname', 'Times New Roman', 'Fontsize', 18, 'ColorScale', 'log');

save_fig3_png(figD, fullfile(out_dir, 'Fig3D_freezing_time.png'), export_resolution);

fprintf('Exported Fig3 toy-model panels to %s\n', out_dir);


function save_fig3_png(fig_handle, output_file, resolution)
    % Keep a consistent full canvas for manual assembly.
    set(fig_handle, 'Color', 'w', 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
    set(findall(fig_handle, 'Type', 'axes'), 'Color', 'w');
    drawnow;
    print(fig_handle, output_file, '-dpng', sprintf('-r%d', resolution));
end
