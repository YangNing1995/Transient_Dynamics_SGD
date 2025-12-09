%% Fig6A: Illustration of effective loss landscape
%  Two-Valley Loss Landscape Visualization
%  Plot L(x,y) and L_eff(x,y) along a fixed y-section
% ===============================

% ----- Model parameters -----
x1 = 0.8;
x2 = 0.65;
x0 = 1;
f0 = 1;
y_b = 30;
y_f = 20;
y_d = 100;
L_d = 1;

% Choose a y-section
y_fixed = 40;   % you can change this to e.g., 10, 30, 60 to see different cross-sections

% ----- Define component functions -----
g1 = @(y) 2*x1.*y./(y + y_b);
g2 = @(y) 2*x2.*y./(y + y_b);
f1 = @(y) f0.*(x1.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
f2 = @(y) f0.*(x2.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
L0 = @(y) L_d.*exp(-y/y_d) + x0^2/f0;  % Base loss drift term

% ----- Define piecewise loss L(x,y) -----
L = @(x, y) arrayfun(@(xx) ...
    L0(y) + (xx>=0)*xx.*(xx - g1(y))./f1(y) + ...
    (xx<0)*xx.*(xx + g2(y))./f2(y), x);

% ----- Compute L(x, y_fixed) -----
x_vals = linspace(-1.0, 1.2, 500);
L_vals = L(x_vals, y_fixed);

% ----- Compute effective loss L_eff(x, y_fixed) -----
% f1, f2 evaluated at y_fixed
f1_val = f1(y_fixed);
f2_val = f2(y_fixed);

% For x > 0 use valley 1, for x < 0 use valley 2
L_eff = zeros(size(x_vals));
for i = 1:length(x_vals)
    xx = x_vals(i);
    if xx >= 0
        ratio = sqrt(f1_val / f2_val);
        L_eff(i) = ratio * L(xx, y_fixed) + (1 - ratio) * L0(y_fixed);
    else
        ratio = sqrt(f2_val / f1_val);
        L_eff(i) = ratio * L(xx, y_fixed) + (1 - ratio) * L0(y_fixed);
    end
end

% ----- Plot -----
figure('Position',[200 200 600 400]); hold on;
plot(x_vals, L_vals, 'b-', 'LineWidth', 2.5, 'DisplayName', '$\mathcal{L}(x,y)$');
plot(x_vals, L_eff, 'r-', 'LineWidth', 2.5, 'DisplayName', '$\mathcal{L}_\mathrm{eff}(x,y)$');
plot(x_vals, L_eff - L_vals + 1.3, 'k-', 'LineWidth', 2.5, 'DisplayName', '$\mathcal{L}_\mathrm{SGD}(x, y)$');

xlabel('$x$','FontSize',18,'FontWeight','bold', 'Interpreter','latex');
ylabel('Loss','FontSize',18);
xlim([-0.8 1.2])
% ylim([1.1 1.9])
legend('Location','best', 'Interpreter','latex');
legend box off
box off;
grid off;
axis off;    
set(gca,'FontSize',14,'LineWidth',1.2, 'Fontname', 'Times New Roman');

%% Fig6B: Comparison of Simulation vs Theoretical NESS/ESS Escape Rates
% Run ../Two_valleys_model/P_flat_simulation.m to save the rseults
% === List of files to load ===
file_list = {
    '../Two_valleys_model/P_right_eta0.010_sigma0.100.mat', ...
    '../Two_valleys_model/P_right_eta0.010_sigma0.050.mat', ...
    '../Two_valleys_model/P_right_eta0.010_sigma0.010.mat'
};

% === Pre-allocate storage ===
n_files = numel(file_list);
y_all   = cell(n_files, 1);
P_sim_all = cell(n_files, 1);
P_th_all  = cell(n_files, 1);
P_ess_all  = cell(n_files, 1);
Delta_s_all = zeros(n_files, 1);

% === Loop to load data and compute theoretical results ===
for i = 1:n_files
    if exist(file_list{i}, 'file')
        data = load(file_list{i});
        sigma = data.noise_strength;
        Delta_s = data.learning_rate * data.noise_strength;
        
        % Extract variables from file (adjust according to your saved names)
        y_vals = data.mean_y;
        P_sim  = data.P_right_simulation;
        
        % Calculate theoretical curves
        y_interp = linspace(0, 20, 100);
        
        ness_escape_rates_vs_time_left  = NESS_escape_rate('left',  y_interp, f1, f2, g1, g2, L, Delta_s);
        ness_escape_rates_vs_time_right = NESS_escape_rate('right', y_interp, f1, f2, g1, g2, L, Delta_s);
        P_th = ness_escape_rates_vs_time_left ./ (ness_escape_rates_vs_time_right + ness_escape_rates_vs_time_left);
        
        ess_escape_rates_vs_time_left  = ESS_escape_rate('left',  y_interp, f1, f2, g1, g2, L, Delta_s);
        ess_escape_rates_vs_time_right = ESS_escape_rate('right', y_interp, f1, f2, g1, g2, L, Delta_s);
        P_ess = ess_escape_rates_vs_time_left ./ (ess_escape_rates_vs_time_right + ess_escape_rates_vs_time_left);
        
        % Store results
        y_all{i} = y_vals;
        P_sim_all{i} = P_sim;
        P_th_all{i} = P_th;
        P_ess_all{i} = P_ess;
        Delta_s_all(i) = Delta_s;
    else
        warning('File %s not found. Skipping.', file_list{i});
    end
end

% === Plotting ===
custom_colors = [
    0.00, 0.45, 0.74;  % R,G,B for Blue
    0.85, 0.33, 0.10;  % R,G,B for Orange
    0.47, 0.67, 0.19   % R,G,B for Green
];

line_styles_th = {':', ':', ':'}; 
lin_width = 2.5; % Thicken theoretical curves

legend_name_sim = ["Sim: $\Delta_S=1\mathrm{e}{-3}$", "Sim: $\Delta_S=5\mathrm{e}{-4}$", "Sim: $\Delta_S=1\mathrm{e}{-4}$"];
legend_name_th = ["NESS: $\Delta_S=1\mathrm{e}{-3}$", "NESS: $\Delta_S=5\mathrm{e}{-4}$", "NESS: $\Delta_S=1\mathrm{e}{-4}$"];

figure('Units','points','PaperUnits','points','Position',[100 100 500 400]);
hold on;

% Get indices of valid loaded data
valid_indices = find(~cellfun(@isempty, y_all));

for i = reshape(valid_indices, 1, [])
    y_vals = y_all{i};
    P_sim  = P_sim_all{i};
    
    % 1. Plot simulation results: as background, thinner lines, higher transparency, and not shown in legend
    plot(y_vals, P_sim, '-', ...
        'Color', [custom_colors(i,:) 0.5], ... 
        'LineWidth', 2.0, ...
        'DisplayName', legend_name_sim(i)); 
end

for i = reshape(valid_indices, 1, [])
    P_th   = P_th_all{i};
    % 2. Plot theoretical results: use custom colors and line styles, and create legend for them
    plot(y_interp, P_th, line_styles_th{i}, ... % Use different line styles
        'Color', custom_colors(i,:), ...
        'LineWidth', lin_width, ...
        'DisplayName', legend_name_th(i));
end

if ~isempty(valid_indices)
    % Plot ESS for the last dataset as an example
    plot(y_interp, P_ess_all{valid_indices(end)}, ':', ...
        'Color', [0.5 0.5 0.5], ...
        'LineWidth', lin_width, ...
        'DisplayName', 'ESS');
end

xlabel('$y$', 'Interpreter','LaTeX');
ylabel('$P_\mathrm{flat}$', 'Interpreter','LaTeX');
ylim([0.5 1.02]) % Slightly raise the upper limit of y-axis to avoid overlap between lines and border
xlim([1 20])
yticks([0.5 0.75 1])
legend('Location','best','Interpreter','LaTeX','Box','on','FontSize', 14);
grid on;
box on; % Ensure border is always visible
hold off;
set(gca, 'FontName','Times New Roman', 'FontSize',22, ...
    'Box','on', 'XScale','log');

%% Fig6C. Visualization of P_flat^tr,SGD Surface
% Model Parameters
x2 = 0.65;          % Sharp valley width parameter (base)
x0 = 1;
f0 = 1;
y_b = 30;
y_f = 20;
y_d = 100;
L_d = 1;
epsilon = 1e-1;     % Small constant epsilon (permissible deviation)

% Define Grid for Independent Variables
% Delta_S: Product of learning rate and noise strength (Log scale)
dS_vals = logspace(-6, 0, 50); 

% Gamma: Flatness ratio f1/f2 = x1^2/x2^2 (Linear scale, >1)
gamma_vals = linspace(1.0, 1.2, 50); 

[Delta_S, Gamma] = meshgrid(dS_vals, gamma_vals);

% Calculate Intermediate Variable Phi
% Note: Phi depends on (x1^2 - x2^2).
diff_x2 = x2^2 .* (Gamma - 1); 

% Calculate Phi based on the derived approximation
% Phi approx = [2(x1^2 - x2^2) / 27 y_b] * [L_d/y_d + 8 x0^2 / 27 y_f]
term1 = (2 .* diff_x2) ./ (27 * y_b);
term2 = (L_d / y_d) + (8 * x0^2) / (27 * y_f);
Phi = term1 .* term2;

% Calculate Probability P
% Eq: P = [ 1 + gamma^(-0.5) * ( sqrt(Delta_S) / (epsilon * Phi) )^(1-gamma) ]^(-1)

Numerator_Inside_Log = sqrt(Delta_S);
Denominator_Inside_Log = epsilon .* Phi;
Base_Term = Numerator_Inside_Log ./ Denominator_Inside_Log;

Exponent = 1 - Gamma;
Prefactor = Gamma.^(-0.5);

% Combine to get final probability
P_flat_tr = (1 + Prefactor .* (Base_Term .^ Exponent)).^(-1);

% Plotting
figure('Units', 'points', 'Position', [100, 100, 500, 400]);

% Draw Surface
s = surf(Gamma, Delta_S, P_flat_tr);

% Appearance Settings
s.EdgeColor = 'none';       % Remove grid lines
s.FaceColor = 'interp';     % Interpolated shading
colormap(parula);           

% --- Axis Settings ---
set(gca, 'YScale', 'log');  % Log scale for Delta_S
ylim([min(dS_vals), max(dS_vals)]);
xlim([min(gamma_vals), max(gamma_vals)]);
yticks([1e-6, 1e-3, 1])
xticks([1, 1.1, 1.2])
axis square

% Define Z-axis limits and ticks (to be shared with Colorbar)
z_limits = [0.45, 1];
z_ticks  = [0.5, 0.75, 1];

% Apply to Z-axis
zlim(z_limits);
zticks(z_ticks);

% --- Labels ---
ylabel('$\Delta_S$', 'Interpreter', 'latex', 'FontSize', 16);
xlabel('$\gamma = f_1/f_2$', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('$P_{\mathrm{flat}}^{\mathrm{tr,SGD}}$', 'Interpreter', 'latex', 'FontSize', 16);

% --- View and Grid ---
view(-40, 25);
grid on;
box on;

% --- Colorbar Settings (Synced with Z-axis) ---
cb = colorbar;
clim(z_limits);         % Set color data limits to match Z-axis limits
cb.Ticks = z_ticks;     % Set colorbar ticks to match Z-axis ticks

% Font Settings
set(gca, 'FontName', 'Times New Roman', 'FontSize', 18);

%% Helper Functions
function rate = NESS_escape_rate(x_start, y_trajectory, f1, f2, g1, g2, Loss, Delta_s)
    % Calculate NESS escape rate
    rate = zeros(1,length(y_trajectory));
    for t_idx = 1:length(y_trajectory)
        y_val = y_trajectory(t_idx);
        if x_start == "right"  % in the right valley
            f_val = f1(y_val);
            % Loss at the minimum of the right valley
            L_min = Loss(g1(y_val)/2, y_val);
            % Loss at the barrier (x=0)
            L_barrier = Loss(0, y_val);
            % Escape time 
            rate(t_idx) = 1/((pi/2) * f_val * erfi(sqrt((L_barrier - L_min) / (2*Delta_s/f_val))));
        else   % in the left valley
            % Flatness of the left valley
            f_val = f2(y_val);
            % Loss at the minimum of the left valley
            L_min = Loss(-g2(y_val)/2, y_val);
            % Loss at the barrier (x=0)
            L_barrier = Loss(0, y_val);
            % Escape time 
            rate(t_idx) = 1/((pi/2) * f_val * erfi(sqrt((L_barrier - L_min) / (2*Delta_s/f_val))));
        end
    end
end

function rate = ESS_escape_rate(x_start, y_trajectory, f1, f2, g1, g2, Loss, Delta_s)
    % Calculate ESS escape rate
    rate = zeros(1,length(y_trajectory));
    yi = y_trajectory(1);
    fi = sqrt(f1(yi)*f2(yi));
    for t_idx = 1:length(y_trajectory)
        y_val = y_trajectory(t_idx);
        if x_start == "right"  % in the right valley
            f_val = f1(y_val);
            % Loss at the minimum of the right valley
            L_min = Loss(g1(y_val)/2, y_val);
            % Loss at the barrier (x=0)
            L_barrier = Loss(0, y_val);
            % Escape time
            rate(t_idx) = 1/((pi/2) * f_val * erfi(sqrt((L_barrier - L_min) / (2*Delta_s/fi))));
        else   % in the left valley
            % Flatness of the left valley
            f_val = f2(y_val);
            % Loss at the minimum of the left valley
            L_min = Loss(-g2(y_val)/2, y_val);
            % Loss at the barrier (x=0)
            L_barrier = Loss(0, y_val);
            % Escape time
            rate(t_idx) = 1/((pi/2) * f_val * erfi(sqrt((L_barrier - L_min) / (2*Delta_s/fi))));
        end
    end
end
