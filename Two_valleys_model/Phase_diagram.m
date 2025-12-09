% Simulations of the convergence probability and freezing time in the two-valley model 
% across different learning rates and noise strengths
% =========================================================

%% Initialization and Parallel Pool Setup
close all;
clear;

% Check if a parallel pool exists, if not, create one
poolobj = gcp('nocreate'); 
if isempty(poolobj)
    parpool('local', 6);
    fprintf('Parallel pool started.\n');
else
    fprintf('Using existing parallel pool.\n');
end

%% Definition of Loss Function and Derivatives
seed = 42;
rng(seed);

% --- Model Parameters ---
x1 = 0.8; 
x2 = 0.65;
x0 = 1;
f0 = 1;
y_b = 30;   
y_f = 20;   
y_d = 100;  
L_d = 1;    
yi = 5;     

% --- Component Functions ---
g1 = @(y) 2*x1.*y./(y + y_b);
g2 = @(y) 2*x2.*y./(y + y_b);
f1 = @(y) f0.*(x1.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
f2 = @(y) f0.*(x2.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);

% Base loss term 
L0 = @(y) L_d.*exp(-y/y_d) + x0^2/f0;

% --- Derivatives ---

% Derivatives of g w.r.t y
dg1dy = @(y) (2*x1*y_b) ./ ((y + y_b).^2);
dg2dy = @(y) (2*x2*y_b) ./ ((y + y_b).^2);
d2g1dy2 = @(y) -(4*x1*y_b) ./ ((y + y_b).^3);
d2g2dy2 = @(y) -(4*x2*y_b) ./ ((y + y_b).^3);

% Derivatives of f w.r.t y
% Derived using quotient rule on ((y+yf)/(y+yb))^2
df1dy = @(y) (2*f0*(x1^2/x0^2) .* (y + y_f) * (y_b - y_f)) ./ ((y + y_b).^3);
df2dy = @(y) (2*f0*(x2^2/x0^2) .* (y + y_f) * (y_b - y_f)) ./ ((y + y_b).^3);

% Derivatives of L0 w.r.t y
dL0dy = @(y) -(L_d/y_d) .* exp(-y/y_d);
d2L0dy2 = @(y) (L_d/y_d^2) .* exp(-y/y_d);

% --- Loss Function L(x,y) ---
Loss = @(x, y) (x >= 0) .* (x.*(x - g1(y)) ./ f1(y) + L0(y)) + ...
               (x < 0)  .* (x.*(x + g2(y)) ./ f2(y) + L0(y));

% --- Gradients ---
dLdx = @(x, y) (x >= 0) .* ((2*x - g1(y)) ./ f1(y)) + ... 
               (x < 0)  .* ((2*x + g2(y)) ./ f2(y));

dLdy = @(x, y) dL0dy(y) - ...
               (x >= 0) .* (x.*(x-g1(y)).*df1dy(y)./(f1(y).^2) + x.*dg1dy(y)./f1(y)) - ...
               (x < 0)  .* (x.*(x+g2(y)).*df2dy(y)./(f2(y).^2) - x.*dg2dy(y)./f2(y));

% --- Full Hessian Matrix Components ---
H11 = @(x, y) (x >= 0) .* (2 ./ f1(y)) + ...
              (x < 0)  .* (2 ./ f2(y));

H12 = @(x, y) (x >= 0) .* (-dg1dy(y)./f1(y) - (2*x - g1(y)) .* df1dy(y) ./ (f1(y).^2)) + ...
              (x < 0)  .* (dg2dy(y)./f2(y) - (2*x + g2(y)) .* df2dy(y) ./ (f2(y).^2));

H22 = @(x, y) (x >= 0) .* (-x .* d2g1dy2(y)./ f1(y) + ...
                           2*x .* dg1dy(y) .* df1dy(y) ./ (f1(y).^2) + ...
                           2*x .* (x - g1(y)) .* (df1dy(y).^2) ./ (f1(y).^3) + ...
                           d2L0dy2(y)) + ...
              (x < 0)  .* (x .* d2g2dy2(y)./ f2(y) - ...
                           2*x .* dg2dy(y) .* df2dy(y) ./ (f2(y).^2) + ...
                           2*x .* (x + g2(y)) .* (df2dy(y).^2) ./ (f2(y).^3) + ...
                           d2L0dy2(y));

% --- Define Hessian Handle ---
Hessian = @(x, y) [H11(x, y), H12(x, y); 
                   H12(x, y), H22(x, y)];

%% Simulation: Statistics of Convergence with Varying Hyperparameters
learning_rate_list = logspace(-2, -1,  10);
noise_strength_list = logspace(-2, -1, 10);

% Initialize result matrices
final_position = zeros(length(learning_rate_list), length(noise_strength_list));
divergence_marker = zeros(length(learning_rate_list), length(noise_strength_list));
ness_distribution = zeros(length(learning_rate_list), length(noise_strength_list));
freezing_time_mean = zeros(length(learning_rate_list), length(noise_strength_list));
freezing_y_mean = zeros(length(learning_rate_list), length(noise_strength_list));

% --- Algorithm Parameters ---
initial_pos = [-0.01, yi]; 
iterations = 2e4;       % SGD Iterations
num_repetition = 2000;   % Number of repetitions

tic
% Create linear indices for parfor efficiency
lr_len = length(learning_rate_list);
ns_len = length(noise_strength_list);
total_combinations = lr_len * ns_len;

% Use temporary storage for parallel results
temp_results = cell(total_combinations, 1);

parfor idx = 1:total_combinations
    % Convert linear index to 2D indices
    [i, j] = ind2sub([lr_len, ns_len], idx);
    
    learning_rate = learning_rate_list(i);
    noise_strength = noise_strength_list(j);
    
    m = 0;  % Counter for convergence to positive side (flat valley)
    m0 = 0; % Counter for divergence
    last_y_positions = zeros(num_repetition, 1); % Store last y positions for NESS calculation
    freezing_times = nan(num_repetition, 1);     % Store freezing time for each run
    freezing_y_positions = zeros(num_repetition, 1);
    
    for n = 1:num_repetition
        % Alternate initial x sign
        if mod(n,2) == 1   % Odd
            init_pos = initial_pos;
        else               % Even
            init_pos = abs(initial_pos);
        end
        
        % Run SGD Simulation
        [Traj, ~] = stochastic_gradient_descent(Loss, dLdx, dLdy, Hessian, init_pos, learning_rate, iterations, noise_strength);
        
        % Determine final position
        if Traj(end, 1) > 0
            m = m + 1;
        elseif isnan(Traj(end, 1))
            m0 = m0 + 1;
        end
        
        % Store last y
        last_y_positions(n) = Traj(end, 2);
        
        % Calculate freezing time: the iteration of the last jump between positive and negative x-axis
        x_vals = Traj(:,1);
        sign_changes = find(diff(sign(x_vals)) ~= 0); 
        if ~isempty(sign_changes)
            freezing_times(n) = sign_changes(end) + 1;  % Time of last switch + 1
        else
            freezing_times(n) = 1;  % If no switch, record as 1
        end
        freezing_y_positions(n) = Traj(freezing_times(n), 2);
    end
    
    representative_y = median(last_y_positions(~isnan(last_y_positions)));
    
    % Calculation of NESS probability
    if ~isempty(representative_y)
        ness_temp = NESS_distribution(f1, f2, g1, Loss, learning_rate, noise_strength, representative_y);
    else
        ness_temp = NaN; % All trajectories diverged
    end
    
    % Store results for this hyperparameter combination
    temp_results{idx} = struct(...
        'i', i, ...
        'j', j, ...
        'ness_val', ness_temp, ...
        'final_val', m / num_repetition, ...
        'div_val', m0 / num_repetition, ...
        'freezing_time', mean(freezing_times, 'omitnan'), ... 
        'freezing_y', mean(freezing_y_positions, 'omitnan'));
    
    % Optional: Display progress
    fprintf('Completed %d/%d: lr=%.4f, noise=%.4f\n', idx, total_combinations, learning_rate, noise_strength);
end

% Reconstruct the matrices from temporary results
for idx = 1:total_combinations
    result = temp_results{idx};
    final_position(result.i, result.j) = result.final_val;
    divergence_marker(result.i, result.j) = result.div_val;
    ness_distribution(result.i, result.j) = result.ness_val;
    freezing_time_mean(result.i, result.j) = result.freezing_time;
    freezing_y_mean(result.i, result.j) = result.freezing_y;
end

% Define filename and save
file_name = strcat('result_PhsDgm_Ld', num2str(L_d), '_yd', num2str(y_d), '_x1', num2str(x1), '_x2', num2str(x2), '.mat');
save(file_name, 'ness_distribution', 'divergence_marker', 'final_position', ...
    'freezing_time_mean', 'freezing_y_mean', ...
    'learning_rate_list', 'noise_strength_list', 'initial_pos', 'seed', "yi", ...
    'f0', 'x0', 'x1', 'x2', 'y_f', 'y_b', 'y_d', 'L_d', 'iterations', 'num_repetition');

fprintf('Saved to: %s\n', file_name);
toc

%% Plotting Results
% Plot 1: Convergence Probability (P_flat)
figure('unit','points','PaperUnits','points', 'position', [100 100 400 300])
imagesc(noise_strength_list, learning_rate_list, final_position);
xlabel('Noise strength $\sigma$', 'Interpreter', 'LaTex')
ylabel('Learning rate $\eta$', 'Interpreter', 'LaTex')
xlim([min(noise_strength_list) max(noise_strength_list)])
ylim([min(learning_rate_list) max(learning_rate_list)])
text(1.10, 1.05, '$P_\mathrm{flat}$', 'Interpreter', 'LaTex', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'Fontsize', 12)
colorbar
axis square
box on
set(gca, 'XScale', 'log', 'YScale', 'log', 'YDir', 'normal', 'Fontname', 'Times New Roman', 'Fontsize', 12);

% Plot 2: Scaled Freezing Time
freezing_time_rescaled = learning_rate_list' .* freezing_time_mean;
figure('unit','points','PaperUnits','points', 'position', [100 100 400 300])
imagesc(noise_strength_list, learning_rate_list, log(freezing_time_rescaled)/log(10));
xlabel('Noise strength $\sigma$', 'Interpreter', 'LaTex')
ylabel('Learning rate $\eta$', 'Interpreter', 'LaTex')
xlim([min(noise_strength_list) max(noise_strength_list)])
ylim([min(learning_rate_list) max(learning_rate_list)])
text(1.10, 1.05, '$\eta t_\mathrm{freeze}$', 'Interpreter', 'LaTex', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'Fontsize', 12)
colorbar
axis square
box on
set(gca, 'XScale', 'log', 'YScale', 'log', 'YDir', 'normal', 'Fontname', 'Times New Roman', 'Fontsize', 12);

% Plot 3: Freezing y Position
figure('unit','points','PaperUnits','points', 'position', [100 100 400 300])
imagesc(noise_strength_list, learning_rate_list, freezing_y_mean);
xlabel('Noise strength $\sigma$', 'Interpreter', 'LaTex')
ylabel('Learning rate $\eta$', 'Interpreter', 'LaTex')
xlim([min(noise_strength_list) max(noise_strength_list)])
ylim([min(learning_rate_list) max(learning_rate_list)])
text(1.10, 1.05, '$y_\mathrm{freeze}$', 'Interpreter', 'LaTex', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'Fontsize', 12)
colorbar
axis square
box on
set(gca, 'XScale', 'log', 'YScale', 'log', 'YDir', 'normal', 'Fontname', 'Times New Roman', 'Fontsize', 12);

%% Cleanup
try
    if ~isempty(gcp('nocreate'))
        delete(gcp('nocreate'))
        fprintf('Parallel pool closed.\n')
    end
catch 
    warning('Error closing parallel pool')
end

%% Helper Functions

function [trajectory, loss_history] = stochastic_gradient_descent(Loss, grad_x, grad_y, Hessian, initial_pos, learning_rate, iterations, noise_strength)
    % stochastic_gradient_descent_2D executes 2D stochastic gradient descent
    %
    % Inputs:
    %   Loss           - Handle for Loss function L(x, y)
    %   grad_x         - Handle for partial derivative w.r.t x
    %   grad_y         - Handle for partial derivative w.r.t y
    %   Hessian        - Handle for Hessian matrix, returns 2x2 matrix
    %   initial_pos    - Initial position [x0, y0]
    %   learning_rate  - Learning rate (step size)
    %   iterations     - Number of iterations
    %   noise_strength - Noise strength (scaling factor)
    %
    % Outputs:
    %   trajectory     - (iterations+1) x 2 matrix, position [x, y] at each step
    %   loss_history   - (iterations+1) vector, loss value at each step
    
    % Initialize trajectory matrix and loss history
    trajectory = zeros(iterations+1, 2);
    loss_history = zeros(iterations+1, 1);
    trajectory(1, :) = initial_pos;
    loss_history(1) = Loss(initial_pos(1), initial_pos(2));
    
    % Current position
    x = initial_pos(1);
    y = initial_pos(2);
    
    % Perform SGD
    for i = 1:iterations
        % Calculate gradients
        dx = grad_x(x, y);
        dy = grad_y(x, y);
        
        % Calculate Hessian
        H = Hessian(x, y);
        
        % Ensure Hessian is symmetric
        H = (H + H') / 2;
        
        % Calculate covariance matrix: Cov = 2 * noise_strength * H
        % To ensure covariance matrix is positive semi-definite (PSD), adjust eigenvalues
        [V, D] = eig(H);
        D_positive = max(D, 0); % Set negative eigenvalues to 0
        Cov = 2 * noise_strength * V * D_positive * V';
        
        % Ensure Cov is symmetric
        Cov = (Cov + Cov') / 2;
        
        % Check if Cov is PSD (add epsilon for numerical stability)
        epsilon = 1e-8;
        Cov = Cov + epsilon * eye(2);
        
        % Generate noise, mean 0, covariance Cov
        noise = mvnrnd([0, 0], Cov);
        noise_x = noise(1);
        noise_y = noise(2);
        
        % Update position
        x_new = x - learning_rate * (dx + noise_x);
        y_new = y - learning_rate * (dy + noise_y);
        
        % Record new position
        trajectory(i+1, :) = [x_new, y_new];
        loss_history(i+1) = Loss(x_new, y_new);
        
        % Update current position
        x = x_new;
        y = y_new;
    end
end

function P_flat = NESS_distribution(f1, f2, g1, Loss, learning_rate, noise_strength, y_val)
    % Calculate probability in Non-Equilibrium Steady State (NESS)
    Delta_s = learning_rate * noise_strength / 2;
    
    % Calculate barrier height difference
    % Local minimum on positive axis is at x = g1(y)/2
    x_min = g1(y_val) / 2;
    L_min = Loss(x_min, y_val);
    L_barrier = Loss(0, y_val);
    Delta_L = L_barrier - L_min;
    
    % Calculate noise coefficient difference (f1, f2 depend on y)
    Delta_f = f1(y_val) - f2(y_val);
    
    % Calculate ratio and probability
    Ratio = exp(Delta_L * Delta_f / (2 * Delta_s));
    P_flat = Ratio / (1 + Ratio);
end