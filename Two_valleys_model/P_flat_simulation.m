% Runs multiple stochastic gradient descent trajectories and records statistics
% to compare simulation results with theoretical predictions
% =========================================================

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
yi = 1;     

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

%% Simulation 
tic;

% === Parameter setup ===
learning_rate = 0.01;
noise_strength = 0.1;
num_repetition = 1000; 
iterations = 1e5;
initial_pos = [-0.01, yi];

% === Initialize record matrices ===
right_mask_all = zeros(iterations + 1, num_repetition);
y_all = zeros(iterations + 1, num_repetition);

% === Run repeated stochastic trajectories ===
for n = 1:num_repetition
    % Alternate initial positions for diversity
    if mod(n, 2) == 1
        init_pos = initial_pos;
    else
        init_pos = abs(initial_pos);
    end

    % Run stochastic gradient descent trajectory
    [Traj, ~] = stochastic_gradient_descent( ...
        Loss, dLdx, dLdy, Hessian, init_pos, ...
        learning_rate, iterations, noise_strength);

    % Record x and y coordinates
    x_vals = Traj(:, 1);
    y_vals = Traj(:, 2);

    right_mask_all(:, n) = double(x_vals > 0);
    y_all(:, n) = y_vals;
end

% === Compute empirical probabilities ===
P_right_simulation = mean(right_mask_all, 2);  % right-valley probability vs time
mean_y  = mean(y_all, 2);           % average y vs time

% === Compute theoretical steady-state probabilities ===
Delta_s = learning_rate * noise_strength;

ness_escape_rates_vs_time_left  = NESS_escape_rate('left',  mean_y, f1, f2, g1, g2, Loss, Delta_s);
ness_escape_rates_vs_time_right = NESS_escape_rate('right', mean_y, f1, f2, g1, g2, Loss, Delta_s);

P_right_theory = ness_escape_rates_vs_time_left ./ ...
                  (ness_escape_rates_vs_time_right + ness_escape_rates_vs_time_left);

toc;

% === Plot comparison between simulation and theory ===
figure('Units', 'points', 'PaperUnits', 'points', 'Position', [100 100 400 300]);
plot(mean_y, P_right_simulation, 'b-', 'LineWidth', 2, 'DisplayName', 'Simulation');
hold on;
plot(mean_y, P_right_theory, 'r--', 'LineWidth', 2, 'DisplayName', 'Theory');
xlabel('$y$', 'Interpreter', 'LaTeX');
ylabel('$P_\mathrm{right}(y)$', 'Interpreter', 'LaTeX');
title(sprintf('$\\eta = %.3f,\\; \\sigma = %.3f$', learning_rate, noise_strength), 'Interpreter', 'LaTeX');
legend('Location', 'best', 'Box', 'off');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'Box', 'on', 'XScale', 'log');
grid on;

% === Save variables for later analysis ===
save_filename = sprintf('P_right_eta%.3f_sigma%.3f.mat', learning_rate, noise_strength);

save(save_filename, ...
    'learning_rate', 'noise_strength', ...
    'iterations', 'num_repetition', ...
    'mean_y', 'P_right_simulation');

fprintf('\nResults saved to: %s\n', save_filename);

%% SGD
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



