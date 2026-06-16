% Generate simulation data for revised Fig4 (Panels B and C)
% Fixed eta=0.01, varying sigma
% Saves: final P_flat, y_freeze (paper definition), and Panel B trajectories
% =========================================================================

tic;

%% Model definition
x1 = 0.8; x2 = 0.65; x0 = 1; f0 = 1;
y_b = 30; y_f = 20; y_d = 100; L_d = 1;
yi = 1;
learning_rate = 0.01;

g1 = @(y) 2*x1.*y./(y + y_b);
g2 = @(y) 2*x2.*y./(y + y_b);
f1 = @(y) f0.*(x1.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
f2 = @(y) f0.*(x2.^2/x0.^2).*((y + y_f).^2./(y + y_b).^2);
L0 = @(y) L_d.*exp(-y/y_d) + x0^2/f0;

Loss = @(x, y) (x >= 0) .* (x.*(x - g1(y)) ./ f1(y) + L0(y)) + ...
               (x < 0)  .* (x.*(x + g2(y)) ./ f2(y) + L0(y));

dg1dy = @(y) (2*x1*y_b) ./ ((y + y_b).^2);
dg2dy = @(y) (2*x2*y_b) ./ ((y + y_b).^2);
df1dy = @(y) (2*f0*(x1^2/x0^2) .* (y + y_f) * (y_b - y_f)) ./ ((y + y_b).^3);
df2dy = @(y) (2*f0*(x2^2/x0^2) .* (y + y_f) * (y_b - y_f)) ./ ((y + y_b).^3);
dL0dy = @(y) -(L_d/y_d) .* exp(-y/y_d);
d2g1dy2 = @(y) -(4*x1*y_b) ./ ((y + y_b).^3);
d2g2dy2 = @(y) -(4*x2*y_b) ./ ((y + y_b).^3);
d2L0dy2 = @(y) (L_d/y_d^2) .* exp(-y/y_d);

dLdx = @(x, y) (x >= 0) .* ((2*x - g1(y)) ./ f1(y)) + ...
               (x < 0)  .* ((2*x + g2(y)) ./ f2(y));

dLdy = @(x, y) dL0dy(y) - ...
               (x >= 0) .* (x.*(x-g1(y)).*df1dy(y)./(f1(y).^2) + x.*dg1dy(y)./f1(y)) - ...
               (x < 0)  .* (x.*(x+g2(y)).*df2dy(y)./(f2(y).^2) - x.*dg2dy(y)./f2(y));

H11 = @(x, y) (x >= 0) .* (2 ./ f1(y)) + (x < 0) .* (2 ./ f2(y));
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
Hessian = @(x, y) [H11(x, y), H12(x, y); H12(x, y), H22(x, y)];

%% Simulation parameters
sigma_list = [0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2];
panel_b_sigmas = [0.01, 0.05, 0.1];
n_sigma = numel(sigma_list);

num_rep_B = 1000;   % Panel B: more reps for smoother curves
num_rep_C = 500;    % Panel C: fewer reps, just need final P_flat
iterations = 1e5;   % enough for y to reach ~11

% Output arrays
P_flat_all    = zeros(n_sigma, 1);
y_freeze_all  = zeros(n_sigma, 1);

% Panel B trajectory data (accumulated online to save memory)
pb_mean_y   = cell(3, 1);
pb_P_right  = cell(3, 1);
pb_y_freeze = zeros(3, 1);

%% Main simulation loop
for s = 1:n_sigma
    sigma = sigma_list(s);
    is_pb = any(abs(sigma - panel_b_sigmas) < 1e-10);

    if is_pb
        num_rep = num_rep_B;
        pb_idx  = find(abs(panel_b_sigmas - sigma) < 1e-10);
        y_acc   = zeros(iterations+1, 1);   % accumulate y trajectory
        r_acc   = zeros(iterations+1, 1);   % accumulate right-valley indicator
    else
        num_rep = num_rep_C;
    end

    right_count  = 0;
    yf_sum       = 0;

    for n = 1:num_rep
        if mod(n, 2) == 1
            init_pos = [-0.01, yi];
        else
            init_pos = [ 0.01, yi];
        end

        [Traj, ~] = sgd_2d(Loss, dLdx, dLdy, Hessian, ...
                            init_pos, learning_rate, iterations, sigma);

        xv = Traj(:, 1);

        % Final valley
        if xv(end) > 0, right_count = right_count + 1; end

        % y_freeze: y at last valley hop (paper definition)
        sc = find(diff(sign(xv)) ~= 0);
        if ~isempty(sc)
            tf = sc(end) + 1;
        else
            tf = 1;
        end
        yf_sum = yf_sum + Traj(tf, 2);

        % Accumulate Panel B trajectory data
        if is_pb
            y_acc = y_acc + Traj(:, 2);
            r_acc = r_acc + double(xv > 0);
        end
    end

    P_flat_all(s)   = right_count / num_rep;
    y_freeze_all(s) = yf_sum / num_rep;

    if is_pb
        pb_mean_y{pb_idx}   = y_acc / num_rep;
        pb_P_right{pb_idx}  = r_acc / num_rep;
        pb_y_freeze(pb_idx) = y_freeze_all(s);
    end

    fprintf('[%2d/%d] sigma=%.4f  Delta_S=%.2e  P_flat=%.3f  y_freeze=%.2f  (%.0fs elapsed)\n', ...
        s, n_sigma, sigma, learning_rate*sigma, P_flat_all(s), y_freeze_all(s), toc);
end

%% Save
save_path = fullfile(fileparts(mfilename('fullpath')), 'Fig4_simulation_data.mat');
save(save_path, ...
    'learning_rate', 'sigma_list', 'P_flat_all', 'y_freeze_all', ...
    'panel_b_sigmas', 'pb_mean_y', 'pb_P_right', 'pb_y_freeze', ...
    'yi', 'num_rep_B', 'num_rep_C', 'iterations');

fprintf('\nDone. Saved to %s  (total %.0fs)\n', save_path, toc);

%% SGD function (self-contained)
function [trajectory, loss_history] = sgd_2d(Loss, grad_x, grad_y, Hessian, ...
        initial_pos, learning_rate, iterations, noise_strength)

    trajectory   = zeros(iterations+1, 2);
    loss_history = zeros(iterations+1, 1);
    trajectory(1, :) = initial_pos;
    loss_history(1)  = Loss(initial_pos(1), initial_pos(2));

    x = initial_pos(1);
    y = initial_pos(2);

    for i = 1:iterations
        dx = grad_x(x, y);
        dy = grad_y(x, y);

        H = Hessian(x, y);
        H = (H + H') / 2;

        [V, D] = eig(H);
        D = max(D, 0);
        Cov = 2 * noise_strength * V * D * V';
        Cov = (Cov + Cov') / 2 + 1e-8 * eye(2);

        noise = mvnrnd([0, 0], Cov);

        x = x - learning_rate * (dx + noise(1));
        y = y - learning_rate * (dy + noise(2));

        trajectory(i+1, :) = [x, y];
        loss_history(i+1)  = Loss(x, y);
    end
end
