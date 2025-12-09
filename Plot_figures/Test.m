% Visualization of P_flat^tr,SGD surface vs Delta_S and Gamma
% Based on the freezing mechanism derivation
% =============================================================

clear; close all; clc;

% 1. 定义模型参数 (Model Parameters)
x2 = 0.65;          % Sharp valley width parameter (base)
x0 = 1;
f0 = 1;
y_b = 30;
y_f = 20;
y_d = 100;
L_d = 1;
epsilon = 1e-2;     % Small constant epsilon (permissible deviation)

% 2. 定义自变量网格 (Grid for Independent Variables)
% Delta_S: 噪声强度与学习率的乘积 (Log scale)
dS_vals = logspace(-6, -1, 20); 

% Gamma: 平坦度比率 f1/f2 = x1^2/x2^2 (Linear scale, >1)
gamma_vals = linspace(1.0, 1.2, 20); 

[Delta_S, Gamma] = meshgrid(dS_vals, gamma_vals);

% 3. 计算中间变量 Phi (Calculating Phi)
% 注意: Phi 依赖于 x1^2 - x2^2。
% 根据定义 gamma = x1^2 / x2^2 => x1^2 = x2^2 * gamma
% 因此 x1^2 - x2^2 = x2^2 * (gamma - 1)

diff_x2 = x2^2 .* (Gamma - 1); % (x1^2 - x2^2)

% 根据文中公式计算 Phi (近似常数项)
% Phi approx = [2(x1^2 - x2^2) / 27 y_b] * [L_d/y_d + 8 x0^2 / 27 y_f]
term1 = (2 .* diff_x2) ./ (27 * y_b);
term2 = (L_d / y_d) + (8 * x0^2) / (27 * y_f);
Phi = term1 .* term2;

% 4. 计算概率 P (Calculating Probability)
% Eq: P = [ 1 + gamma^(-0.5) * ( sqrt(Delta_S) / (epsilon * Phi) )^(1-gamma) ]^(-1)

% 分解计算以防出错
Numerator_Inside_Log = sqrt(Delta_S);
Denominator_Inside_Log = epsilon .* Phi;
Base_Term = Numerator_Inside_Log ./ Denominator_Inside_Log;

Exponent = 1 - Gamma;
Prefactor = Gamma.^(-0.5);

% 组合公式
P_flat_tr = (1 + Prefactor .* (Base_Term .^ Exponent)).^(-1);

% 5. 绘图 (Plotting)
figure('Units', 'points', 'Position', [100, 100, 500, 400]);

% 绘制曲面
s = surf(Gamma, Delta_S, P_flat_tr);

% 美化设置
s.EdgeColor = 'none';       % 去除网格线
s.FaceColor = 'interp';     % 平滑着色
colormap(parula);           % 颜色映射

% 坐标轴设置
set(gca, 'YScale', 'log');  % Delta_S 使用对数坐标
ylim([min(dS_vals), max(dS_vals)]);
xlim([min(gamma_vals), max(gamma_vals)]);
zlim([0.45, 1]);

% 标签
ylabel('$\Delta_S$', 'Interpreter', 'latex', 'FontSize', 16);
xlabel('$\gamma = f_1/f_2$', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('$P_{\mathrm{flat}}^{\mathrm{tr,SGD}}$', 'Interpreter', 'latex', 'FontSize', 16);

% 视角调整
view(-45, 30);
grid on;
box on;
colorbar;

% 设置字体
set(gca, 'FontName', 'Times New Roman', 'FontSize', 18);
