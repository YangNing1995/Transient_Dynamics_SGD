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

%% 读取总Iteration数据 
Data_dir = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Train_with_different_hyperparas\';
% Data_dir = '../../../Data/Train_with_different_hyperparas/';

Iteration_all = zeros(length(bs_list), length(lr_list), num_timepoints);                       % [#bs, #lr, #iteration]
Train_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);    % [#bs, #lr, #realization, #iteration]
Test_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Train_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Test_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);      % [#bs, #lr, #realization, #iteration]
Wrong_indices_all = cell(length(bs_list), length(lr_list), num_realizations, num_timepoints);  % {#bs, #lr, #realization, #iteration}
Weights_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 2500); % [#bs, #lr, #realization, #iteration, #weight]
Hessian_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 2500); % [#bs, #lr, #realization, #iteration, #weight]

% 存freeze time
idx_freeze_all = zeros(length(bs_list), length(lr_list), num_realizations);   % [#bs, #lr, #irealization]
for i = 1:length(bs_list)
    bs = bs_list(i);
    for j = 1:length(lr_list)
        lr = lr_list(j);
        max_iteration = 100/lr;
        iterations_list = linspace(0, max_iteration, num_timepoints);
        for k = 1:num_realizations
            if bs == 1000 && k>1
                load(strcat(Data_dir, 'bs',num2str(bs),'_lr', num2str(lr), '/save_metrics_repeat1.mat'));
                load(strcat(Data_dir, 'bs',num2str(bs),'_lr', num2str(lr), '/save_hessian_repeat1.mat'));            
            else
                load(strcat(Data_dir, 'bs',num2str(bs),'_lr', num2str(lr), '/save_metrics_repeat', num2str(k),'.mat'));
                load(strcat(Data_dir, 'bs',num2str(bs),'_lr', num2str(lr), '/save_hessian_repeat', num2str(k),'.mat'));
            end
            Positions = find(ismember(save_iterations, iterations_list));
            Iteration_all(i, j, :) = save_iterations(1, Positions);
            Train_loss_all(i, j, k, :) = train_loss(1, Positions);
            Test_loss_all(i, j, k, :) = test_loss(1, Positions);
            Train_acc_all(i, j, k, :) = train_accuracy(1, Positions);
            Test_acc_all(i, j, k, :) = test_accuracy(1, Positions);
            Wrong_indices_all(i, j, k, :) = wrong_indices(1, Positions);
            Weights_all(i, j, k, :, :) = weight_all(Positions, :);
            Hessian_all(i, j, k, :, :) = Hessian';
               
            idx_freeze = find(train_loss <= 0.1, 1, 'first'); % freeze time定义为train loss首次低于0.1的时刻
            if ~isempty(idx_freeze)
                idx_freeze_all(i, j, k) = save_iterations(idx_freeze); 
            else
                idx_freeze_all(i, j, k) = nan;
            end
      
        end
    end
end

Flatness_all = prod(Hessian_all(:, :, :, :, 1:10), 5).^(-1/10);
Convergence_probability = sum(Train_acc_all(:,:,:,end)==1, 3)/num_realizations;

%% train loss & train acc
bs = 5;
lr = 6;
num = 2;
max_iteration = 100/(lr_list(lr));
iterations_list = linspace(0, max_iteration, num_timepoints);

% squeeze 去掉多余维度，保证是向量
train_loss = squeeze(Train_loss_all(bs, lr, num, :));
train_acc  = squeeze(Train_acc_all(bs, lr, num, :));

% 期刊风格配色 
loss_color = [204 51 136]/255;   % 洋红
acc_color  = [0 153 136]/255;    % 青绿


figure('unit','points','PaperUnits','points', 'position', [100 100 500 400])

% --- 左轴 (Loss) ---
yyaxis left
semilogy(iterations_list, train_loss, '-', ...
    'LineWidth',1.8, 'Color',loss_color)
ylabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex', 'Color',loss_color)
ylim([0 max(train_loss)*1.1])
ax = gca;
ax.YColor = loss_color;   % 左轴颜色与曲线保持一致

% 水平线 loss=0.1
yline(0.1,'--','Color','k','LineWidth',1.2)

% 找到交点（第一个 loss <=0.1 的位置）
idx = find(train_loss <= 0.1, 1);
if ~isempty(idx)
    t_cross = iterations_list(idx);
    hold on
    plot(t_cross, train_loss(idx),'o','MarkerSize',7, ...
         'MarkerEdgeColor',loss_color,'LineWidth',2)
    xline(t_cross, '--k', 'LineWidth',1.2)   % 竖直虚线保持黑色，突出参考
end

% --- 右轴 (Accuracy) ---
yyaxis right
plot(iterations_list, train_acc, '-', ...
    'LineWidth',1.8, 'Color',acc_color)
ylabel('$Acc_\mathrm{train}$', 'Interpreter', 'latex', 'Color',acc_color)
ax.YColor = acc_color;   % 右轴颜色与曲线保持一致

% --- 统一格式 ---
xlabel('Iteration $t$', 'Interpreter', 'latex')
xlim([0 1500])
set(gca,'Fontname','Times New Roman','Fontsize',22)
grid off
box on
%% freeze time
idx_freeze_mean = mean(idx_freeze_all, 3, "omitmissing");
idx_freeze_rescaled = idx_freeze_mean.*lr_list;

figure('unit','points','PaperUnits','points', 'position', [100 100 400 400])
imagesc(idx_freeze_rescaled,[12 60]) % [10 100]
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)

cb = colorbar;
% set(gca, 'ColorScale', 'log')   % 设置对数色标
% 
% %---- 设置 colorbar 刻度和标签 ----
% ticks = 10.^(1:2);   % 根据你的数据范围调整
% cb.Ticks = ticks;
% cb.TickLabels = arrayfun(@(x) sprintf('$10^{%d}$', round(log10(x))), ticks, 'UniformOutput', false);
% set(cb, 'TickLabelInterpreter','latex')   % 用 LaTeX 显示 10^x

xlabel('Learning rate $\eta$', 'Interpreter','latex')
xticklabels(string(lr_list));
ylabel('Batch size $B$', 'Interpreter','latex')
yticklabels(string(bs_list));
set(gca,'Fontname', 'Times New Roman',  'Fontsize', 18);

% % -------- 在 NaN 的位置加上 '**' --------
% [m, n] = size(idx_freeze_rescaled);
% for i = 1:m
%     for j = 1:n
%         if isnan(idx_freeze_rescaled(i,j))
%             text(j, i, '*', 'HorizontalAlignment','center', ...
%                  'VerticalAlignment','middle', 'Color','w', ...
%                  'FontSize',16, 'FontWeight','bold', 'FontName','Arial');
%         end
%     end
% end

% -------- 叠加 Convergence_probability 信息 --------
[m, n] = size(Convergence_probability);

for i = 1:m
    for j = 1:n
        % 如果想让部分区域变灰或有阴影
        if Convergence_probability(i,j) == 0
            % 绘制灰色方块（半透明）
            patch([j-0.5 j+0.5 j+0.5 j-0.5], ...
                  [i-0.5 i-0.5 i+0.5 i+0.5], ...
                  [0.5 0.5 0.5], ...        % 灰色RGB
                  'EdgeColor', 'none');     % 无边框
        end
    end
end
hold off;

%% Mean Train loss 
lr = 5; 
max_iteration = 100/(lr_list(lr));
iterations_list = linspace(0, max_iteration, num_timepoints);
Train_loss_mean = squeeze(mean(Train_loss_all(:, lr, :, :), 3));

% 新建画布
figure('unit','points','PaperUnits','points', 'position', [100 100 500 400]) 
colors = get(gca, 'ColorOrder');
num_lines = size(Train_loss_mean, 1);
h = zeros(1, num_lines);

for i = 1:num_lines
    y_mean = Train_loss_mean(i, :);
    legend_name = strcat('$B = ', num2str(bs_list(i)), '$');
    h(i) = semilogy(iterations_list, y_mean,...
                   'LineWidth',1.5,...
                   'Color',colors(i,:),...
                   'DisplayName', legend_name); 
    hold on
end
% 水平线 loss=0.1
yline(0.1,'--','Color','k','LineWidth',1.2)


xlim([0 3000])
ylim([0  4.6147]) 
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$\langle\mathcal{L}_\mathrm{train}\rangle$', 'Interpreter', 'latex');
yticks([1e-2 1e-1 1])
yticklabels({'10^{-2}','10^{-1}','10^{0}'})
box on
grid on

legend(h, 'Location','northeast',...
         'FontName','Times New Roman',...
         'FontSize', 16, 'Interpreter', 'latex');  
legend box on

set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 22, 'Yscale', 'log');

%% Mean Flatness
lr = 5; 
max_iteration = 100/(lr_list(lr));
iterations_list = linspace(0, max_iteration, num_timepoints);
Flatness_mean   = squeeze(mean(Flatness_all(:, lr, :, :), 3));

% 新建画布
figure('unit','points','PaperUnits','points', 'position', [100 100 500 400]) 
colors = get(gca, 'ColorOrder');
num_lines = size(Train_loss_mean, 1);
h = zeros(1, num_lines);

for i = 1:num_lines
    y_mean = Flatness_mean(i, :);
    legend_name = strcat('$B = ', num2str(bs_list(i)), '$');
    h(i) = plot(iterations_list, y_mean,...
                   'LineWidth',1.5,...
                   'Color',colors(i,:),...
                   'DisplayName', legend_name); 
    hold on
end
% 水平线 loss=0.1

xlim([0 3000])
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel('$\langle F\rangle$', 'Interpreter', 'latex');
box on
grid on

legend(h, 'Location','northwest',...
         'FontName','Times New Roman',...
         'FontSize', 16, 'Interpreter', 'latex');  
legend box on

set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 22);



%% 合并画图
lr = 5; 
max_iteration = 100/(lr_list(lr));
iterations_list = linspace(0, max_iteration, num_timepoints);

% 计算均值
Train_loss_mean = squeeze(mean(Train_loss_all(:, lr, :, :), 3));
Flatness_mean   = squeeze(mean(Flatness_all(:, lr, :, :), 3));

% 新建画布
figure('unit','points','PaperUnits','points', 'position', [100 100 500 400]) 
% t = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');  % 紧凑布局
t = tiledlayout(2,1, ...
               'TileSpacing','none', ...  % 子图间间距：'none' 最小
               'Padding','none');         % figure 边缘空白

% --------- (A) Train Loss ----------
nexttile
hold on;
colors = get(gca, 'ColorOrder');
num_lines = size(Train_loss_mean, 1);
h = zeros(1, num_lines);

for i = 1:num_lines
    y_mean = Train_loss_mean(i, :);
    legend_name = strcat('$B = ', num2str(bs_list(i)), '$');
    h(i) = semilogy(iterations_list, y_mean,...
                   'LineWidth',1.5,...
                   'Color',colors(i,:),...
                   'DisplayName', legend_name); 
end

hold off;
xlim([0 max_iteration])
ylabel_handle1 = ylabel('$\langle\mathcal{L}_\mathrm{train}\rangle$', 'Interpreter', 'latex');
yticks([1e-2 1e-1 1])
yticklabels({'10^{-2}','10^{-1}','10^{0}'})
box on
grid on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 22, 'Yscale', 'log');
set(gca,'XTickLabel',[]);   % 不显示xticklabel


% 图例只放在上图
legend(h, 'Location','northeast',...
         'FontName','Times New Roman',...
         'FontSize', 16, 'Interpreter', 'latex');  
legend box off

% --------- (B) Flatness ----------
nexttile
hold on;
colors = get(gca, 'ColorOrder');
num_lines = size(Flatness_mean, 1);

for i = 1:num_lines
    y_mean = Flatness_mean(i, :);
    plot(iterations_list, y_mean,...
             'LineWidth',1.5,...
             'Color',colors(i,:)); 
end

hold off;
xlim([0 max_iteration])
xlabel('Iteration $t$', 'Interpreter', 'latex')
ylabel_handle2 = ylabel('$\langle F\rangle$', 'Interpreter', 'latex');
grid on
box on
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 24);

% 获取位置
pos1 = get(ylabel_handle1,'Position');
pos2 = get(ylabel_handle2,'Position');
pos2_new = [pos1(1), pos2(2:3)];
set(ylabel_handle2,'Position',pos2_new);
