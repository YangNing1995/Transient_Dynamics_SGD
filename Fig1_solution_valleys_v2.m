% 从同一初始点出发所有SGD轨迹数据分析
% ---------------超参数组合-----------------
% Bs = [1000, 500, 200, 100, 50, 20, 10]
% Lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
% 每组超参数均跑20次重复（GD即Bs=1000除外）
bs_list = [1000, 500, 200, 100, 50, 20, 10];
lr_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
num_bs = length(bs_list);
num_lr = length(lr_list);
num_realizations = 20;
num_timepoints = 101;

%% 读取数据 
Data_dir = 'E:\SynologyDrive\SynologyDrive\Deep learning\Waddington_landscape\Data\Train_with_different_hyperparas\';

Iteration_all = zeros(length(bs_list), length(lr_list), num_timepoints);                       % [#bs, #lr, #iteration]
Train_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);    % [#bs, #lr, #realization, #iteration]
Test_loss_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Train_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);     % [#bs, #lr, #realization, #iteration]
Test_acc_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints);      % [#bs, #lr, #realization, #iteration]
Wrong_indices_all = cell(length(bs_list), length(lr_list), num_realizations, num_timepoints);  % {#bs, #lr, #realization, #iteration}
Weights_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 2500); % [#bs, #lr, #realization, #iteration, #weight]
Hessian_all = zeros(length(bs_list), length(lr_list), num_realizations, num_timepoints, 2500); % [#bs, #lr, #realization, #iteration, #weight]

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
        end
    end
end

%% 3D Loss landscape & weight dynamic visualization (different hyper-paras for one realization)
Realization_index = 5;
Weights_all_onerun = reshape(Weights_all(:, :, Realization_index, :, :), [num_bs*num_lr*num_timepoints, 2500]);
[coeff, score, latent] = pca(Weights_all_onerun); % score为pca坐标系下的投影
Weights_pca = reshape(score, [num_bs, num_lr, num_timepoints, 2500]);

% 2D + Loss
figure('unit','points','PaperUnits','points', 'position', [100 100 600 600])

% 定义颜色和标记
colors = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
markers = {'.', 'x','o', '+', '*', 's', 'd'}; 

% 循环绘制轨迹
for i = 3: num_bs - 1  % [200, 100, 50, 20]
    for j = 3: num_lr -1  %  0.005, 0.01, 0.02, 0.05]
        % 绘制轨迹
        plot3(squeeze(Weights_pca(i, j, :, 1)), squeeze(Weights_pca(i, j, :, 2)), squeeze(Train_loss_all(i, j, Realization_index, :)), ...
              'color', colors(j, :), 'marker', markers{i}, 'LineWidth', 0.1, 'MarkerSize', 5)
        hold on
    end
end
% 绘制初始位置（使用黑色五角星）
initial_position = squeeze(Weights_pca(1, 1, 1, :));  % 初始位置
plot3(initial_position(1), initial_position(2), squeeze(Train_loss_all(1, 1, Realization_index, 1)), ...
      'color', 'k', 'marker', 'pentagram', 'MarkerSize', 10);  % 黑色五角星标记

% 设置视角、网格、标签
view(-230, 21)
grid on
box on
% xlabel('$\theta_\mathrm{pc1}$', 'Interpreter', 'latex','Units','normalized')
% ylabel('$\theta_\mathrm{pc2}$', 'Interpreter', 'latex')
hx = xlabel('$\theta_\mathrm{pc1}$', 'Interpreter', 'latex','Units','normalized');
hy = ylabel('$\theta_\mathrm{pc2}$', 'Interpreter', 'latex','Units','normalized');
zlabel('$\mathcal{L}_\mathrm{train}$', 'Interpreter', 'latex')
zlim([min(Train_loss_all, [], 'all'), max(Train_loss_all, [], 'all')])

% 设置字体和轴
set(gca,'Fontname', 'Times New Roman', 'Zscale', 'log', 'Fontsize', 24);

% 获取当前位置
posx = get(hx, 'Position');
posy = get(hy, 'Position');

% 调整 label 靠近坐标轴
set(hx, 'Position', posx + [0 0 0]);   % y 方向移动靠近
set(hy, 'Position', posy + [-0.08 0.08 0]);   % x 方向移动靠近

%% legend of loss landscape
% Legend
colors = [187, 187, 187; 68, 119, 179; 102, 204, 238; 34, 136, 51; 204, 187, 68; 238, 102, 119; 170, 51, 119]/255;
markers = {'.', 'x','o', '+', '*', 's', 'd'}; 

figure('unit','points','PaperUnits','points', 'position', [200 200 150 400]);
for j = 3 : num_lr -1
    % 展示颜色
    subplot(5, 2, 2*j-5);
    line([0.2 0.8], [1 1], 'color', colors(j, :), 'LineWidth', 2, 'MarkerSize', 5);
    xlim([0 1])
    axis off;
    title(['$\eta = {' num2str(lr_list(j)) '}$'], 'Interpreter', 'latex', 'Fontname', 'Times New Roman', 'Fontsize', 14);

    % 展示标记符号
    subplot(5, 2, 2*j-4);
    plot(0, 0, 'marker', markers{j}, 'color', 'k', 'MarkerSize', 12);  
    axis off;
    title(['$B = {' num2str(bs_list(j)) '}$'], 'Interpreter', 'latex', 'Fontname', 'Times New Roman', 'Fontsize', 14);
end
subplot(5, 2, 10);
plot(0, 0, 'marker', 'pentagram', 'color', 'k', 'MarkerSize', 12);  
axis off;
title('Initial point', 'Fontname', 'Times New Roman', 'Fontsize', 14);


%% Jaccard similarity
load("mycolormap.mat")
Realization_index = 5;
bs_list = [1000, 500, 200, 100, 50, 20, 10];
lr_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];

% 选择需要的数据
Wrong_indices_final = Wrong_indices_all(3:6, 3:6, Realization_index, end); % 选择需要的部分
Wrong_indices_final_reshape = reshape(Wrong_indices_final, [4*4, 1]);  % reshape为25个条件

% 计算Jaccard相似度
Jaccard_similarities_all = zeros(length(Wrong_indices_final_reshape), length(Wrong_indices_final_reshape));

for i = 1:length(Wrong_indices_final_reshape)
    for j = 1:length(Wrong_indices_final_reshape)
        if i ~= j  % 只计算不同的解对
            Jaccard_similarities_all(i, j) = jaccard_similarity(Wrong_indices_final_reshape{i}, Wrong_indices_final_reshape{j});
        else
            Jaccard_similarities_all(i, j) = 1;  % 同一个解的相似度为1
        end
    end
end

% 绘制图像
figure('unit','points','PaperUnits','points', 'position', [100 100 600 600])
imagesc(Jaccard_similarities_all)
set(gca, 'YDir', 'normal');
axis square
colormap(viridis)
% colormap(mycolormap.blue_white_red_improved)
colorbar
% title('Jaccard similarities')

% 设置横纵轴的刻度和标签
xticks(1:16)  % 横轴刻度为25
yticks(1:16)  % 纵轴刻度为25

% 生成横轴和纵轴的标签，表示对应的 (bs, lr) 组合
% 计算所有的 (bs, lr) 组合
[bs_grid, lr_grid] = meshgrid(bs_list(3:6), lr_list(3:6));  % 生成组合网格

% 直接拼接标签字符串
labels = strings(16, 1);  % 创建字符串数组
for i = 1:16
    labels(i) = ['$\eta =' num2str(lr_grid(i)) ',  B =' num2str(bs_grid(i)) '$'];  % 拼接字符串
end
% 设置 LaTeX 解释器并直接应用标签
set(gca, 'TickLabelInterpreter', 'latex');
xticklabels(labels);
yticklabels(labels);

% 设置字体和字体大小
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 14);
% text(1.05, 0.5, 'Jaccard Similarity', 'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 14);

%% Define Jaccard similarity
function similarity = jaccard_similarity(list1, list2)
    % 转换为集合
    set1 = unique(list1);
    set2 = unique(list2);

    % 计算交集和并集的大小
    intersection_size = numel(intersect(set1, set2));
    union_size = numel(union(set1, set2));

    % 计算Jaccard相似度
    similarity = intersection_size / union_size;
end