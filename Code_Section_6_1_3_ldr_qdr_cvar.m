clear;
clc;
close all;

%% Experiment description
% Experiment for Section 6.1.3
% Compares three methods for the contextual CVaR newsvendor problem:
% 1) RKHS policy
% 2) Linear decision rule (LDR)
% 3) Quadratic decision rule (QDR)
%
% The evaluation metrics are:
% - relative average distance to the oracle conditional-optimal CVaR policy
% - out-of-sample performance
% - running time

%% Settings
h = 0.2;
b = 1;
beta = 0.9;

a1 = 5;
a2 = 20;

mu_x1 = 1;
sd_x1 = 0.5;
mu_x2 = 0;
sd_x2 = 2;

% Context dimension in this experiment
d_context = 2;

n_train_list = [25, 50, 100, 200, 400, 800];
num_train_sizes = length(n_train_list);

num_replications = 20;

% Fixed context grid for policy-distance evaluation
n_x_context = 100;

% Fixed nested Monte Carlo test set
n_test_context = 1000;
n_test_y = 400;

rng(2026);

%% Reference hyperparameters for RKHS
h_ref_rkhs = 5;
lambda_ref_z_rkhs = 1e-6;
lambda_ref_t_rkhs = 0.8;

%% Solver settings
ops = sdpsettings('solver', 'mosek', 'verbose', 0);

%% Truncated distributions
pd_x1 = makedist('Normal', 'mu', mu_x1, 'sigma', sd_x1);
truncated_pd_x1 = truncate(pd_x1, mu_x1 - 2 * sd_x1, mu_x1 + 2 * sd_x1);

pd_x2 = makedist('Normal', 'mu', mu_x2, 'sigma', sd_x2);
truncated_pd_x2 = truncate(pd_x2, mu_x2 - 2 * sd_x2, mu_x2 + 2 * sd_x2);

pd_standard = makedist('Normal', 'mu', 0, 'sigma', 1);
truncated_pd_standard = truncate(pd_standard, -2, 2);

fprintf('Comparing RKHS, LDR, and QDR for the contextual CVaR newsvendor problem.\n');
fprintf('Training sample sizes: ');
fprintf('%d ', n_train_list);
fprintf('\n');
fprintf('Number of replications: %d\n', num_replications);
fprintf('Fixed test set: %d contexts and %d conditional samples per context.\n', ...
    n_test_context, n_test_y);
fprintf('Sample-size-dependent schedule for RKHS:\n');
fprintf('  lambda_N^z = lambda_ref^z * (10/N)^(1/3)\n');
fprintf('  lambda_N^t = lambda_ref^t * (10/N)^(1/3)\n');
fprintf('  h_N        = h_ref * (10/N)^(1/(d+4)) with d = %d\n\n', d_context);

%% Oracle conditional-optimal CVaR policy on a fixed context grid
x_generate = linspace(0.01, 0.99, n_x_context - 1);

cdf_x1_lower = normcdf(mu_x1 - 2 * sd_x1, mu_x1, sd_x1);
cdf_x1_upper = normcdf(mu_x1 + 2 * sd_x1, mu_x1, sd_x1);
x1_context = cdf_x1_lower + x_generate .* (cdf_x1_upper - cdf_x1_lower);
x1_context = norminv(x1_context, mu_x1, sd_x1);

cdf_x2_lower = normcdf(mu_x2 - 2 * sd_x2, mu_x2, sd_x2);
cdf_x2_upper = normcdf(mu_x2 + 2 * sd_x2, mu_x2, sd_x2);
x2_context = cdf_x2_lower + x_generate .* (cdf_x2_upper - cdf_x2_lower);
x2_context = norminv(x2_context, mu_x2, sd_x2);

[X1_grid, X2_grid] = meshgrid(x1_context, x2_context);
x_context_reshape = [X1_grid(:), X2_grid(:)];

p_left = b * (1 - beta) / (h + b);
p_right = (h * beta + b) / (h + b);

q_left_true = truncatedStandardQuantile(p_left);
q_right_true = truncatedStandardQuantile(p_right);

z_opt_cvar = trueContextOptCVaR2D(X1_grid, X2_grid, a1, a2, h, b, q_left_true, q_right_true);
norm_z_opt_cvar = norm(z_opt_cvar, 'fro');

fprintf('Oracle policy on the context grid has been constructed.\n');

%% Fixed nested Monte Carlo test set
x1_test = random(truncated_pd_x1, n_test_context, 1);
x2_test = random(truncated_pd_x2, n_test_context, 1);

mu_test = a1 * exp(x1_test) + a2 * cos(x2_test) + 100;
sd_test = (4 * x1_test + x2_test + 20) / 2;

epsilon_test = random(truncated_pd_standard, n_test_y, n_test_context);
y_test = repmat(mu_test', n_test_y, 1) + repmat(sd_test', n_test_y, 1) .* epsilon_test;
y_test = max(y_test, 0);

z_opt_test = trueContextOptCVaR2D(x1_test, x2_test, a1, a2, h, b, q_left_true, q_right_true);
oracle_oos = averageConditionalCVAR(z_opt_test, y_test, h, b, beta);

fprintf('Fixed out-of-sample test set has been generated.\n');
fprintf('Oracle out-of-sample performance (average conditional CVaR): %.6f\n\n', oracle_oos);

%% Storage
distance_rkhs = nan(num_train_sizes, num_replications);
distance_ldr = nan(num_train_sizes, num_replications);
distance_qdr = nan(num_train_sizes, num_replications);

oos_rkhs = nan(num_train_sizes, num_replications);
oos_ldr = nan(num_train_sizes, num_replications);
oos_qdr = nan(num_train_sizes, num_replications);

time_rkhs = nan(num_train_sizes, num_replications);
time_ldr = nan(num_train_sizes, num_replications);
time_qdr = nan(num_train_sizes, num_replications);

%% Storage for policy surfaces at a selected training sample size
plot_n_value = 800;
plot_n_idx = find(n_train_list == plot_n_value, 1);

if isempty(plot_n_idx)
    error('The selected plot_n_value is not included in n_train_list.');
end

num_x1_grid = size(X1_grid, 1);
num_x2_grid = size(X1_grid, 2);

z_grid_rkhs_store = nan(num_x1_grid, num_x2_grid, num_replications);
z_grid_ldr_store = nan(num_x1_grid, num_x2_grid, num_replications);
z_grid_qdr_store = nan(num_x1_grid, num_x2_grid, num_replications);

%% Main loop
for i = 1:num_train_sizes
    n_train = n_train_list(i);

    h_rkhs = h_ref_rkhs * (10 / n_train)^(1 / (d_context + 4));
    lambda_z_rkhs = lambda_ref_z_rkhs * (10 / n_train)^(1 / 3);
    lambda_t_rkhs = lambda_ref_t_rkhs * (10 / n_train)^(1 / 3);

    fprintf('============================================================\n');
    fprintf('Training sample size n = %d\n', n_train);
    fprintf('============================================================\n');
    fprintf('RKHS hyperparameters:\n');
    fprintf('  h_N^RKHS       = %.6f\n', h_rkhs);
    fprintf('  lambda_N^z     = %.6e\n', lambda_z_rkhs);
    fprintf('  lambda_N^t     = %.6e\n\n', lambda_t_rkhs);

    for j = 1:num_replications
        %% Generate training data
        x1_train = random(truncated_pd_x1, n_train, 1);
        x2_train = random(truncated_pd_x2, n_train, 1);

        sd_train = (4 * x1_train + x2_train + 20) / 2;
        epsilon_train = random(truncated_pd_standard, n_train, 1);
        y_train = max(a1 * exp(x1_train) + a2 * cos(x2_train) + sd_train .* epsilon_train + 100, 0);

        x_train = [x1_train, x2_train];

        %% ------------------------------------------------------------
        % Method 1: RKHS policy
        % ------------------------------------------------------------
        tic;
        yalmip('clear');

        K_train = exp(-pdist2(x_train, x_train, 'euclidean').^2 / (2 * h_rkhs^2));

        alpha_z = sdpvar(n_train, 1);
        alpha_t = sdpvar(n_train, 1);
        diff_yz = sdpvar(n_train, 1);
        diff_zy = sdpvar(n_train, 1);
        l = sdpvar(n_train, 1);

        z = K_train * alpha_z;
        t_x = K_train * alpha_t;

        Constraints = [diff_yz >= 0, diff_zy >= 0, l >= 0];
        Constraints = [Constraints, diff_zy >= z - y_train, diff_yz >= y_train - z];
        Constraints = [Constraints, l >= b * diff_yz + h * diff_zy - t_x];

        Objective = sum(t_x) / n_train ...
            + sum(l) / (n_train * (1 - beta)) ...
            + lambda_z_rkhs * (alpha_z' * alpha_z) ...
            + lambda_t_rkhs * (alpha_t' * alpha_t);

        diagnostics = optimize(Constraints, Objective, ops);

        if diagnostics.problem == 0
            alpha_z_rkhs = value(alpha_z);
            train_time_rkhs = toc;

            % Decision time on test contexts only
            tic;
            K_test = exp(-pdist2([x1_test, x2_test], x_train, 'euclidean').^2 / (2 * h_rkhs^2));
            z_test_rkhs = K_test * alpha_z_rkhs;
            decision_time_rkhs = toc;

            % Policy on the context grid for distance evaluation
            K_grid = exp(-pdist2(x_context_reshape, x_train, 'euclidean').^2 / (2 * h_rkhs^2));
            z_grid_rkhs = reshape(K_grid * alpha_z_rkhs, size(X1_grid));

            distance_rkhs(i, j) = norm(z_grid_rkhs - z_opt_cvar, 'fro') / norm_z_opt_cvar;
            oos_rkhs(i, j) = averageConditionalCVAR(z_test_rkhs, y_test, h, b, beta);
            time_rkhs(i, j) = train_time_rkhs + decision_time_rkhs;
            
            if i == plot_n_idx
                z_grid_rkhs_store(:, :, j) = z_grid_rkhs;
            end
        else
            warning('RKHS failed at n = %d, replication = %d.', n_train, j);
        end

        %% ------------------------------------------------------------
        % Method 2: Linear decision rule (LDR)
        % ------------------------------------------------------------
        tic;
        yalmip('clear');

        X_linear = [x1_train, x2_train, ones(n_train, 1)];
        g = sdpvar(3, 1);
        t = sdpvar(3, 1);
        diff_yz = sdpvar(n_train, 1);
        diff_zy = sdpvar(n_train, 1);
        l = sdpvar(n_train, 1);

        z = X_linear * g;
        t_x = X_linear * t;

        Constraints = [diff_yz >= 0, diff_zy >= 0, l >= 0];
        Constraints = [Constraints, diff_zy >= z - y_train, diff_yz >= y_train - z];
        Constraints = [Constraints, l >= b * diff_yz + h * diff_zy - t_x];

        Objective = sum(t_x) / n_train + sum(l) / (n_train * (1 - beta));
        diagnostics = optimize(Constraints, Objective, ops);

        if diagnostics.problem == 0
            g_ldr = value(g);
            train_time_ldr = toc;

            tic;
            X_test_linear = [x1_test, x2_test, ones(n_test_context, 1)];
            z_test_ldr = X_test_linear * g_ldr;
            decision_time_ldr = toc;

            X_grid_linear = [x_context_reshape, ones(size(x_context_reshape, 1), 1)];
            z_grid_ldr = reshape(X_grid_linear * g_ldr, size(X1_grid));

            distance_ldr(i, j) = norm(z_grid_ldr - z_opt_cvar, 'fro') / norm_z_opt_cvar;
            oos_ldr(i, j) = averageConditionalCVAR(z_test_ldr, y_test, h, b, beta);
            time_ldr(i, j) = train_time_ldr + decision_time_ldr;
            
            if i == plot_n_idx
                z_grid_ldr_store(:, :, j) = z_grid_ldr;
            end
        else
            warning('LDR failed at n = %d, replication = %d.', n_train, j);
        end

        %% ------------------------------------------------------------
        % Method 3: Quadratic decision rule (QDR)
        % ------------------------------------------------------------
        tic;
        yalmip('clear');

        X_quad = [ones(n_train,1), x1_train, x2_train, x1_train.^2, x2_train.^2, x1_train .* x2_train];
        g = sdpvar(6, 1);
        t = sdpvar(6, 1);
        diff_yz = sdpvar(n_train, 1);
        diff_zy = sdpvar(n_train, 1);
        l = sdpvar(n_train, 1);

        z = X_quad * g;
        t_x = X_quad * t;

        Constraints = [diff_yz >= 0, diff_zy >= 0, l >= 0];
        Constraints = [Constraints, diff_zy >= z - y_train, diff_yz >= y_train - z];
        Constraints = [Constraints, l >= b * diff_yz + h * diff_zy - t_x];

        Objective = sum(t_x) / n_train + sum(l) / (n_train * (1 - beta));
        diagnostics = optimize(Constraints, Objective, ops);

        if diagnostics.problem == 0
            g_qdr = value(g);
            train_time_qdr = toc;

            tic;
            X_test_quad = [ones(n_test_context,1), x1_test, x2_test, x1_test.^2, x2_test.^2, x1_test .* x2_test];
            z_test_qdr = X_test_quad * g_qdr;
            decision_time_qdr = toc;

            X_grid_quad = [ones(size(x_context_reshape, 1), 1), ...
                x_context_reshape(:,1), x_context_reshape(:,2), ...
                x_context_reshape(:,1).^2, x_context_reshape(:,2).^2, ...
                x_context_reshape(:,1) .* x_context_reshape(:,2)];
            z_grid_qdr = reshape(X_grid_quad * g_qdr, size(X1_grid));

            distance_qdr(i, j) = norm(z_grid_qdr - z_opt_cvar, 'fro') / norm_z_opt_cvar;
            oos_qdr(i, j) = averageConditionalCVAR(z_test_qdr, y_test, h, b, beta);
            time_qdr(i, j) = train_time_qdr + decision_time_qdr;
            
            if i == plot_n_idx
                z_grid_qdr_store(:, :, j) = z_grid_qdr;
            end
        else
            warning('QDR failed at n = %d, replication = %d.', n_train, j);
        end

        fprintf('Completed n = %d, replication = %d\n', n_train, j);
    end
end

%% Print how the metrics are tested
fprintf('\n============================================================\n');
fprintf('How the out-of-sample performance is tested\n');
fprintf('============================================================\n');
fprintf(['A fixed nested Monte Carlo test set is used for all methods and all replications:\n' ...
         '- %d test contexts are sampled first.\n' ...
         '- For each context, %d conditional problem data samples are generated.\n' ...
         '- For each method, the policy decision is evaluated at the %d test contexts.\n' ...
         '- The out-of-sample performance is defined as the average empirical conditional CVaR over the %d test contexts.\n'], ...
         n_test_context, n_test_y, n_test_context, n_test_context);

fprintf('\n============================================================\n');
fprintf('How the running time is measured\n');
fprintf('============================================================\n');
fprintf(['For RKHS, running time = training time + decision time on the fixed test contexts.\n' ...
         'For LDR and QDR, running time = optimization time for fitting the decision rule + decision time on the fixed test contexts.\n' ...
         'The cost of computing the oracle benchmark and the out-of-sample metric itself is not included.\n']);

%% Display mean tables
sample_sizes_column = n_train_list(:);

mean_distance_table = table(sample_sizes_column, ...
    mean(distance_rkhs, 2, 'omitnan'), ...
    mean(distance_ldr, 2, 'omitnan'), ...
    mean(distance_qdr, 2, 'omitnan'), ...
    'VariableNames', {'TrainingSize', 'RKHS', 'LDR', 'QDR'});

mean_oos_table = table(sample_sizes_column, ...
    mean(oos_rkhs, 2, 'omitnan'), ...
    mean(oos_ldr, 2, 'omitnan'), ...
    mean(oos_qdr, 2, 'omitnan'), ...
    'VariableNames', {'TrainingSize', 'RKHS', 'LDR', 'QDR'});

mean_time_table = table(sample_sizes_column, ...
    mean(time_rkhs, 2, 'omitnan'), ...
    mean(time_ldr, 2, 'omitnan'), ...
    mean(time_qdr, 2, 'omitnan'), ...
    'VariableNames', {'TrainingSize', 'RKHS', 'LDR', 'QDR'});

fprintf('\n============================================================\n');
fprintf('Mean relative average distance\n');
fprintf('============================================================\n');
disp(mean_distance_table);

fprintf('\n============================================================\n');
fprintf('Mean out-of-sample performance (average conditional CVaR)\n');
fprintf('============================================================\n');
disp(mean_oos_table);

fprintf('\n============================================================\n');
fprintf('Mean running time (seconds)\n');
fprintf('============================================================\n');
disp(mean_time_table);

fprintf('\nOracle out-of-sample performance: %.6f\n', oracle_oos);

%% Plot settings
sample_labels = arrayfun(@(n) sprintf('n=%d', n), n_train_list, 'UniformOutput', false);

method_labels = {'RKHS', 'LDR', 'QDR'};
method_colors = [
    0.85 0.33 0.10;   % orange
    0.00 0.45 0.74;   % blue
    0.47 0.67 0.19    % green
];

fig_width_cm = 24;
fig_height_cm = 7.8;

%% Grouped colored boxplots
fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, fig_width_cm, fig_height_cm]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

distance_ylim = computeUnifiedYLim([distance_rkhs(:); distance_ldr(:); distance_qdr(:)], true,true);
oos_ylim = computeUnifiedYLim([oos_rkhs(:); oos_ldr(:); oos_qdr(:)], true,true);
time_ylim = computeUnifiedYLim([time_rkhs(:); time_ldr(:); time_qdr(:)], true,false);

nexttile;
plotCompactGroupedBoxplot( ...
    {distance_rkhs, distance_ldr, distance_qdr}, ...
    sample_labels, method_labels, method_colors, ...
    'Training sample size', ...
    'Relative average distance', ...
    distance_ylim, true, 'northwest');

nexttile;
plotCompactGroupedBoxplot( ...
    {oos_rkhs, oos_ldr, oos_qdr}, ...
    sample_labels, method_labels, method_colors, ...
    'Training sample size', ...
    'Out-of-sample performance', ...
    oos_ylim, true, 'northwest');

nexttile;
plotCompactGroupedBoxplot( ...
    {time_rkhs, time_ldr, time_qdr}, ...
    sample_labels, method_labels, method_colors, ...
    'Training sample size', ...
    'Time (seconds)', ...
    time_ylim, true, 'northwest');

drawnow;
exportgraphics(fig, 'Section_613_rkhs_ldr_qdr_boxplot.pdf', ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
fprintf('\nSaved grouped boxplot figure to Section_613_rkhs_ldr_qdr_boxplot.pdf\n');



%% Plot 3D policy surfaces for the selected training sample size

% rep_to_plot = 1;
% z_grid_rkhs_mean = z_grid_rkhs_store(:, :, rep_to_plot);
% z_grid_ldr_mean = z_grid_ldr_store(:, :, rep_to_plot);
% z_grid_qdr_mean = z_grid_qdr_store(:, :, rep_to_plot);

z_grid_rkhs_mean = mean(z_grid_rkhs_store, 3, 'omitnan');
z_grid_ldr_mean = mean(z_grid_ldr_store, 3, 'omitnan');
z_grid_qdr_mean = mean(z_grid_qdr_store, 3, 'omitnan');

all_surface_values = [z_opt_cvar(:); z_grid_ldr_mean(:); z_grid_qdr_mean(:); z_grid_rkhs_mean(:)];
z_surface_limits = [min(all_surface_values), max(all_surface_values)];

fig_surface_width_cm = 27;
fig_surface_height_cm = 6;

fig_surface = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, fig_surface_width_cm, fig_surface_height_cm]);

tiledlayout(1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) Oracle optimal policy
nexttile;
surf(X1_grid, X2_grid, z_opt_cvar, 'EdgeColor', 'none');
view(245, 20);
% zlim(z_surface_limits);
zlim([80 180])
xlabel('$X_1$', 'Interpreter', 'latex');
ylabel('$X_2$', 'Interpreter', 'latex');
zlabel('$z$', 'Interpreter', 'latex');
title('Conditional-CVaR policy', 'FontSize', 11);
colormap(parula);
box on;
grid on;
set(gca, 'FontSize', 10);

% (b) LDR
nexttile;
surf(X1_grid, X2_grid, z_grid_ldr_mean, 'EdgeColor', 'none');
view(245, 20);
% zlim(z_surface_limits);
zlim([80 180])
xlabel('$X_1$', 'Interpreter', 'latex');
ylabel('$X_2$', 'Interpreter', 'latex');
zlabel('$z$', 'Interpreter', 'latex');
title('Expected-CVaR policy with LDR', 'FontSize', 11);
colormap(parula);
box on;
grid on;
set(gca, 'FontSize', 10);

% (c) QDR
nexttile;
surf(X1_grid, X2_grid, z_grid_qdr_mean, 'EdgeColor', 'none');
view(245, 20);
% zlim(z_surface_limits);
zlim([80 180])
xlabel('$X_1$', 'Interpreter', 'latex');
ylabel('$X_2$', 'Interpreter', 'latex');
zlabel('$z$', 'Interpreter', 'latex');
title('Expected-CVaR policy with QDR', 'FontSize', 11);
colormap(parula);
box on;
grid on;
set(gca, 'FontSize', 10);

% (d) RKHS
nexttile;
surf(X1_grid, X2_grid, z_grid_rkhs_mean, 'EdgeColor', 'none');
view(245, 20);
% zlim(z_surface_limits);
zlim([80 180])
xlabel('$X_1$', 'Interpreter', 'latex');
ylabel('$X_2$', 'Interpreter', 'latex');
zlabel('$z$', 'Interpreter', 'latex');
title('Expected-CVaR policy within RKHS', 'FontSize', 11);
colormap(parula);
box on;
grid on;
set(gca, 'FontSize', 10);

drawnow;

% surface_file_name = sprintf('policy_surfaces_N%d.pdf', plot_n_value);
exportgraphics(fig_surface, 'Section_613_rkhs_ldr_qdr_surface.pdf', ...
    'ContentType', 'vector', 'BackgroundColor', 'white');

fprintf('\nSaved 3D policy surface figure to Section_613_rkhs_ldr_qdr_surface.pdf');

%% Local functions
function q = truncatedStandardQuantile(p)
    cdf_lower = normcdf(-2, 0, 1);
    cdf_upper = normcdf(2, 0, 1);
    p_adjusted = cdf_lower + p * (cdf_upper - cdf_lower);
    q = norminv(p_adjusted, 0, 1);
end

function z_opt = trueContextOptCVaR2D(x1, x2, a1, a2, h, b, q_left, q_right)
    mu = a1 * exp(x1) + a2 * cos(x2) + 100;
    sigma = (4 * x1 + x2 + 20) / 2;
    z_opt = h / (h + b) * (mu + sigma .* q_left) ...
          + b / (h + b) * (mu + sigma .* q_right);
end

function avg_cvar = averageConditionalCVAR(z_policy, y_matrix, h, b, beta)
    z_row = reshape(z_policy, 1, []);
    loss = h * max(repmat(z_row, size(y_matrix, 1), 1) - y_matrix, 0) ...
         + b * max(y_matrix - repmat(z_row, size(y_matrix, 1), 1), 0);

    cvar_each_context = empiricalCVARColumns(loss, beta);
    avg_cvar = mean(cvar_each_context);
end

function cvar_values = empiricalCVARColumns(loss_matrix, beta)
    sorted_loss = sort(loss_matrix, 1, 'ascend');
    n = size(sorted_loss, 1);
    k = floor(beta * n);

    if k >= n
        cvar_values = sorted_loss(end, :);
        return;
    end

    weight = k + 1 - beta * n;
    cvar_values = (weight * sorted_loss(k + 1, :) + sum(sorted_loss(k + 2:end, :), 1)) ...
        / (n * (1 - beta));
end

function y_limits = computeUnifiedYLim(values, force_nonnegative, add_height)
    values = values(~isnan(values));
    if isempty(values)
        y_limits = [0, 1];
        return;
    end

    vmin = min(values);
    vmax = max(values);

    if abs(vmax - vmin) < 1e-12
        padding = max(1e-3, 0.05 * max(abs(vmax), 1));
    else
        padding = 0.05 * (vmax - vmin);
    end
    
    lower_bound = vmin - padding;
    if add_height
        upper_bound = vmax + padding+0.4 * (vmax - vmin);
    else
        upper_bound = vmax + padding;
    end

    if force_nonnegative
        lower_bound = max(0, lower_bound);
    end

    y_limits = [lower_bound, upper_bound];
end

function plotCompactGroupedBoxplot(data_cell, sample_labels, method_labels, method_colors, x_label_text, y_label_text, y_limits, show_legend, legend_location)
    num_methods = numel(data_cell);
    num_groups = size(data_cell{1}, 1);

    offsets = linspace(-0.28, 0.28, num_methods);
    box_width = 0.18;

    max_replications = 0;
    for m = 1:num_methods
        max_replications = max(max_replications, size(data_cell{m}, 2));
    end

    num_boxes = num_groups * num_methods;
    data_matrix = nan(max_replications, num_boxes);
    positions = zeros(1, num_boxes);

    col = 0;
    for g = 1:num_groups
        for m = 1:num_methods
            col = col + 1;
            vals = data_cell{m}(g, :);
            vals = vals(:);
            data_matrix(1:length(vals), col) = vals;
            positions(col) = g + offsets(m);
        end
    end

    hold on;
    boxplot(data_matrix, ...
        'Positions', positions, ...
        'Widths', box_width, ...
        'Symbol', 'k.', ...
        'Colors', 'k', ...
        'Whisker', 1.5, ...
        'Labels', repmat({''}, 1, num_boxes));

    set(gca, 'XTick', 1:num_groups, 'XTickLabel', sample_labels, ...
        'FontSize', 11, 'LineWidth', 1.0, 'Box', 'on', 'Layer', 'top');
    xlim([0.5, num_groups + 0.5]);
    ylim(y_limits);

    ax = gca;
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.GridAlpha = 0.18;
    ax.GridColor = [0.7, 0.7, 0.7];

    h_box = findobj(gca, 'Tag', 'Box');
    h_box = flipud(h_box);

    for k = 1:length(h_box)
        method_idx = mod(k - 1, num_methods) + 1;
        patch(get(h_box(k), 'XData'), get(h_box(k), 'YData'), method_colors(method_idx, :), ...
            'FaceAlpha', 0.75, ...
            'EdgeColor', method_colors(method_idx, :), ...
            'LineWidth', 1.0);
    end

    h_median = findobj(gca, 'Tag', 'Median');
    for k = 1:length(h_median)
        set(h_median(k), 'Color', [0.2, 0.2, 0.2], 'LineWidth', 1.3);
    end

    h_whisker = findobj(gca, 'Tag', 'Whisker');
    for k = 1:length(h_whisker)
        set(h_whisker(k), 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.0);
    end

    h_cap = findobj(gca, 'Tag', 'Cap');
    for k = 1:length(h_cap)
        set(h_cap(k), 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.0);
    end

    h_outlier = findobj(gca, 'Tag', 'Outliers');
    for k = 1:length(h_outlier)
        set(h_outlier(k), 'Marker', 'o', 'MarkerSize', 3, ...
            'MarkerEdgeColor', [0.2, 0.2, 0.2]);
    end

    xlabel(x_label_text, 'FontSize', 12);
    ylabel(y_label_text, 'FontSize', 12);

    if show_legend
        legend_handles = gobjects(num_methods, 1);
        for m = 1:num_methods
            legend_handles(m) = patch(nan, nan, method_colors(m, :), ...
                'FaceAlpha', 0.75, ...
                'EdgeColor', method_colors(m, :));
        end
        legend(legend_handles, method_labels, 'Location', legend_location, 'Box', 'off', 'FontSize', 10);
    end

    hold off;
end
