clear;
clc;
close all;

%% Experiment description
% Experiment for Section 6.1.4
%
% The script compares:
% 1) RKHS policy for expected CVaR
% 2) RKHS policy for ex ante CVaR
% 3) SLO with Gaussian-kernel conditional distribution estimation 
%
% For each block dimension d, the script generates one figure with three subplots:
% - relative average distance
% - Out-of-sample performance
% - running time

%% Settings
h = 0.2;
b = 1;
beta = 0.9;

a1 = 5;
a2 = 20;

mu_x = 1;
sd_x = 0.5;

dimension_list = [2,4,6,8,10,12];
num_dimensions = length(dimension_list);

n_train_list = [100, 200, 400, 800];
num_train_sizes = length(n_train_list);

num_replications = 5;

% Fixed evaluation and test sizes for each dimension
n_eval_context = 1000;
n_test_context = 1000;
n_test_y = 400;

rng(2026);

%% Fixed reference sample size
n_ref = 10;

%% Regularization scaling exponent
alpha_reg = 1 / 3;

%% Dimension-specific reference hyperparameters
sigma_ref_expected_list = [10, 10, 7, 7, 7, 4];
sigma_ref_exante_list   = [10, 10, 7, 4, 3, 2];
sigma_ref_slo_list      = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];

lambda_z_expected_ref_list = [5e-7, 5e-7, 5e-7, 5e-7, 1e-7, 1e-7];
lambda_t_expected_ref_list = [0.05, 0.01, 0.1, 0.5, 0.5, 0.5];
lambda_z_exante_ref_list   = [5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 1e-7];

%% Length checks
if length(sigma_ref_expected_list) ~= num_dimensions
    error('sigma_ref_expected_list must have the same length as dimension_list.');
end

if length(sigma_ref_exante_list) ~= num_dimensions
    error('sigma_ref_exante_list must have the same length as dimension_list.');
end

if length(sigma_ref_slo_list) ~= num_dimensions
    error('sigma_ref_slo_list must have the same length as dimension_list.');
end

if length(lambda_z_expected_ref_list) ~= num_dimensions
    error('lambda_z_expected_ref_list must have the same length as dimension_list.');
end

if length(lambda_t_expected_ref_list) ~= num_dimensions
    error('lambda_t_expected_ref_list must have the same length as dimension_list.');
end

if length(lambda_z_exante_ref_list) ~= num_dimensions
    error('lambda_z_exante_ref_list must have the same length as dimension_list.');
end

%% Solver settings
ops = sdpsettings('solver', 'mosek', 'verbose', 0);

%% Truncated distributions
pd_x = makedist('Normal', 'mu', mu_x, 'sigma', sd_x);
truncated_pd_x = truncate(pd_x, mu_x - 2 * sd_x, mu_x + 2 * sd_x);

pd_standard = makedist('Normal', 'mu', 0, 'sigma', 1);
truncated_pd_standard = truncate(pd_standard, -2, 2);

fprintf('Studying the effect of both dimension and training sample size.\n');
fprintf('Each figure corresponds to one fixed block dimension d and varies the training sample size.\n');
fprintf('The reference sample size n_ref is fixed at %d for both bandwidth and regularization scaling.\n\n', n_ref);

%% Closed-form benchmark constants for the true conditionally optimal policy
p_left = b * (1 - beta) / (h + b);
p_right = (h * beta + b) / (h + b);

q_left_true = truncatedStandardQuantile(p_left);
q_right_true = truncatedStandardQuantile(p_right);

%% Storage for results
distance_expected = nan(num_dimensions, num_train_sizes, num_replications);
distance_exante = nan(num_dimensions, num_train_sizes, num_replications);
distance_slo = nan(num_dimensions, num_train_sizes, num_replications);

oop_expected = nan(num_dimensions, num_train_sizes, num_replications);
oop_exante = nan(num_dimensions, num_train_sizes, num_replications);
oop_slo = nan(num_dimensions, num_train_sizes, num_replications);

time_expected_total = nan(num_dimensions, num_train_sizes, num_replications);
time_exante_total = nan(num_dimensions, num_train_sizes, num_replications);
time_slo_total = nan(num_dimensions, num_train_sizes, num_replications);

%% Main loop over dimensions
for dim_idx = 1:num_dimensions
    d = dimension_list(dim_idx);
    d_x = 2 * d;
    B = alternatingWeightVector(d);

    sigma_ref_expected_current = sigma_ref_expected_list(dim_idx);
    sigma_ref_exante_current = sigma_ref_exante_list(dim_idx);
    sigma_ref_slo_current = sigma_ref_slo_list(dim_idx);

    lambda_z_expected_ref_current = lambda_z_expected_ref_list(dim_idx);
    lambda_t_expected_ref_current = lambda_t_expected_ref_list(dim_idx);
    lambda_z_exante_ref_current = lambda_z_exante_ref_list(dim_idx);

    fprintf('============================================================\n');
    fprintf('Processing block dimension d = %d (full context dimension = %d)\n', d, d_x);
    fprintf('============================================================\n');
    fprintf('Reference sigma values for this dimension:\n');
    fprintf('  sigma_ref_expected = %.6f\n', sigma_ref_expected_current);
    fprintf('  sigma_ref_exante   = %.6f\n', sigma_ref_exante_current);
    fprintf('  sigma_ref_slo      = %.6f\n', sigma_ref_slo_current);
    fprintf('Reference lambda values for this dimension:\n');
    fprintf('  lambda_z_expected_ref = %.6e\n', lambda_z_expected_ref_current);
    fprintf('  lambda_t_expected_ref = %.6e\n', lambda_t_expected_ref_current);
    fprintf('  lambda_z_exante_ref   = %.6e\n\n', lambda_z_exante_ref_current);

    %% Fixed evaluation set for this dimension
    X1_eval = random(truncated_pd_x, n_eval_context, d);
    X2_eval = random(truncated_pd_x, n_eval_context, d);
    x_eval = [X1_eval, X2_eval];

    [phi1_eval, phi2_eval] = latentFeatures(X1_eval, X2_eval, B);
    z_opt_eval = trueContextOptCVaRLatent(phi1_eval, phi2_eval, a1, a2, h, b, q_left_true, q_right_true);
    norm_z_opt_eval = norm(z_opt_eval);

    %% Fixed large test set for this dimension
    X1_test = random(truncated_pd_x, n_test_context, d);
    X2_test = random(truncated_pd_x, n_test_context, d);
    x_test = [X1_test, X2_test];

    [phi1_test, phi2_test] = latentFeatures(X1_test, X2_test, B);
    [mu_test, sigma_test] = latentLocationScale(phi1_test, phi2_test, a1, a2);

    epsilon_test = random(truncated_pd_standard, n_test_y, n_test_context);
    y_test = repmat(mu_test', n_test_y, 1) + repmat(sigma_test', n_test_y, 1) .* epsilon_test;
    y_test = max(y_test, 0);

    z_opt_test = trueContextOptCVaRLatent(phi1_test, phi2_test, a1, a2, h, b, q_left_true, q_right_true);
    avg_cvar_opt_test = averageConditionalCVAR(z_opt_test, y_test, h, b, beta);

    fprintf('Fixed evaluation set size: %d contexts\n', n_eval_context);
    fprintf('Fixed test set size: %d contexts, %d conditional samples per context\n', n_test_context, n_test_y);
    fprintf('Average conditional CVaR of the oracle policy on the fixed test set: %.6f\n\n', avg_cvar_opt_test);

    %% Loop over training sample sizes
    for n_idx = 1:num_train_sizes
        n_train = n_train_list(n_idx);
        fprintf('  Training sample size n = %d\n', n_train);

        for rep = 1:num_replications
            %% Generate training data
            X1_train = random(truncated_pd_x, n_train, d);
            X2_train = random(truncated_pd_x, n_train, d);
            x_train = [X1_train, X2_train];

            [phi1_train, phi2_train] = latentFeatures(X1_train, X2_train, B);
            [mu_train, sigma_train] = latentLocationScale(phi1_train, phi2_train, a1, a2);

            epsilon_train = random(truncated_pd_standard, n_train, 1);
            y_train = mu_train + sigma_train .* epsilon_train;
            y_train = max(y_train, 0);

            %% Sample-dependent bandwidths and regularization parameters
            bw_expected = theoryBandwidthScalar(sigma_ref_expected_current, n_train, n_ref, d_x);
            bw_exante = theoryBandwidthScalar(sigma_ref_exante_current, n_train, n_ref, d_x);
            bw_slo = theoryBandwidthScalar(sigma_ref_slo_current, n_train, n_ref, d_x);

            lambda_z_expected = theoryLambda(lambda_z_expected_ref_current, n_train, n_ref, alpha_reg);
            lambda_t_expected = theoryLambda(lambda_t_expected_ref_current, n_train, n_ref, alpha_reg);
            lambda_z_exante = theoryLambda(lambda_z_exante_ref_current, n_train, n_ref, alpha_reg);

            if rep == 1
                fprintf('    Expected-CVaR bandwidth = %.6f, lambda_z = %.6e, lambda_t = %.6e\n', ...
                    bw_expected, lambda_z_expected, lambda_t_expected);

                fprintf('    Ex-ante-CVaR bandwidth = %.6f, lambda_z = %.6e\n', ...
                    bw_exante, lambda_z_exante);

                fprintf('    SLO bandwidth = %.6f\n', bw_slo);
            end

            %% Method 1: RKHS + Expected CVaR
            tic;

            K_train = gaussianKernelScaled(x_train, x_train, bw_expected);
            K_reg = (K_train + K_train') / 2 + 1e-8 * eye(n_train);

            yalmip('clear');

            alpha_z = sdpvar(n_train, 1);
            alpha_t = sdpvar(n_train, 1);
            diff_yz = sdpvar(n_train, 1);
            diff_zy = sdpvar(n_train, 1);
            l = sdpvar(n_train, 1);

            z_train = K_train * alpha_z;
            t_train = K_train * alpha_t;

            Constraints = [diff_yz >= 0, diff_zy >= 0, l >= 0];
            Constraints = [Constraints, z_train >= 0];
            Constraints = [Constraints, diff_zy >= z_train - y_train, diff_yz >= y_train - z_train];
            Constraints = [Constraints, l >= b * diff_yz + h * diff_zy - t_train];

            Objective = sum(t_train) / n_train ...
                + sum(l) / (n_train * (1 - beta)) ...
                + lambda_z_expected * (alpha_z' * K_reg * alpha_z) ...
                + lambda_t_expected * (alpha_t' * K_reg * alpha_t);

            diagnostics = optimize(Constraints, Objective, ops);

            alpha_z_expected = value(alpha_z);

            if diagnostics.problem ~= 0 || any(isnan(alpha_z_expected))
                warning('Expected CVaR RKHS model failed at d = %d, n = %d, replication = %d.', d, n_train, rep);
            else
                train_time_expected = toc;

                tic;
                K_eval = gaussianKernelScaled(x_eval, x_train, bw_expected);
                z_eval_expected = max(K_eval * alpha_z_expected, 0);

                K_test = gaussianKernelScaled(x_test, x_train, bw_expected);
                z_test_expected = max(K_test * alpha_z_expected, 0);
                test_time_expected = toc;

                distance_expected(dim_idx, n_idx, rep) = norm(z_eval_expected - z_opt_eval) / norm_z_opt_eval;

                avg_cvar_expected = averageConditionalCVAR(z_test_expected, y_test, h, b, beta);
                oop_expected(dim_idx, n_idx, rep) = (avg_cvar_expected - avg_cvar_opt_test) / avg_cvar_opt_test;

                time_expected_total(dim_idx, n_idx, rep) = train_time_expected + test_time_expected;
            end

            %% Method 2: RKHS + Ex ante CVaR
            tic;

            K_train = gaussianKernelScaled(x_train, x_train, bw_exante);
            K_reg = (K_train + K_train') / 2 + 1e-8 * eye(n_train);

            yalmip('clear');

            alpha_z = sdpvar(n_train, 1);
            t_scalar = sdpvar(1, 1);
            diff_yz = sdpvar(n_train, 1);
            diff_zy = sdpvar(n_train, 1);
            l = sdpvar(n_train, 1);

            z_train = K_train * alpha_z;

            Constraints = [diff_yz >= 0, diff_zy >= 0, l >= 0];
            Constraints = [Constraints, z_train >= 0];
            Constraints = [Constraints, diff_zy >= z_train - y_train, diff_yz >= y_train - z_train];
            Constraints = [Constraints, l >= b * diff_yz + h * diff_zy - t_scalar];

            Objective = t_scalar ...
                + sum(l) / (n_train * (1 - beta)) ...
                + lambda_z_exante * (alpha_z' * K_reg * alpha_z);

            diagnostics = optimize(Constraints, Objective, ops);

            alpha_z_exante = value(alpha_z);

            if diagnostics.problem ~= 0 || any(isnan(alpha_z_exante))
                warning('Ex ante CVaR RKHS model failed at d = %d, n = %d, replication = %d.', d, n_train, rep);
            else
                train_time_exante = toc;

                tic;
                K_eval = gaussianKernelScaled(x_eval, x_train, bw_exante);
                z_eval_exante = max(K_eval * alpha_z_exante, 0);

                K_test = gaussianKernelScaled(x_test, x_train, bw_exante);
                z_test_exante = max(K_test * alpha_z_exante, 0);
                test_time_exante = toc;

                distance_exante(dim_idx, n_idx, rep) = norm(z_eval_exante - z_opt_eval) / norm_z_opt_eval;

                avg_cvar_exante = averageConditionalCVAR(z_test_exante, y_test, h, b, beta);
                oop_exante(dim_idx, n_idx, rep) = (avg_cvar_exante - avg_cvar_opt_test) / avg_cvar_opt_test;

                time_exante_total(dim_idx, n_idx, rep) = train_time_exante + test_time_exante;
            end

            %% Method 3: SLO + Gaussian kernel conditional distribution estimation
            try
                z_eval_slo = sloPolicyCVaRByOptimization(x_eval, x_train, y_train, bw_slo, h, b, beta);
                distance_slo(dim_idx, n_idx, rep) = norm(z_eval_slo - z_opt_eval) / norm_z_opt_eval;

                tic;
                z_test_slo = sloPolicyCVaRByOptimization(x_test, x_train, y_train, bw_slo, h, b, beta);
                time_slo_total(dim_idx, n_idx, rep) = toc;

                avg_cvar_slo = averageConditionalCVAR(z_test_slo, y_test, h, b, beta);
                oop_slo(dim_idx, n_idx, rep) = (avg_cvar_slo - avg_cvar_opt_test) / avg_cvar_opt_test;
            catch
                warning('SLO method failed at d = %d, n = %d, replication = %d.', d, n_train, rep);
            end

            fprintf('    Completed: d = %d, n = %d, replication = %d\n', d, n_train, rep);
        end

        fprintf('  Finished all replications for d = %d, n = %d.\n\n', d, n_train);
    end
end

%% Build mean tables
distance_mean_table = buildMetricMeanTable(dimension_list, n_train_list, ...
    distance_expected, distance_exante, distance_slo, ...
    {'Mean_RKHS_Expected_CVaR', 'Mean_RKHS_ExAnte_CVaR', 'Mean_SLO'});

oop_mean_table = buildMetricMeanTable(dimension_list, n_train_list, ...
    oop_expected, oop_exante, oop_slo, ...
    {'Mean_RKHS_Expected_CVaR', 'Mean_RKHS_ExAnte_CVaR', 'Mean_SLO'});

runtime_mean_table = buildMetricMeanTable(dimension_list, n_train_list, ...
    time_expected_total, time_exante_total, time_slo_total, ...
    {'Mean_RKHS_Expected_CVaR', 'Mean_RKHS_ExAnte_CVaR', 'Mean_SLO'});

fprintf('\n============================================================\n');
fprintf('Mean relative average distance by dimension and training sample size\n');
fprintf('============================================================\n');
disp(distance_mean_table);

fprintf('\n============================================================\n');
fprintf('Mean relative out-of-performance by dimension and training sample size\n');
fprintf('============================================================\n');
disp(oop_mean_table);

fprintf('\n============================================================\n');
fprintf('Mean running time (seconds) by dimension and training sample size\n');
fprintf('============================================================\n');
disp(runtime_mean_table);

fprintf('\n============================================================\n');
fprintf('Compact summary by dimension and sample size\n');
fprintf('============================================================\n');
for dim_idx = 1:num_dimensions
    d = dimension_list(dim_idx);
    for n_idx = 1:num_train_sizes
        n_train = n_train_list(n_idx);
        fprintf(['d = %-3d | n = %-4d | Distance: [Expected %.4f, ExAnte %.4f, SLO %.4f] ', ...
                 '| OOP: [Expected %.4f, ExAnte %.4f, SLO %.4f] ', ...
                 '| Time(s): [Expected %.4f, ExAnte %.4f, SLO %.4f]\n'], ...
            d, n_train, ...
            mean(distance_expected(dim_idx, n_idx, :), 3, 'omitnan'), ...
            mean(distance_exante(dim_idx, n_idx, :), 3, 'omitnan'), ...
            mean(distance_slo(dim_idx, n_idx, :), 3, 'omitnan'), ...
            mean(oop_expected(dim_idx, n_idx, :), 3, 'omitnan'), ...
            mean(oop_exante(dim_idx, n_idx, :), 3, 'omitnan'), ...
            mean(oop_slo(dim_idx, n_idx, :), 3, 'omitnan'), ...
            mean(time_expected_total(dim_idx, n_idx, :), 3, 'omitnan'), ...
            mean(time_exante_total(dim_idx, n_idx, :), 3, 'omitnan'), ...
            mean(time_slo_total(dim_idx, n_idx, :), 3, 'omitnan'));
    end
end

%% Method labels and colors
sample_labels = arrayfun(@(n) sprintf('n=%d', n), n_train_list, 'UniformOutput', false);

method_labels = {'Expected-CVaR policy (RKHS)', 'Ex-ante-CVaR policy (RKHS)', 'SLO (kernel regression)'};
method_colors = [
    0.85 0.37 0.01;
    0.00 0.60 0.65;
    0.47 0.67 0.19
];

%% Figure export settings
fig_width_cm = 27;
fig_height_cm = 7.8;

%% Plot one figure per dimension: boxplots only
for dim_idx = 1:num_dimensions
    d = dimension_list(dim_idx);

    distance_values_dim = [ ...
        squeeze(distance_expected(dim_idx, :, :)), ...
        squeeze(distance_exante(dim_idx, :, :)), ...
        squeeze(distance_slo(dim_idx, :, :)) ];
    distance_ylim_dim = computeUnifiedYLim(distance_values_dim(:), true, true);

    oop_values_dim = [ ...
        squeeze(oop_expected(dim_idx, :, :)), ...
        squeeze(oop_exante(dim_idx, :, :)), ...
        squeeze(oop_slo(dim_idx, :, :)) ];
    oop_ylim_dim = computeUnifiedYLim(oop_values_dim(:), false, true);

    time_values_dim = [ ...
        squeeze(time_expected_total(dim_idx, :, :)), ...
        squeeze(time_exante_total(dim_idx, :, :)), ...
        squeeze(time_slo_total(dim_idx, :, :)) ];
    time_ylim_dim = computeUnifiedYLim(time_values_dim(:), true, true);

    fig = figure('Color', 'w', 'Units', 'centimeters', ...
        'Position', [2, 2, fig_width_cm, fig_height_cm]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plotCompactGroupedBoxplot( ...
        {squeeze(distance_expected(dim_idx, :, :)), ...
         squeeze(distance_exante(dim_idx, :, :)), ...
         squeeze(distance_slo(dim_idx, :, :))}, ...
        sample_labels, method_labels, method_colors, ...
        'Training sample size', ...
        'Relative average distance', ...
        distance_ylim_dim, true, 'northwest');

    nexttile;
    plotCompactGroupedBoxplot( ...
        {squeeze(oop_expected(dim_idx, :, :)), ...
         squeeze(oop_exante(dim_idx, :, :)), ...
         squeeze(oop_slo(dim_idx, :, :))}, ...
        sample_labels, method_labels, method_colors, ...
        'Training sample size', ...
        'Relative out-of-sample performance', ...
        oop_ylim_dim, true, 'northwest');

    nexttile;
    plotCompactGroupedBoxplot( ...
        {squeeze(time_expected_total(dim_idx, :, :)), ...
         squeeze(time_exante_total(dim_idx, :, :)), ...
         squeeze(time_slo_total(dim_idx, :, :))}, ...
        sample_labels, method_labels, method_colors, ...
        'Training sample size', ...
        'Time (seconds)', ...
        time_ylim_dim, true, 'northwest');

    drawnow;

    save_name = sprintf('Section_614_boxplots_d%d.pdf', d);
    exportgraphics(fig, save_name, 'ContentType', 'vector', 'BackgroundColor', 'white');
    fprintf('Saved figure to %s\n', save_name);
end

%% Local functions
function B = alternatingWeightVector(d)
    B = zeros(d, 1);
    for k = 1:d
        level = ceil(k / 2);
        sign_value = (-1)^(k + 1);
        B(k) = sign_value * level;
    end
end

function [phi1, phi2] = latentFeatures(X1, X2, B)
    phi1 = X1 * B;
    phi2 = X2 * B;
end

function [mu, sigma] = latentLocationScale(phi1, phi2, a1, a2)
    mu = a1 * exp(2 + 0.5 * phi1) + a2 * cos(2 * phi2) + 100;
    sigma = abs(phi1 + phi2 + 12);
end

function z_opt = trueContextOptCVaRLatent(phi1, phi2, a1, a2, h, b, q_left, q_right)
    [mu, sigma] = latentLocationScale(phi1, phi2, a1, a2);
    z_opt = h / (h + b) * (mu + sigma .* q_left) ...
          + b / (h + b) * (mu + sigma .* q_right);
end

function q = truncatedStandardQuantile(p)
    cdf_lower = normcdf(-2, 0, 1);
    cdf_upper = normcdf(2, 0, 1);
    p_adjusted = cdf_lower + p * (cdf_upper - cdf_lower);
    q = norminv(p_adjusted, 0, 1);
end

function K = gaussianKernelScaled(XA, XB, bw_scalar)
    D2 = pdist2(XA / bw_scalar, XB / bw_scalar, 'euclidean').^2;
    K = exp(-0.5 * D2);
end

function z_query = sloPolicyCVaRByOptimization(x_query, x_train, y_train, bw_scalar, h, b, beta)
    num_query = size(x_query, 1);

    K = gaussianKernelScaled(x_query, x_train, bw_scalar);
    row_sum = sum(K, 2) + eps;
    W = bsxfun(@rdivide, K, row_sum);

    lower_bound = 0;
    upper_bound = max(y_train) + 0.1 * (max(y_train) - min(y_train)) + 1;

    options = optimset('Display', 'off', 'TolX', 1e-3);

    z_query = zeros(num_query, 1);

    for i = 1:num_query
        weights_i = W(i, :)';
        objective_i = @(z) weightedConditionalCVARObjective(z, y_train, weights_i, h, b, beta);
        z_query(i) = fminbnd(objective_i, lower_bound, upper_bound, options);
    end
end

function value = weightedConditionalCVARObjective(z, y_support, weights, h, b, beta)
    weights = weights(:);
    weights = weights / (sum(weights) + eps);

    losses = h * max(z - y_support, 0) + b * max(y_support - z, 0);
    value = weightedEmpiricalCVAR(losses, weights, beta);
end

function value = weightedEmpiricalCVAR(losses, weights, beta)
    losses = losses(:);
    weights = weights(:);
    weights = weights / (sum(weights) + eps);

    [losses_sorted, idx] = sort(losses, 'ascend');
    weights_sorted = weights(idx);

    cumulative_weights = cumsum(weights_sorted);
    var_index = find(cumulative_weights >= beta, 1, 'first');

    if isempty(var_index)
        var_index = length(losses_sorted);
    end

    var_beta = losses_sorted(var_index);
    value = var_beta + sum(weights_sorted .* max(losses_sorted - var_beta, 0)) / (1 - beta);
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

% function y_limits = computeUnifiedYLim(values, force_nonnegative)
%     values = values(~isnan(values));
%     if isempty(values)
%         y_limits = [0, 1];
%         return;
%     end
% 
%     vmin = min(values);
%     vmax = max(values);
% 
%     if abs(vmax - vmin) < 1e-12
%         padding = max(1e-3, 0.05 * max(abs(vmax), 1));
%     else
%         padding = 0.05 * (vmax - vmin);
%     end
% 
%     lower_bound = vmin - padding;
%     upper_bound = vmax + padding;
% 
%     if force_nonnegative
%         lower_bound = max(0, lower_bound);
%     end
% 
%     y_limits = [lower_bound, upper_bound];
% end

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

function bw_scalar = theoryBandwidthScalar(sigma_ref, n_train, n_ref, d_x)
    bw_scalar = sigma_ref * (n_ref / n_train)^(1 / (d_x + 4));
    bw_scalar = max(bw_scalar, 1e-6);
end

function lambda_n = theoryLambda(lambda_ref, n_train, n_ref, alpha_reg)
    lambda_n = lambda_ref * (n_ref / n_train)^alpha_reg;
end

function T = buildMetricMeanTable(dimension_list, n_train_list, data_expected, data_exante, data_slo, value_names)
    num_dimensions = length(dimension_list);
    num_train_sizes = length(n_train_list);
    total_rows = num_dimensions * num_train_sizes;

    BlockDimension = zeros(total_rows, 1);
    FullContextDimension = zeros(total_rows, 1);
    TrainingSize = zeros(total_rows, 1);
    Value1 = zeros(total_rows, 1);
    Value2 = zeros(total_rows, 1);
    Value3 = zeros(total_rows, 1);

    row = 0;
    for dim_idx = 1:num_dimensions
        for n_idx = 1:num_train_sizes
            row = row + 1;
            BlockDimension(row) = dimension_list(dim_idx);
            FullContextDimension(row) = 2 * dimension_list(dim_idx);
            TrainingSize(row) = n_train_list(n_idx);
            Value1(row) = mean(data_expected(dim_idx, n_idx, :), 3, 'omitnan');
            Value2(row) = mean(data_exante(dim_idx, n_idx, :), 3, 'omitnan');
            Value3(row) = mean(data_slo(dim_idx, n_idx, :), 3, 'omitnan');
        end
    end

    T = table(BlockDimension, FullContextDimension, TrainingSize, Value1, Value2, Value3, ...
        'VariableNames', {'BlockDimension', 'FullContextDimension', 'TrainingSize', ...
        value_names{1}, value_names{2}, value_names{3}});
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