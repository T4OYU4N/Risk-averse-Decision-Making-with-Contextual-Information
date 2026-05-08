%% experiment1_conditional_opt_twofig.m
% ========================================================================
% Experiment for Section 6.1.2
%
% Prerequisites:
%   - YALMIP
%   - MOSEK
%
% Output files:
%   - Fig1_CVaR_combined.pdf / png / eps
%   - Fig2_Entropic_combined.pdf / png / eps
%   - experiment1_conditional_opt_twofig_results.mat
% ========================================================================

clear;
clc;
close all;

%% User options
save_figures = true;
save_results = true;
output_dir = pwd;

if (save_figures || save_results) && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Plot style
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 9);
set(groot, 'defaultAxesLineWidth', 0.8);
set(groot, 'defaultLineLineWidth', 1.2);

%% Settings
h = 0.2;
b = 1;
beta = 0.9;
a = 5;
gamma = 0.5;
mu_x = 1;
sd_x = 0.5;

d = 1;

n_x_context = 100;
n_y_context = 200;

rng(2024, 'twister');

pd_logx = makedist('Normal', 'mu', mu_x, 'sigma', sd_x);
pd_logx_trunc = truncate(pd_logx, mu_x - 2 * sd_x, mu_x + 2 * sd_x);

pd_standard = makedist('Normal', 'mu', 0, 'sigma', 1);
pd_standard_trunc = truncate(pd_standard, -2, 2);

ops = sdpsettings('solver', 'mosek', 'verbose', 0);

%% Context grid and conditional demand samples
u_x = linspace(0.01, 0.99, n_x_context - 1);
cdf_x_lower = normcdf(mu_x - 2 * sd_x, mu_x, sd_x);
cdf_x_upper = normcdf(mu_x + 2 * sd_x, mu_x, sd_x);
u_x = cdf_x_lower + u_x .* (cdf_x_upper - cdf_x_lower);
x_context = exp(norminv(u_x, mu_x, sd_x));     % row vector for plotting
x_context_col = x_context(:);                  % column vector for optimization

u_eps = linspace(0.001, 0.999, n_y_context - 1);
cdf_eps_lower = normcdf(-2, 0, 1);
cdf_eps_upper = normcdf(2, 0, 1);
u_eps = cdf_eps_lower + u_eps .* (cdf_eps_upper - cdf_eps_lower);
z_trunc_grid = norminv(u_eps, 0, 1);

x_matrix = repmat(x_context, n_y_context - 1, 1);
sd_epsilon = abs(40 - 3 * x_context) / 2;
epsilon_matrix = repmat(z_trunc_grid', 1, n_x_context - 1) .* repmat(sd_epsilon, n_y_context - 1, 1);
y_matrix = max(a * x_matrix + epsilon_matrix + 100, 0);

n_total = numel(y_matrix);
x_total = reshape(x_matrix, n_total, 1);
y_total = reshape(y_matrix, n_total, 1);

x_context_monomial = generateMonomial(x_context_col, d);
x_total_monomial = generateMonomial(x_total, d);

%% Ex ante CVaR
fprintf('Solving ex ante CVaR policy...\n');
yalmip('clear');

g = sdpvar(d + 1, 1);
t = sdpvar(1, 1);
diff_yz = sdpvar(n_total, 1);
diff_zy = sdpvar(n_total, 1);
l = sdpvar(n_total, 1);

z = x_total_monomial * g;

Constraints = [diff_yz >= 0, diff_zy >= 0, l >= 0];
Constraints = [Constraints, diff_zy >= z - y_total, diff_yz >= y_total - z];
Constraints = [Constraints, l >= b * diff_yz + h * diff_zy - t];

Objective = t + (1 / (n_total * (1 - beta))) * sum(l);

sol = optimize(Constraints, Objective, ops);
checkYalmipStatus(sol, 'Ex ante CVaR');

g_ex_ante_cvar = value(g);
z_ex_ante_cvar = x_context_monomial * g_ex_ante_cvar;

%% Conditional-optimal CVaR
fprintf('Computing conditional-optimal CVaR benchmark on the context grid...\n');
q_left = truncatedStandardQuantile(b * (1 - beta) / (h + b));
q_right = truncatedStandardQuantile((h * beta + b) / (h + b));
cvar_shift = (h / (h + b)) * q_left + (b / (h + b)) * q_right;

z_condopt_cvar = a * x_context_col + 100 + sd_epsilon(:) * cvar_shift;

%% Expected CVaR
fprintf('Solving expected CVaR policy...\n');
yalmip('clear');

g = sdpvar(d + 1, 1);
t = sdpvar(d + 1, 1);
diff_yz = sdpvar(n_total, 1);
diff_zy = sdpvar(n_total, 1);
l = sdpvar(n_total, 1);

z = x_total_monomial * g;
t_x = x_total_monomial * t;

Constraints = [diff_yz >= 0, diff_zy >= 0, l >= 0];
Constraints = [Constraints, diff_zy >= z - y_total, diff_yz >= y_total - z];
Constraints = [Constraints, l >= b * diff_yz + h * diff_zy - t_x];

Objective = sum(t_x) / n_total + (1 / (n_total * (1 - beta))) * sum(l);

sol = optimize(Constraints, Objective, ops);
checkYalmipStatus(sol, 'Expected CVaR');

g_expected_cvar = value(g);
z_expected_cvar = x_context_monomial * g_expected_cvar;

%% Ex ante entropic risk measure
fprintf('Solving ex ante entropic-RM policy...\n');
yalmip('clear');

g = sdpvar(d + 1, 1);
diff_yz = sdpvar(n_total, 1);
diff_zy = sdpvar(n_total, 1);

z = x_total_monomial * g;

Constraints = [diff_yz >= 0, diff_zy >= 0];
Constraints = [Constraints, diff_zy >= z - y_total, diff_yz >= y_total - z];

Objective = (1 / gamma) * logsumexp(gamma * (b * diff_yz + h * diff_zy));

sol = optimize(Constraints, Objective, ops);
checkYalmipStatus(sol, 'Ex ante entropic-RM');

g_ex_ante_ent = value(g);
z_ex_ante_ent = x_context_monomial * g_ex_ante_ent;

%% Conditional-optimal entropic risk measure
fprintf('Computing the conditional-optimal entropic benchmark on the context grid...\n');
z_condopt_ent = computeConditionalOptimalEntropic(y_matrix, h, b, gamma);

%% Expected entropic risk measure
fprintf('Solving expected entropic-RM policy...\n');
yalmip('clear');

g = sdpvar(d + 1, 1);
t = sdpvar(d + 1, 1);
diff_yz = sdpvar(n_total, 1);
diff_zy = sdpvar(n_total, 1);

z = x_total_monomial * g;
t_x = x_total_monomial * t;

Constraints = [diff_yz >= 0, diff_zy >= 0];
Constraints = [Constraints, diff_zy >= z - y_total, diff_yz >= y_total - z];

Objective = sum(t_x) / n_total - 1 / gamma ...
    + (1 / (gamma * n_total)) * sum(exp(gamma * (b * diff_yz + h * diff_zy - t_x)));

sol = optimize(Constraints, Objective, ops);
checkYalmipStatus(sol, 'Expected entropic-RM');

g_expected_ent = value(g);
z_expected_ent = x_context_monomial * g_expected_ent;

%% Risk-neutral baseline
fprintf('Solving risk-neutral baseline...\n');
yalmip('clear');

g = sdpvar(d + 1, 1);
diff_yz = sdpvar(n_total, 1);
diff_zy = sdpvar(n_total, 1);

z = x_total_monomial * g;

Constraints = [diff_yz >= 0, diff_zy >= 0];
Constraints = [Constraints, diff_zy >= z - y_total, diff_yz >= y_total - z];

Objective = sum(b * diff_yz + h * diff_zy);

sol = optimize(Constraints, Objective, ops);
checkYalmipStatus(sol, 'Risk-neutral baseline');

g_rn = value(g);
z_rn = x_context_monomial * g_rn;

%% Conditional metrics on the context grid
fprintf('Evaluating conditional metrics on the context grid...\n');

[expected_rn, cvar_rn, ent_rn] = conditionalMetrics(z_rn, y_matrix, h, b, beta, gamma);

[expected_condopt_cvar, cvar_condopt_cvar, ~] = conditionalMetrics(z_condopt_cvar, y_matrix, h, b, beta, gamma);
[expected_ex_ante_cvar, cvar_ex_ante_cvar, ~] = conditionalMetrics(z_ex_ante_cvar, y_matrix, h, b, beta, gamma);
[expected_expected_cvar, cvar_expected_cvar, ~] = conditionalMetrics(z_expected_cvar, y_matrix, h, b, beta, gamma);

[expected_condopt_ent, ~, ent_condopt_ent] = conditionalMetrics(z_condopt_ent, y_matrix, h, b, beta, gamma);
[expected_ex_ante_ent, ~, ent_ex_ante_ent] = conditionalMetrics(z_ex_ante_ent, y_matrix, h, b, beta, gamma);
[expected_expected_ent, ~, ent_expected_ent] = conditionalMetrics(z_expected_ent, y_matrix, h, b, beta, gamma);

%% Relative metrics used in figures
rel_cvar_rn = cvar_rn ./ cvar_condopt_cvar;
rel_cvar_ex_ante = cvar_ex_ante_cvar ./ cvar_condopt_cvar;
rel_cvar_expected = cvar_expected_cvar ./ cvar_condopt_cvar;

rel_ent_rn = ent_rn ./ ent_condopt_ent;
rel_ent_ex_ante = ent_ex_ante_ent ./ ent_condopt_ent;
rel_ent_expected = ent_expected_ent ./ ent_condopt_ent;

%% relative expected loss 
rel_expected_condopt_cvar = expected_condopt_cvar ./ expected_rn;
rel_expected_ex_ante_cvar = expected_ex_ante_cvar ./ expected_rn;
rel_expected_expected_cvar = expected_expected_cvar ./ expected_rn;

rel_expected_condopt_ent = expected_condopt_ent ./ expected_rn;
rel_expected_ex_ante_ent = expected_ex_ante_ent ./ expected_rn;
rel_expected_expected_ent = expected_expected_ent ./ expected_rn;

%% =========================================================================
%  Combined figure 1: CVaR
% =========================================================================
fprintf('Generating Fig. 1 (CVaR)...\n');

fig1 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, 17.2, 7.4], 'Renderer', 'painters');

ax1 = subplot(1, 2, 1);
plotDemandBands(x_context, y_matrix);
hold on;
plot_rn = plot(x_context_col, z_rn, '--k', 'LineWidth', 0.5);

plot_ex_ante_cvar = plot(x_context_col, z_ex_ante_cvar, 'r', 'LineWidth', 1.5);
plot_expected_cvar = plot(x_context_col, z_expected_cvar, 'c', 'LineWidth', 1.5);
plot_condopt_cvar = plot(x_context_col, z_condopt_cvar, '--b', 'LineWidth', 1.2);

xlim([1, 7]);
xticks(1:7);
ylim([110, 160]);

legend([plot_rn, plot_condopt_cvar, plot_ex_ante_cvar, plot_expected_cvar], ...
    'Risk-neutral policy', 'Conditional-CVaR policy', 'Ex-ante-CVaR policy', 'Expected-CVaR policy', ...
    'Location', 'northwest', 'Box', 'off', 'FontSize', 8);

xlabel('Covariate $X$', Interpreter='latex');
ylabel('Order quantity $z$', Interpreter='latex');
% text(0.02, 0.96, '(a)', 'Units', 'normalized', ...
%     'FontWeight', 'bold', 'VerticalAlignment', 'top');
% set(ax1, 'TickDir', 'out', 'Layer', 'top', 'Box', 'on');

ax2 = subplot(1, 2, 2);
hold on;

plot_rn_rel = plot(x_context_col, rel_cvar_rn, '--k', 'LineWidth', 0.5);
plot_ex_ante_rel = plot(x_context_col, rel_cvar_ex_ante, 'r', 'LineWidth', 1.5);
plot_expected_rel = plot(x_context_col, rel_cvar_expected, 'c', 'LineWidth', 1.5);
plot([1, 7], [1, 1], 'k:', 'LineWidth', 1.0, 'HandleVisibility', 'off');

xlim([1, 7]);
xticks(1:7);
% ylim(computePaddedLimits([1; rel_cvar_rn; rel_cvar_ex_ante; rel_cvar_expected]));
ylim([0.99, 1.22]);

legend([plot_rn_rel, plot_ex_ante_rel, plot_expected_rel], ...
    'Risk-neutral policy', 'Ex-ante-CVaR policy', 'Expected-CVaR policy', ...
    'Location', 'northwest', 'Box', 'off', 'FontSize', 8);

xlabel('Covariate $X$', Interpreter='latex');
ylabel('Normalized CVaR');
% text(0.02, 0.96, '(b)', 'Units', 'normalized', ...
%     'FontWeight', 'bold', 'VerticalAlignment', 'top');
% set(ax2, 'TickDir', 'out', 'Layer', 'top', 'Box', 'on');

if save_figures
    saveFigureAllFormats(fig1, fullfile(output_dir, 'Section_612_linear_cvar'));
end

%% =========================================================================
%  Combined figure 2: Entropic risk measure
% =========================================================================
fprintf('Generating Fig. 2 (entropic risk measure)...\n');

fig2 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, 17.2, 7.4], 'Renderer', 'painters');

ax3 = subplot(1, 2, 1);
plotDemandBands(x_context, y_matrix);
hold on;
plot_rn = plot(x_context_col, z_rn, '--k', 'LineWidth', 0.5);
plot_ex_ante_ent = plot(x_context_col, z_ex_ante_ent, 'm', 'LineWidth', 1.5);
plot_expected_ent = plot(x_context_col, z_expected_ent, 'Color', [0.6, 0.1961, 0.8], 'LineWidth', 1.5);
plot_condopt_ent = plot(x_context_col, z_condopt_ent, '--g', 'LineWidth', 1.2);



xlim([1, 7]);
xticks(1:7);
ylim([110, 160]);

legend([plot_rn, plot_condopt_ent, plot_ex_ante_ent, plot_expected_ent], ...
    'Risk-neutral policy', 'Conditional-entropic-RM policy', 'Ex-ante-entropic-RM policy', 'Expected-entropic-RM policy', ...
    'Location', 'northwest', 'Box', 'off', 'FontSize', 8);

xlabel('Covariate $X$', Interpreter='latex');
ylabel('Order quantity $z$', Interpreter='latex');
% text(0.02, 0.96, '(a)', 'Units', 'normalized', ...
    % 'FontWeight', 'bold', 'VerticalAlignment', 'top');
% set(ax3, 'TickDir', 'out', 'Layer', 'top', 'Box', 'on');

ax4 = subplot(1, 2, 2);
hold on;

plot_rn_rel = plot(x_context_col, rel_ent_rn, '--k', 'LineWidth', 0.5);
plot_ex_ante_rel = plot(x_context_col, rel_ent_ex_ante, 'm', 'LineWidth', 1.5);
plot_expected_rel = plot(x_context_col, rel_ent_expected, 'Color', [0.6, 0.1961, 0.8], 'LineWidth', 1.5);
plot([1, 7], [1, 1], 'k:', 'LineWidth', 1.0, 'HandleVisibility', 'off');


xlim([1, 7]);
xticks(1:7);
% ylim(computePaddedLimits([1; rel_ent_rn; rel_ent_ex_ante; rel_ent_expected]));
ylim([0.9, 1.7]);


legend([plot_rn_rel, plot_ex_ante_rel, plot_expected_rel], ...
    'Risk-neutral policy', 'Ex-ante-entropic-RM policy', 'Expected-entropic-RM policy', ...
    'Location', 'northwest', 'Box', 'off', 'FontSize', 8);

xlabel('Covariate $X$', Interpreter='latex');
ylabel('Normalized entropic risk measure');
% text(0.02, 0.96, '(b)', 'Units', 'normalized', ...
%     'FontWeight', 'bold', 'VerticalAlignment', 'top');
% set(ax4, 'TickDir', 'out', 'Layer', 'top', 'Box', 'on');

if save_figures
    saveFigureAllFormats(fig2, fullfile(output_dir, 'Section_612_linear_ent'));
end

%% Large-sample out-of-sample evaluation
n_x_test = 200;
n_y_test = 5000;

fprintf('\nGenerating a large out-of-sample test set...\n');

logx_test = random(pd_logx_trunc, n_x_test, 1);
x_test_context = exp(logx_test);

z_test_standard = random(pd_standard_trunc, n_y_test, n_x_test);
sd_epsilon_test = abs(40 - 3 * x_test_context') / 2;
epsilon_test_matrix = z_test_standard .* repmat(sd_epsilon_test, n_y_test, 1);
y_test_matrix = max(a * repmat(x_test_context', n_y_test, 1) + epsilon_test_matrix + 100, 0);

x_test_monomial = generateMonomial(x_test_context, d);

z_test_rn = x_test_monomial * g_rn;
z_test_ex_ante_cvar = x_test_monomial * g_ex_ante_cvar;
z_test_expected_cvar = x_test_monomial * g_expected_cvar;

z_test_ex_ante_ent = x_test_monomial * g_ex_ante_ent;
z_test_expected_ent = x_test_monomial * g_expected_ent;

z_test_condopt_cvar = a * x_test_context + 100 + (abs(40 - 3 * x_test_context) / 2) * cvar_shift;

fprintf('Computing the conditional-optimal entropic benchmark on the test set...\n');
z_test_condopt_ent = computeConditionalOptimalEntropic(y_test_matrix, h, b, gamma);

[expected_test_rn, cvar_test_rn, ent_test_rn] = conditionalMetrics(z_test_rn, y_test_matrix, h, b, beta, gamma);

[expected_test_condopt_cvar, cvar_test_condopt_cvar, ~] = conditionalMetrics(z_test_condopt_cvar, y_test_matrix, h, b, beta, gamma);
[expected_test_ex_ante_cvar, cvar_test_ex_ante_cvar, ~] = conditionalMetrics(z_test_ex_ante_cvar, y_test_matrix, h, b, beta, gamma);
[expected_test_expected_cvar, cvar_test_expected_cvar, ~] = conditionalMetrics(z_test_expected_cvar, y_test_matrix, h, b, beta, gamma);

[expected_test_condopt_ent, ~, ent_test_condopt_ent] = conditionalMetrics(z_test_condopt_ent, y_test_matrix, h, b, beta, gamma);
[expected_test_ex_ante_ent, ~, ent_test_ex_ante_ent] = conditionalMetrics(z_test_ex_ante_ent, y_test_matrix, h, b, beta, gamma);
[expected_test_expected_ent, ~, ent_test_expected_ent] = conditionalMetrics(z_test_expected_ent, y_test_matrix, h, b, beta, gamma);

fprintf('\nOut-of-sample evaluation uses %d contexts and %d conditional demand samples per context (%d joint observations in total).\n', ...
    n_x_test, n_y_test, n_x_test * n_y_test);

all_model_names = {'RN'; ...
    'Conditional-optimal CVaR'; ...
    'Ex ante CVaR'; ...
    'Expected CVaR'; ...
    'Conditional-optimal entropic-RM'; ...
    'Ex ante entropic-RM'; ...
    'Expected entropic-RM'};

joint_expected_loss_all = [mean(expected_test_rn); ...
    mean(expected_test_condopt_cvar); ...
    mean(expected_test_ex_ante_cvar); ...
    mean(expected_test_expected_cvar); ...
    mean(expected_test_condopt_ent); ...
    mean(expected_test_ex_ante_ent); ...
    mean(expected_test_expected_ent)];

joint_loss_table = table(all_model_names, joint_expected_loss_all, ...
    'VariableNames', {'Model', 'JointExpectedLoss'});

fprintf('\nJoint expected loss on the large test set:\n');
disp(joint_loss_table);

cvar_model_names = {'RN'; 'Conditional-optimal CVaR'; 'Ex ante CVaR'; 'Expected CVaR'};

joint_expected_loss_cvar = [mean(expected_test_rn); ...
    mean(expected_test_condopt_cvar); ...
    mean(expected_test_ex_ante_cvar); ...
    mean(expected_test_expected_cvar)];

average_conditional_cvar = [mean(cvar_test_rn); ...
    mean(cvar_test_condopt_cvar); ...
    mean(cvar_test_ex_ante_cvar); ...
    mean(cvar_test_expected_cvar)];

gap_to_conditional_opt_cvar = average_conditional_cvar - average_conditional_cvar(2);

cvar_results_table = table(cvar_model_names, joint_expected_loss_cvar, average_conditional_cvar, gap_to_conditional_opt_cvar, ...
    'VariableNames', {'Model', 'JointExpectedLoss', 'AverageConditionalCVaR', 'GapToConditionalOptimalCVaR'});

fprintf('\nCVaR-oriented out-of-sample performance:\n');
fprintf('Smaller GapToConditionalOptimalCVaR indicates better conditional optimality under CVaR.\n');
disp(cvar_results_table);

ent_model_names = {'RN'; 'Conditional-optimal entropic-RM'; 'Ex ante entropic-RM'; 'Expected entropic-RM'};

joint_expected_loss_ent = [mean(expected_test_rn); ...
    mean(expected_test_condopt_ent); ...
    mean(expected_test_ex_ante_ent); ...
    mean(expected_test_expected_ent)];

average_conditional_ent = [mean(ent_test_rn); ...
    mean(ent_test_condopt_ent); ...
    mean(ent_test_ex_ante_ent); ...
    mean(ent_test_expected_ent)];

gap_to_conditional_opt_ent = average_conditional_ent - average_conditional_ent(2);

ent_results_table = table(ent_model_names, joint_expected_loss_ent, average_conditional_ent, gap_to_conditional_opt_ent, ...
    'VariableNames', {'Model', 'JointExpectedLoss', 'AverageConditionalEntropicRM', 'GapToConditionalOptimalEntropicRM'});

fprintf('\nEntropic-oriented out-of-sample performance:\n');
fprintf('Smaller GapToConditionalOptimalEntropicRM indicates better conditional optimality under the entropic risk measure.\n');
disp(ent_results_table);

%% Save results
if save_results
    results = struct();

    results.parameters = struct( ...
        'h', h, 'b', b, 'beta', beta, 'a', a, 'gamma', gamma, ...
        'mu_x', mu_x, 'sd_x', sd_x, 'd', d, ...
        'n_x_context', n_x_context, 'n_y_context', n_y_context, ...
        'n_x_test', n_x_test, 'n_y_test', n_y_test);

    results.x_context = x_context_col;
    results.y_matrix = y_matrix;

    results.policy_coefficients = struct( ...
        'g_rn', g_rn, ...
        'g_ex_ante_cvar', g_ex_ante_cvar, ...
        'g_expected_cvar', g_expected_cvar, ...
        'g_ex_ante_ent', g_ex_ante_ent, ...
        'g_expected_ent', g_expected_ent, ...
        'cvar_shift', cvar_shift);

    results.training_policies = struct( ...
        'z_rn', z_rn, ...
        'z_condopt_cvar', z_condopt_cvar, ...
        'z_ex_ante_cvar', z_ex_ante_cvar, ...
        'z_expected_cvar', z_expected_cvar, ...
        'z_condopt_ent', z_condopt_ent, ...
        'z_ex_ante_ent', z_ex_ante_ent, ...
        'z_expected_ent', z_expected_ent);

    results.training_metrics = struct( ...
        'expected_rn', expected_rn, ...
        'cvar_rn', cvar_rn, ...
        'ent_rn', ent_rn, ...
        'expected_condopt_cvar', expected_condopt_cvar, ...
        'cvar_condopt_cvar', cvar_condopt_cvar, ...
        'expected_ex_ante_cvar', expected_ex_ante_cvar, ...
        'cvar_ex_ante_cvar', cvar_ex_ante_cvar, ...
        'expected_expected_cvar', expected_expected_cvar, ...
        'cvar_expected_cvar', cvar_expected_cvar, ...
        'expected_condopt_ent', expected_condopt_ent, ...
        'ent_condopt_ent', ent_condopt_ent, ...
        'expected_ex_ante_ent', expected_ex_ante_ent, ...
        'ent_ex_ante_ent', ent_ex_ante_ent, ...
        'expected_expected_ent', expected_expected_ent, ...
        'ent_expected_ent', ent_expected_ent, ...
        'rel_cvar_rn', rel_cvar_rn, ...
        'rel_cvar_ex_ante', rel_cvar_ex_ante, ...
        'rel_cvar_expected', rel_cvar_expected, ...
        'rel_ent_rn', rel_ent_rn, ...
        'rel_ent_ex_ante', rel_ent_ex_ante, ...
        'rel_ent_expected', rel_ent_expected, ...
        'rel_expected_condopt_cvar', rel_expected_condopt_cvar, ...
        'rel_expected_ex_ante_cvar', rel_expected_ex_ante_cvar, ...
        'rel_expected_expected_cvar', rel_expected_expected_cvar, ...
        'rel_expected_condopt_ent', rel_expected_condopt_ent, ...
        'rel_expected_ex_ante_ent', rel_expected_ex_ante_ent, ...
        'rel_expected_expected_ent', rel_expected_expected_ent);

    results.out_of_sample_tables = struct( ...
        'joint_loss_table', joint_loss_table, ...
        'cvar_results_table', cvar_results_table, ...
        'ent_results_table', ent_results_table);

    save(fullfile(output_dir, 'Section_612_conditional_opt_twofig_results.mat'), 'results');
    fprintf('\nSaved results to:\n  %s\n', fullfile(output_dir, 'Section_612_conditional_opt_twofig_results.mat'));
end

if save_figures
    fprintf('\nSaved figures to:\n');
    fprintf('  %s\n', fullfile(output_dir, 'Fig1_CVaR_combined.pdf'));
    fprintf('  %s\n', fullfile(output_dir, 'Fig2_Entropic_combined.pdf'));
end

fprintf('\nFinished.\n');

%% ========================================================================
% Local functions
% ========================================================================

function q = truncatedStandardQuantile(p)
    cdf_lower = normcdf(-2, 0, 1);
    cdf_upper = normcdf(2, 0, 1);
    q = norminv(cdf_lower + p * (cdf_upper - cdf_lower), 0, 1);
end

function plotDemandBands(x_context, y_matrix)
    % Journal-friendly solid nested bands (no transparency).
    x = x_context(:)';
    y_sorted = sort(y_matrix, 1, 'ascend');
    n = size(y_sorted, 1);

    idx_outer_low  = max(1, round(0.05 * (n - 1)) + 1);
    idx_outer_high = min(n, round(0.95 * (n - 1)) + 1);

    idx_mid_low    = max(1, round(0.10 * (n - 1)) + 1);
    idx_mid_high   = min(n, round(0.90 * (n - 1)) + 1);

    idx_inner_low  = max(1, round(0.25 * (n - 1)) + 1);
    idx_inner_high = min(n, round(0.75 * (n - 1)) + 1);

    hold on;
    fill([x, fliplr(x)], ...
         [y_sorted(idx_outer_low, :), fliplr(y_sorted(idx_outer_high, :))], ...
         [1.00, 0.99, 0.90], 'EdgeColor', 'none', 'HandleVisibility', 'off');

    fill([x, fliplr(x)], ...
         [y_sorted(idx_mid_low, :), fliplr(y_sorted(idx_mid_high, :))], ...
         [1.00, 0.97, 0.80], 'EdgeColor', 'none', 'HandleVisibility', 'off');

    fill([x, fliplr(x)], ...
         [y_sorted(idx_inner_low, :), fliplr(y_sorted(idx_inner_high, :))], ...
         [1.00, 0.94, 0.68], 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function z_opt = computeConditionalOptimalEntropic(y_matrix, h, b, gamma)
    n_context = size(y_matrix, 2);
    z_opt = zeros(n_context, 1);
    options = optimset('Display', 'off');

    for i = 1:n_context
        y_i = y_matrix(:, i);
        upper_bound = max(y_i) + 10;
        z_opt(i) = fminbnd(@(z) entropicObjective(z, y_i, h, b, gamma), 0, upper_bound, options);
    end
end

function value = entropicObjective(z, y, h, b, gamma)
    loss = h * max(z - y, 0) + b * max(y - z, 0);
    value = empiricalEntropic(loss, gamma);
end

function [expected_loss, conditional_cvar, conditional_ent] = conditionalMetrics(z_policy, y_matrix, h, b, beta, gamma)
    n_context = numel(z_policy);
    expected_loss = zeros(n_context, 1);
    conditional_cvar = zeros(n_context, 1);
    conditional_ent = zeros(n_context, 1);

    for i = 1:n_context
        loss = h * max(z_policy(i) - y_matrix(:, i), 0) + b * max(y_matrix(:, i) - z_policy(i), 0);
        expected_loss(i) = mean(loss);
        conditional_cvar(i) = empiricalCVAR(loss, beta);
        conditional_ent(i) = empiricalEntropic(loss, gamma);
    end
end

function value = empiricalCVAR(loss, beta)
    sorted_loss = sort(loss(:), 'ascend');
    n = numel(sorted_loss);
    k = floor(beta * n);

    if k >= n
        value = sorted_loss(end);
        return;
    end

    weight = k + 1 - beta * n;
    value = (weight * sorted_loss(k + 1) + sum(sorted_loss(k + 2:end))) / (n * (1 - beta));
end

function value = empiricalEntropic(loss, gamma)
    value = (1 / gamma) * log(mean(exp(gamma * loss)));
end

function monomial = generateMonomial(x,degree)
    [m,~] = size(x);
    monomial = zeros(m, degree + 1); 
    for d = 0:degree
        monomial(:,d + 1) = x.^d; 
    end
end

function checkYalmipStatus(sol, model_name)
    if sol.problem ~= 0
        error('%s failed: %s', model_name, sol.info);
    else
        fprintf('%s solved successfully.\n', model_name);
    end
end

function lim = computePaddedLimits(y)
    y = y(:);
    y = y(isfinite(y));

    ymin = min(y);
    ymax = max(y);
    span = ymax - ymin;

    if span < 1e-8
        span = 1e-8;
    end

    pad = 0.08 * span;
    lim = [ymin - pad, ymax + pad];
end

function saveFigureAllFormats(fig_handle, file_base)
    try
        exportgraphics(fig_handle, [file_base, '.pdf'], 'ContentType', 'vector');
        exportgraphics(fig_handle, [file_base, '.png'], 'Resolution', 600);
    catch
        print(fig_handle, [file_base, '.pdf'], '-dpdf', '-painters');
        print(fig_handle, [file_base, '.png'], '-dpng', '-r600');
    end

    print(fig_handle, [file_base, '.eps'], '-depsc2', '-painters');
end

