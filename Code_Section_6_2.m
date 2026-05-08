clear;
clc;
close all;

%% Experiment description
% Experiment for Section 6.2

% rng(1)

beta = 0.9;
d_z = 50;
d_x = 5;

% this parameter control the data-processing proce
eta = 3; % 1/3, 1, 3
tau = 2; %1 2
poly_deg = 4; % 1 4


n_test = 50000;

pd_standard = makedist('Normal', 'mu', 0, 'sigma', 1);
truncated_pd_standard = truncate(pd_standard, -2, 2);

B = rand(d_z, d_x) < 0.5;
L = -0.0025 * tau + (0.0025 * tau + 0.0025 * tau) * rand(d_z, 4);


% n = 100
% n_train = 100;
% sigma_kernel = 1.5;
% sigma_kernel_t = 0.05;
% lambda_z = 0.00001;
% lambda_t = 0.025;

% n = 500
n_train = 500;
sigma_kernel = 2.25;
sigma_kernel_t = 0.5;
lambda_z = 1e-9;
lambda_t = 1e-8;

N = 50;

% mean_cost_opt = zeros(N,1);
% std_cost_opt = zeros(N,1);
% relative_diff_cost_opt = zeros(N,1);
% sharpe_ratio_cost_opt = zeros(N,1);
% cvar_cost_opt = zeros(N,1);

mean_cost_ew = zeros(N,1);
std_cost_ew = zeros(N,1);
relative_diff_cost_ew = zeros(N,1);
sharpe_ratio_cost_ew = zeros(N,1);
cvar_cost_ew = zeros(N,1);

mean_cost_uncvar_test = zeros(N,1);
std_cost_uncvar_test = zeros(N,1);
relative_diff_cost_uncvar_test = zeros(N,1);
sharpe_ratio_cost_uncvar_test = zeros(N,1);
cvar_cost_uncvar_test = zeros(N,1);

mean_cost_ecvar_test = zeros(N,1);
std_cost_ecvar_test = zeros(N,1);
relative_diff_cost_ecvar_test = zeros(N,1);
sharpe_ratio_cost_ecvar_test = zeros(N,1);
cvar_cost_ecvar_test = zeros(N,1);


mean_cost_cvar_test = zeros(N,1);
std_cost_cvar_test = zeros(N,1);
relative_diff_cost_cvar_test = zeros(N,1);
sharpe_ratio_cost_cvar_test = zeros(N,1);
cvar_cost_cvar_test = zeros(N,1);

mean_cost_ecvar_regret_test = zeros(N,1);
std_cost_ecvar_regret_test = zeros(N,1);
relative_diff_cost_ecvar_regret_test = zeros(N,1);
sharpe_ratio_cost_ecvar_regret_test = zeros(N,1);
cvar_cost_ecvar_regret_test = zeros(N,1);

% mean_cost_ecvar_lineart_test = zeros(N,1);
% std_cost_ecvar_lineart_test = zeros(N,1);
% relative_diff_cost_ecvar_lineart_test = zeros(N,1);
% sharpe_ratio_cost_ecvar_lineart_test = zeros(N,1);
% cvar_cost_ecvar_lineart_test = zeros(N,1);

% mean_cost_ecvar_linear_test = zeros(N,1);
% std_cost_ecvar_linear_test = zeros(N,1);
% relative_diff_cost_ecvar_linear_test = zeros(N,1);
% sharpe_ratio_cost_ecvar_linear_test = zeros(N,1);
% cvar_cost_ecvar_linear_test = zeros(N,1);


%%

train_ex_ante_cvar_cvar = zeros(N,1);
train_ex_ante_cvar_mean = zeros(N,1);

train_ecvar_cvar = zeros(N,1);
train_ecvar_mean = zeros(N,1);


% train_ecvar_lineart_cvar = zeros(N,1);
% train_ecvar_lineart_mean = zeros(N,1);

% train_ecvar_linear_cvar = zeros(N,1);
% train_ecvar_linear_mean = zeros(N,1);


for i = 1:N
    tic
    yalmip('clear')
    x_train = random(truncated_pd_standard, d_x, n_train);
    x_test = random(truncated_pd_standard, d_x, n_test);
    
    r_mean = (0.05/sqrt(d_x) * (B * x_train) + 0.1^(1/poly_deg)).^poly_deg;
    r_mean_test = (0.05/sqrt(d_x) * (B * x_test) + 0.1^(1/poly_deg)).^poly_deg;
    
    epsilon_1 = random(truncated_pd_standard, 4, n_train);
    epsilon_2 = random(truncated_pd_standard, 1, n_train);
    epsilon_1_test = random(truncated_pd_standard, 4, n_test);
    epsilon_2_test = random(truncated_pd_standard, 1, n_test);
    
    r = r_mean + L * epsilon_1 + 0.01 *tau *epsilon_2;
    y_train = -r;
    r_test = r_mean_test + L * epsilon_1_test + 0.01 *tau *epsilon_2_test;
    y_test = -r_test;
    
    cost_opt_train = min(y_train);
    cost_opt_test = min(y_test);
    
    %% opt strategy (for comparison)
%     mean_cost_opt(i) = -mean(cost_opt_test);
%     std_cost_opt(i) = std(cost_opt_test);
%     relative_diff_cost_opt(i) = -sum(cost_opt_test' - cost_opt_test') / sum(cost_opt_test);
%     sharpe_ratio_cost_opt(i) = -mean(cost_opt_test)/std(cost_opt_test);
%     cvar_cost_opt(i) = -mean(cost_opt_test(cost_opt_test >= quantile(cost_opt_test, 0.9)));
    
    
    %% Equal weight strategy
    cost_ew = sum(y_test)'/50;
    mean_cost_ew(i) = -mean(cost_ew);
    std_cost_ew(i) = std(cost_ew);
    relative_diff_cost_ew(i) = -mean((cost_ew - cost_opt_test')./cost_opt_test');
    sharpe_ratio_cost_ew(i) = -mean(cost_ew)/std(cost_ew);
    cvar_cost_ew(i) = -mean(cost_ew(cost_ew >= quantile(cost_ew, 0.9)));
    
    %% unconditional cvar
    yalmip('clear')
    z = sdpvar(d_z,1);
    t = sdpvar(1,1);
    l = sdpvar(n_train, 1);
    
    cost = y_train'*z;
    Constraint = [z>=0, sum(z)<=1, l>= 0, l>= cost-t];
    
    Objective = t + 1/(n_train*(1-beta)) * sum(l,'all') + eta * mean(cost);
    
    ops = sdpsettings('verbose', 0);
    optimize(Constraint, Objective, ops);
    
    cost_uncvar_test = y_test'*value(z);
    
    mean_cost_uncvar_test(i) = -mean(cost_uncvar_test);
    std_cost_uncvar_test(i) = std(cost_uncvar_test);
    relative_diff_cost_uncvar_test(i) = -mean((cost_uncvar_test - cost_opt_test')./cost_opt_test');
    sharpe_ratio_cost_uncvar_test(i) = -mean(cost_uncvar_test)/std(cost_uncvar_test);
    cvar_cost_uncvar_test(i) = -mean(cost_uncvar_test(cost_uncvar_test >= quantile(cost_uncvar_test, 0.9)));
    
    %% ecvar + mean
    yalmip('clear')
    D = pdist2(x_train', x_train', 'euclidean').^2;
    K = exp(-D / (2 * sigma_kernel^2));
    K_t = exp(-D / (2 * sigma_kernel_t^2));
    
    alpha_z = sdpvar(n_train, d_z);
    alpha_t = sdpvar(n_train, 1);
    l = sdpvar(n_train, 1);
    
    z = K * alpha_z;
    t = K_t * alpha_t;
    
    cost = sum(y_train'.*z,2);
    
    Constraint = [l>=0, l>= cost-t, z>=0, sum(z,2)<=1];
    
    Objective = 1/n_train*sum(t,'all') + 1/(n_train*(1-beta)) * sum(l,'all') + eta * mean(cost) +...
        lambda_z * sum(alpha_z.* alpha_z, 'all') +  lambda_t * sum(alpha_t.* alpha_t,'all');
    
    ops = sdpsettings('verbose', 0);
    optimize(Constraint, Objective, ops);
    
    % test ecvar strategy
    K_new = exp(-pdist2(x_test', x_train',  'euclidean').^2 / (2 * sigma_kernel^2));
    z_pred = K_new * value(alpha_z);
    z_pred = max(z_pred,0);
    
    row_sums = sum(z_pred, 2);
    rows_to_normalize = row_sums > 1;
    if any(rows_to_normalize)
        row_norms = vecnorm(z_pred(rows_to_normalize, :), 2, 2);
        z_pred(rows_to_normalize, :) = z_pred(rows_to_normalize, :) ./ row_sums(rows_to_normalize);
    end
    cost_ecvar_test = sum(y_test'.* z_pred,2);
    
    mean_cost_ecvar_test(i) = -mean(cost_ecvar_test);
    std_cost_ecvar_test(i) = std(cost_ecvar_test);
    relative_diff_cost_ecvar_test(i) = -mean((cost_ecvar_test - cost_opt_test')./cost_opt_test');
    sharpe_ratio_cost_ecvar_test(i) = -mean(cost_ecvar_test)/std(cost_ecvar_test);
    cvar_cost_ecvar_test(i) = -mean(cost_ecvar_test(cost_ecvar_test >= quantile(cost_ecvar_test, 0.9)));

    train_ecvar_cvar(i) = -value(1/n_train*sum(t,'all') + 1/(n_train*(1-beta)) * sum(l,'all'));
    train_ecvar_mean(i) = -value(mean(cost));


    
    %% ex-ante cvar
    yalmip('clear')
    D = pdist2(x_train', x_train', 'euclidean').^2;
    K = exp(-D / (2 * sigma_kernel^2));
    
    alpha_z = sdpvar(n_train, d_z);
    t = sdpvar(1, 1);
    l = sdpvar(n_train, 1);
    
    z = K * alpha_z;
    
    cost = sum(y_train'.*z,2);
    
    Constraint = [l>=0, l>= cost-t, z>=0, sum(z,2)<=1];
    
    Objective = t + 1/(n_train*(1-beta)) * sum(l,'all') + eta * mean(cost) +...
        lambda_z * sum(alpha_z.* alpha_z, 'all');
    
    ops = sdpsettings('verbose', 0);
    optimize(Constraint, Objective, ops);
    
    % test ecvar strategy
    K_new = exp(-pdist2(x_test', x_train',  'euclidean').^2 / (2 * sigma_kernel^2));
    z_pred = K_new * value(alpha_z);
    z_pred = max(z_pred,0);
    
    row_sums = sum(z_pred, 2);
    rows_to_normalize = row_sums > 1;
    if any(rows_to_normalize)
        row_norms = vecnorm(z_pred(rows_to_normalize, :), 2, 2);
        z_pred(rows_to_normalize, :) = z_pred(rows_to_normalize, :) ./ row_sums(rows_to_normalize);
    end
    cost_cvar_test = sum(y_test'.* z_pred,2);
    
    mean_cost_cvar_test(i) = -mean(cost_cvar_test);
    std_cost_cvar_test(i) = std(cost_cvar_test);
    relative_diff_cost_cvar_test(i) = -mean((cost_cvar_test - cost_opt_test')./cost_opt_test');
    sharpe_ratio_cost_cvar_test(i) = -mean(cost_cvar_test)/std(cost_cvar_test);
    cvar_cost_cvar_test(i) = -mean(cost_cvar_test(cost_cvar_test >= quantile(cost_cvar_test, 0.9)));

    train_ex_ante_cvar_cvar(i) = -value(t + 1/(n_train*(1-beta)) * sum(l,'all'));
    train_ex_ante_cvar_mean(i) = -value(mean(cost));
    
    i
    toc
end
%%
aaaverage_mean_cost_ew = mean(mean_cost_ew);
% average_std_cost_ew = mean(std_cost_ew);
adaaverage_relative_diff_cost_ew = mean(relative_diff_cost_ew);
aeaaverage_sharpe_ratio_cost_ew = mean(sharpe_ratio_cost_ew);
abaaverage_cvar_cost_ew = mean(cvar_cost_ew);
acaaverage_cvarmean_cost_ew = eta*aaaverage_mean_cost_ew + abaaverage_cvar_cost_ew;

aabaverage_mean_cost_uncvar_test = mean(mean_cost_uncvar_test);
% average_std_cost_uncvar_test = mean(std_cost_uncvar_test);
adbaverage_relative_diff_cost_uncvar_test = mean(relative_diff_cost_uncvar_test);
aebaverage_sharpe_ratio_cost_uncvar_test = mean(sharpe_ratio_cost_uncvar_test);
abbaverage_cvar_cost_uncvar_test = mean(cvar_cost_uncvar_test);
acbaverage_cvarmean_cost_uncvar_test = eta*aabaverage_mean_cost_uncvar_test + abbaverage_cvar_cost_uncvar_test;

aacaverage_mean_cost_cvar_test = mean(mean_cost_cvar_test);
% average_std_cost_cvar_test = mean(std_cost_cvar_test);
adcaverage_relative_diff_cost_cvar_test = mean(relative_diff_cost_cvar_test);
aecaverage_sharpe_ratio_cost_cvar_test = mean(sharpe_ratio_cost_cvar_test);
abcaverage_cvar_cost_cvar_test = mean(cvar_cost_cvar_test);
accaverage_cvarmean_cost_cvar_test = eta*aacaverage_mean_cost_cvar_test + abcaverage_cvar_cost_cvar_test;

aadaverage_mean_cost_ecvar_test = mean(mean_cost_ecvar_test);
% average_std_cost_ecvar_test = mean(std_cost_ecvar_test);
addaverage_relative_diff_cost_ecvar_test = mean(relative_diff_cost_ecvar_test);
aedaverage_sharpe_ratio_cost_ecvar_test = mean(sharpe_ratio_cost_ecvar_test);
abdaverage_cvar_cost_ecvar_test = mean(cvar_cost_ecvar_test);
acdaverage_cvarmean_cost_ecvar_test = eta*aadaverage_mean_cost_ecvar_test + abdaverage_cvar_cost_ecvar_test;

%%
paramStr = sprintf('(%g,%g,%g)', eta, tau, poly_deg);

Model = {'EW'; 'MC'; 'CMEAC'; 'CMEC'};

E = [
    aaaverage_mean_cost_ew;
    aabaverage_mean_cost_uncvar_test;
    aacaverage_mean_cost_cvar_test;
    aadaverage_mean_cost_ecvar_test
];

CVaR = -[
    abaaverage_cvar_cost_ew;
    abbaverage_cvar_cost_uncvar_test;
    abcaverage_cvar_cost_cvar_test;
    abdaverage_cvar_cost_ecvar_test
];

etaE_minus_CVaR = [
    acaaverage_cvarmean_cost_ew;
    acbaverage_cvarmean_cost_uncvar_test;
    accaverage_cvarmean_cost_cvar_test;
    acdaverage_cvarmean_cost_ecvar_test
];

rela_regret = [
    adaaverage_relative_diff_cost_ew;
    adbaverage_relative_diff_cost_uncvar_test;
    adcaverage_relative_diff_cost_cvar_test;
    addaverage_relative_diff_cost_ecvar_test
];

fprintf('\n');
fprintf('---------------------------------------------------------------\n');
fprintf('%-12s %-8s %10s %10s %14s %14s\n', ...
    '(eta,tau,p)', 'Model', 'E', 'CVaR', 'etaE-CVaR', 'rela.regret');
fprintf('---------------------------------------------------------------\n');

for i = 1:4
    if i == 1
        groupName = paramStr;
    else
        groupName = '';
    end

    fprintf('%-12s %-8s %10.4f %10.4f %14.4f %14.4f\n', ...
        groupName, Model{i}, E(i), CVaR(i), etaE_minus_CVaR(i), rela_regret(i));
end

fprintf('---------------------------------------------------------------\n');


yalmip('clear')