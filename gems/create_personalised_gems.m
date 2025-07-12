%% Personalized GEM Construction Script
% This script generates personalized genome-scale metabolic models (GEMs)
% using the Human-GEM template and GTEx expression data.
%
% Dependencies:
% - Human-GEM model (MATLAB struct format)
% - Gurobi solver (linked via RAVEN Toolbox)
% - RAVEN Toolbox with ftINIT and Human-GEM integration
%
% Workflow:
% 1. Load Human-GEM and configure solver
% 2. Prepare model for ftINIT (metabolic task preprocessing)
% 3. Load tissue-specific gene expression (GTEx-derived)
% 4. Construct personalized models for each tissue sample using ftINIT
% 5. Save each personalized GEM to disk

%% Setup Paths and Load Model

% Add Gurobi and Human-GEM paths recursively
addpath(genpath('GEMs/gurobi1201/linux64/matlab/'));   % Gurobi solver path
addpath(genpath('Thesis/GEMs/'));                     % Human-GEM and tools path

% Load Human-GEM model
load('Human-GEM/model/Human-GEM.mat', 'ihuman');

% Set RAVEN to use Gurobi as the solver
setRavenSolver('gurobi');

%% Prepare model for INIT

% This preprocesses Human-GEM to make it compatible with ftINIT,
% including setting up essential tasks
prepData = prepHumanModelForftINIT(...
    ihuman, ...
    false, ...
    'SysBioChalmers-Human-GEM/data/metabolicTasks/metabolicTasks_Essential.txt', ...
    'SysBioChalmers-Human-GEM/model/reactions.tsv' ...
);

%% Load gene expression data

% Read GTEx-like gene expression data table
% First column: Gene symbols
% Remaining columns: Samples (e.g. patient tissues)
gtex_data = readtable('GEMs/gene_expression_gems.csv');

[~, n] = size(gtex_data);           % Total columns = 1 gene + N samples
numSamp = n - 1;                    % Number of samples (excluding gene column)

% Create data_struct compatible with ftINIT
data_struct.genes = gtex_data{:, 1};               % Gene identifiers
data_struct.tissues = gtex_data.Properties.VariableNames(2:n);  % Sample names
data_struct.levels = gtex_data{:, 2:n};             % Expression matrix
data_struct.threshold = 1;                          % Expression threshold (TPM > 1)

% Optional: inspect the data structure
disp(data_struct)

%% Construct personalized GEMs

models = cell(numSamp, 1);  % Preallocate cell array for models

for i = 1:numSamp
    disp(['Model: ' num2str(i) ' of ' num2str(numSamp)])

    % Run ftINIT for sample i
    % Inputs:
    % - prepData: preprocessed Human-GEM
    % - data_struct.tissues{i}: sample name
    % - [], [], data_struct: expression info and thresholds
    % - getHumanGEMINITSteps('1+0'): selects INIT step configuration
    % - true, true: enforce constraints and verbosity
    model = ftINIT(...
        prepData, ...
        data_struct.tissues{i}, ...
        [], [], ...
        data_struct, ...
        {}, ...
        getHumanGEMINITSteps('1+0'), ...
        true, ...
        true ...
    );

    % Save the resulting personalized GEM
    save(sprintf('personalised_GEMS/model_%d.mat', i), 'model', '-v7');
end
