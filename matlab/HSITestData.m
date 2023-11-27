clear;clc;close all
%% generate dataset
rng(0)
% addpath(genpath('lib'));
dataname = 'ICVL';
key = 'rad';
basedir = ['../data/', dataname];


sz = 512;
preprocess = @(x)(center_crop(rot90(x), sz, sz));
% preprocess = @(x)(center_crop(normalized(x), 340, 340)); % Pavia


%% for iid gaussian
datadir = [basedir, '/filefolder/test_gauss'];
savepath = [basedir, '/testset_gauss'];
if ~isfolder(savepath)
    mkdir(savepath);
end
fns = dir([datadir,'/*.mat']);
fns = {fns.name};

for sigma = [30,50,70]
    newdir = fullfile(savepath, [dataname, '_', num2str(sz), '_', num2str(sigma)]);
    generate_dataset(datadir, fns, newdir, sigma, key, preprocess);
end
newdir = fullfile(savepath, [dataname, '_', num2str(sz), '_', 'blind']);
generate_dataset_blind(datadir, fns, newdir, key, preprocess);

disp('---------------------add noise to clean testing datasets--------------------')

%% for complex
datadir = [basedir, '/filefolder/test_complex'];
savepath = [basedir, '/testset_complex'];
if ~isfolder(savepath)
    mkdir(savepath);
end
fns = dir([datadir,'/*.mat']);
fns = {fns.name};

%%% for non-iid gaussian
sigmas = [10 30 50 70];
newdir = fullfile(savepath, [dataname, '_', num2str(sz), '_', 'noniid']);
generate_dataset_noniid(datadir, fns, newdir, sigmas, key, preprocess);
%%% for non-iid gaussian + stripe
newdir = fullfile(savepath, [dataname, '_', num2str(sz), '_', 'stripe']);
generate_dataset_stripe(datadir, fns, newdir, sigmas, key, preprocess);
%%% for non-iid gaussian + deadline
newdir = fullfile(savepath, [dataname, '_', num2str(sz), '_', 'deadline']);
generate_dataset_deadline(datadir, fns, newdir, sigmas, key, preprocess);
%%% for non-iid gaussian + impluse
newdir = fullfile(savepath, [dataname, '_', num2str(sz), '_', 'impulse']);
generate_dataset_impulse(datadir, fns, newdir, sigmas, key, preprocess);
%%% for mixture noise
newdir = fullfile(savepath, [dataname, '_', num2str(sz), '_','mixture']);
generate_dataset_mixture(datadir, fns, newdir, sigmas, key, preprocess);
