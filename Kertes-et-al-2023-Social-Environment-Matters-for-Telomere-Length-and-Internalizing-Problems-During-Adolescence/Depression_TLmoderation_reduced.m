%% Import data from text file
%%%% Load the data into the variable "dataMat" from wherever you are
%%%% storing it on your computer / server

%% Get ready for JAGS analysis

y = dataMat.depression;
sNum = dataMat.subject; % subject numbers

x = table2array( dataMat(:,[3:19,23,28:53]) ); % load predictors from the data matrix into x
xNames = varNames([3:19,23,28:53])'; % load variable names into xNames
% tl = (x(:,3) - nanmean(x(:,3))) ./ nanstd(x(:,3)); % uncomment if needed to standardize TL

for n = 1:size(x,2)
    x(:,n) = (x(:,n)-nanmean(x(:,n))) / nanstd(x(:,n)); % standardize each of our predictors
    if n ~= 3
        xNames(n)
        x = [x, x(:,n) .* tl]; % calculate interaction between each of the predictors and TL
    end
end

mdl = fitlm(x,y) % fit a default classical linear model, for comparison against the Bayesian results

nPredictors = size(x,2); % specify number of predictors

%% Send to JAGS
nChains = 4; % run 4 chains of MCMC samples
nBurnins = 1000; % throw out the first N samples from each chain (burn-ins)
nSamples = 5000; % run the MCMC samples for N recorded samples

for n = 1:nChains
    sv = struct; % blank structure for initial parameter values (let JAGS choose the starting points from the prior)
    init0(n) = sv;
end


datastruct = struct('y',y,'subject',sNum,'nSubjects',length(sNum),...
    'x',x,'nPredictors',nPredictors,'nData',length(y)); % input structure for JAGS


tic
[samples, stats, structArray] = matjags( ...
    datastruct, ...                     % Observed data   
    fullfile(pwd, 'Anxiety_TLModeration_JAGScode.txt'), ...    % File that contains model definition
    init0, ...                          % Initial values for latent variables
    'doparallel' , 0, ...      % Parallelization flag
    'nchains', nChains,...              % Number of MCMC chains
    'nburnin', nBurnins,...              % Number of burnin steps
    'nsamples', nSamples, ...           % Number of samples to extract
    'thin', 1, ...                      % Thinning parameter
    'monitorparams', {'b','b0','prec'}, ...     % List of latent variables to monitor
    'savejagsoutput' , 0 , ...          % Save command line output produced by JAGS?
    'verbosity' , 1 , ...               % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup' , 0 );                    % clean up of temporary files?
toc

%% calculate HDIs
xNames = [xNames; xNames] % double the list of predictors, so we have labels for direct effects and interactions
hdi = zeros(nPredictors,2); % blank matrix to hold the credible intervals
logBF = zeros(nPredictors,1); % blank vector to hold log bayes factor
BF = zeros(nPredictors,1); % blank vector to hold the bayes factors
prIncl = zeros(nPredictors,1); % blank vector to hold the inclusion probabilities

for j = 1:nPredictors
    theseSamples = samples.b(:,:,j); % get samples from the structure produced by JAGS
    theseSamples = theseSamples(:);
    
    thisHDI = HDIofMCMC(theseSamples,.95); % calculate 95% credible interval for each predictor (including interactions)
    if min(thisHDI) > 0 || max(thisHDI) < 0 % if the credible interval excludes zero, print the name of the variable and its estimates
        xNames(j)
        [j,thisHDI]
    end
    hdi(j,:) = thisHDI; % record the credible interval
    
    [height,~] = ksdensity(theseSamples,0); % identify the height of the posterior at zero
    BF(j) = normpdf(0,0,1) / height; % compute the Savage-Dickey bayes factor by comparing the height of the posterior (height) against the height of the prior (standard normal / N(0,1)) at zero
    logBF(j) = log(normpdf(0,0,1) / height); % compute the log of the bayes factor
    prIncl(j) = BF(j) / (1 + BF(j)); % transform the bayes factor from odds form to a probability, giving the probability of inclusion for each predictor
end