# model2b_experiment2.py - A simulation and model fit of Model 2b and Experiment 2
#
# Copyright (C) 2023 Michael D. Nunez, <m.d.nunez@uva.nl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 05/12/23       Michael NUnez                          Original code


# Modules
import numpy as np
import numpy.matlib
import pyjags
import scipy.io as sio
from scipy import stats
import warnings
import os
import matplotlib.pyplot as plt
import pyhddmjagsutils as phju

### Flags ###
test_fit = True

### Simulations ###

# Generate samples from the joint-model of reaction time and choice
#Note you could remove this if statement and replace with loading your own data to dictionary "gendata"

if not os.path.exists('data/experiment2.mat'):

    # Number of simulated participants per condition
    nparts = 50

    # Number of experimental conditions
    # 1 - stimulate parietal cortex
    # 2 - stimulate temporal cortex
    # 3 - stimulate neck (true sham condition)
    nconds = 3

    # Number of trials per participant and condition
    ntrials = 200

    # Number of total trials in each simulation
    N = ntrials*nparts*nconds

    # Set random seed
    np.random.seed(2021)

    # True non-decision time, the same across conditions
    ndt = np.random.uniform(.3, .5, size=(nparts))

    # True boundary, the same across conditions
    alpha = np.random.uniform(.8, 1.4, size=(nparts))

    # Drift intercept for the three conditions
    xi_0 = np.matlib.repmat(np.array([[1], [3], [1]]).T,nparts,1) + np.random.normal(loc=0, scale=1, size=(nparts,nconds))

    # Effect of CPP slopes for the three conditions
    xi_1 = np.matlib.repmat(np.array([[1], [0], [1]]).T,nparts,1) + np.random.normal(loc=0, scale=.25, size=(nparts,nconds))

    # Mean CPP slope for the three conditions
    mean_cpp_cond = np.matlib.repmat(np.array([[2], [1], [1]]).T,nparts,1) + np.random.normal(loc=0, scale=.25, size=(nparts,nconds))

    # Note: True mean drifts are np.array([3, 3, 2]) for the three conditions respectively

    y = np.zeros((N))
    rt = np.zeros((N))
    acc = np.zeros((N))
    participant = np.zeros((N)) #Participant index
    condition = np.zeros((N)) #Condition index
    cpp = np.zeros((N))
    indextrack = np.arange(ntrials)
    for p in range(nparts):
        for k in range(nconds):
            cpp_per_trial = np.random.normal(loc=mean_cpp_cond[p,k], scale=1, size=ntrials)
            drift_per_trial = xi_0[p,k] + xi_1[p,k]*cpp_per_trial
            tempout = np.empty((ntrials))
            for i in range(ntrials):
                tempout[i] = phju.simulratcliff(N=1, Alpha= alpha[p], Tau= ndt[p], Nu= drift_per_trial[i])
            tempx = np.sign(np.real(tempout))
            tempt = np.abs(np.real(tempout))

            y[indextrack] = tempx*tempt
            rt[indextrack] = tempt
            acc[indextrack] = (tempx + 1)/2
            participant[indextrack] = p+1
            condition[indextrack] = k+1
            cpp[indextrack] = cpp_per_trial
            indextrack += ntrials


    genparam = dict()
    genparam['ndt'] = ndt
    genparam['alpha'] = alpha
    genparam['xi_0'] = xi_0
    genparam['xi_1'] = xi_1
    genparam['mean_cpp_cond'] = mean_cpp_cond
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['cpp'] = cpp
    genparam['participant'] = participant
    genparam['condition'] = condition
    genparam['nparts'] = nparts
    genparam['nconds'] = nconds
    genparam['ntrials'] = ntrials
    genparam['N'] = N
    sio.savemat('data/experiment2.mat', genparam)
else:
    genparam = sio.loadmat('data/experiment2.mat')



#Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
participant = np.squeeze(genparam['participant'])
condition = np.squeeze(genparam['condition'])
nparts = np.squeeze(genparam['nparts'])
nconds = np.squeeze(genparam['nconds'])
cpp = np.squeeze(genparam['cpp'])
ntrials = np.squeeze(genparam['ntrials'])
N = np.squeeze(genparam['N'])

minrt = np.zeros((nparts,nconds))
for p in range(0,nparts):
    for c in range(0,nconds):
        minrt[p,c] = np.min(rt[((participant == (p+1)) & (condition == (c+1)))])



# Set random seed
np.random.seed(2021)

#JAGS code

tojags = '''
model {

    ##########
    #Participant- and condition-level parameter priors, this is not a hierarchical model
    ##########
    for (p in 1:nparts) {

        for (c in 1:nconds) {

            #Boundary parameter per participant and condition
            alpha[p, c] ~ dnorm(1, pow(.25,-2))T(0, 3)

            #Non-decision time per participant and condition
            ndt[p, c] ~ dnorm(.5, pow(.25,-2))T(0, 1)

            #Intercept parameter
            xi_0[p,c] ~ dnorm(0, pow(2, -2))
        }

        #Slope difference between conditions 1 and 3 to calculate BFs using Savage-Dickey
        xi_1_cond1_3_diff[p] ~ dnorm(0, pow(2, -2))

        #Slope difference between conditions 1 and 2 to calcualte BFs using Savage-Dickey
        xi_1_cond1_2_diff[p] ~ dnorm(0, pow(2, -2))

        #Condition 2 slope, note that all three condition xi_1 will not have the same prior variances
        xi_1_cond2[p] ~ dnorm(0, pow(2, -2))

        #Condition 1 slope
        xi_1[p, 1] = xi_1_cond1_2_diff[p] + xi_1_cond2[p]

        #Condition 2 slope
        xi_1[p, 2] = xi_1_cond2[p]

        #Condition 3 slope
        xi_1[p, 3] = xi_1_cond1_2_diff[p] + xi_1_cond2[p] - xi_1_cond1_3_diff[p]
    }

    ##########
    # Wiener likelihood with single-trial drift rate described by single-trial CPP amplitudes
    for (i in 1:N) {

        # Observations of accuracy*RT for DDM process for correct/incorrect
        y[i] ~ dwiener(alpha[participant[i],condition[i]], ndt[participant[i],condition[i]], .5, 
                       xi_0[participant[i],condition[i]] + xi_1[participant[i],condition[i]]*cpp[i])

    }
}
'''


# pyjags code

# Make sure $LD_LIBRARY_PATH sees /usr/local/lib
# Make sure that the correct JAGS/modules-4/ folder contains wiener.so and wiener.la
pyjags.modules.load_module('wiener')
pyjags.modules.load_module('dic')
pyjags.modules.list_modules()

if test_fit == True:
    nchains = 2
    burnin = 40
    nsamps = 200
else:
    nchains = 6
    burnin = 4000
    nsamps = 20000

modelfile = 'jagscode/model2b_experiment2.jags'
f = open(modelfile, 'w')
f.write(tojags)
f.close()

# Track these variables
trackvars = ['alpha', 'ndt', 'xi_0', 'xi_1',
            'xi_1_cond1_3_diff', 'xi_1_cond1_2_diff', 'xi_1_cond2']


initials = []
for c in range(0, nchains):
    chaininit = {
        'ndt': np.random.uniform(.1, .5, size=(nparts,nconds)),
        'alpha': np.random.uniform(.5, 2., size=(nparts,nconds)),
        'xi_0': np.random.uniform(-1., 1., size=(nparts,nconds)),
        'xi_1_cond1_3_diff': np.random.uniform(-1., 1., size=(nparts)),
        'xi_1_cond1_2_diff': np.random.uniform(-1., 1., size=(nparts)),
        'xi_1_cond2': np.random.uniform(-1., 1., size=(nparts))
    }
    for p in range(0, nparts):
        for c in range(0, nconds):
            chaininit['ndt'][p,c] = np.random.uniform(0., minrt[p,c]/2)
    initials.append(chaininit)
print('Fitting a version of Model 2b for Hypothetical Experiment 2...')
threaded = pyjags.Model(file=modelfile, init=initials,
                        data=dict(y=y, N=N, cpp=cpp, nparts=nparts, nconds=nconds, condition=condition,
                                  participant=participant),
                        chains=nchains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(nsamps, vars=trackvars, thin=10)
savestring = ('modelfits/model2b_experiment2.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)

#Diagnostics
samples = sio.loadmat(savestring)
diags = phju.diagnostic(samples)

#Posterior distributions
plt.figure()
phju.jellyfish(samples['xi_0'])
plt.title('Posterior distributions of the drift-rate intercepts')
plt.savefig(('figures/xi_0_posteriors_experiment2.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['xi_1'])
plt.title('Posterior distributions of the effects of single-trial CPP slopes on drift-rates')
plt.savefig(('figures/xi_1_posteriors_experiment2.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ndt'])
plt.title('Posterior distributions of the non-decision time parameters')
plt.savefig(('figures/ndt_posteriors_experiment2.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameters')
plt.savefig(('figures/alpha_posteriors_experiment2.png'), format='png',bbox_inches="tight")

#Recovery
plt.figure()
phju.recovery(samples['xi_0'],genparam['xi_0'][:, :])
plt.title('Recovery of the drift-rate intercepts')
plt.savefig(('figures/xi_0_recovery_experiment2.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['xi_1'],genparam['xi_1'][:, :])
plt.title('Recovery of the drift-rate intercepts')
plt.savefig(('figures/xi_1_recovery_experiment2.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['ndt'],np.matlib.repmat(genparam['ndt'],3,1).T)
plt.title('Recovery of the non-decision time parameter')
plt.savefig(('figures/ndt_recovery_experiment2.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['alpha'],np.matlib.repmat(genparam['alpha'],3,1).T)
plt.title('Recovery of boundary parameter')
plt.savefig(('figures/alpha_recovery_experiment2.png'), format='png',bbox_inches="tight")

# Calculate Bayes Factors using Savage-Dickey density ratio
# See also https://github.com/mdnunez/encodingN200/blob/64e0b4b924bf65d1070c9d00c3c1381c0ddf38af/Models/pdm5b_resultsmodel6.py

bf_cond1_3_diff = np.empty((nparts))
bf_cond1_2_diff = np.empty((nparts))
for p in range(nparts):

    # Number of chains
    nchains_result = samples['ndt'].shape[-1]

    # Number of samples per chain
    nsamps_result = samples['ndt'].shape[-2]

    # Slope difference between conditions 1 and 3
    samples_cond1_3_diff = np.reshape(samples['xi_1_cond1_3_diff'][p,:,:],(nchains_result*nsamps_result))

    # Slope difference between conditions 1 and 2
    samples_cond1_2_diff = np.reshape(samples['xi_1_cond1_2_diff'][p,:,:],(nchains_result*nsamps_result))

    # Estimate density curves from samples
    kde_cond1_3_diff = stats.gaussian_kde(samples_cond1_3_diff)
    kde_cond1_2_diff = stats.gaussian_kde(samples_cond1_2_diff)

    # Prior density of effect parameters, the same for both comparisons
    # This should match the JAGS priors
    denom = stats.norm.pdf(0, loc=0, scale=2)

    # Calculate Bayes Factors 01, evidence for the null hypothesis
    bf_cond1_3_diff[p] = kde_cond1_3_diff(0) / denom
    bf_cond1_2_diff[p] = kde_cond1_2_diff(0) / denom

print('The mean BF01, evidence for the null, for the slope difference conditions 1 and 3 is %3.2f' 
    % np.mean(bf_cond1_3_diff))

print('The number of BF01 > 5 (evidences for the null above 5) for the slope difference conditions 1 and 3 is %d' 
    % np.sum(bf_cond1_3_diff > 5))

print('The number of BF10 > 5 (evidences for the alternative above 5) for the slope difference conditions 1 and 3 is %d' 
    % np.sum(1/bf_cond1_3_diff > 5))

print('The mean BF01, evidence for the null, for the slope difference conditions 1 and 2 is %3.2f' 
    % np.mean(bf_cond1_2_diff))

print('The number of BF01 > 5 (evidences for the null above 5) for the slope difference conditions 1 and 2 is %d' 
    % np.sum(bf_cond1_2_diff > 5))

print('The number of BF10 > 5 (evidences for the alternative above 5) for the slope difference conditions 1 and 2 is %d' 
    % np.sum(1/bf_cond1_2_diff > 5))