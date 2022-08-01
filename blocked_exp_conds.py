# blocked_exp_conds.py - Code to fit a HDDM with fixed start point from a blocked condition experiment using JAGS in Python 3
#
# Copyright (C) 2021 Michael D. Nunez, <m.d.nunez@uva.nl>
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
# 06/24/20      Michael Nunez                             Original code
# 06/25/20      Michael NUnez             Removal of false plots, reduce problapse parameters
# 06/29/20      Michael Nunez                 Save out jags code with fixed name
# 07/06/20      Michael Nunez                Add summary function for parameter estimates
# 07/24/20      Michael Nunez             Include posterior probability for Lapse process
# 08/04/20      Michael Nunez                    Fix lapse probability calculation
# 12/04/20      Michael Nunez               Call definitions from pyhddmjagsutils.py, generate simulations
# 01/26/21      Michael Nunez                 Correct some comments in JAGS code


# Modules
import numpy as np
import pyjags
import scipy.io as sio
from scipy import stats
import warnings
import os
import matplotlib.pyplot as plt
import pyhddmjagsutils as phju



### Simulations ###

# Generate samples from the joint-model of reaction time and choice
#Note you could remove this if statement and replace with loading your own data to dictionary "indata"

if not os.path.exists('data/genparam_test1.mat'):
    # Number of simulated participants
    nparts = 40

    # Number of conditions
    nconds = 6

    # Number of trials per participant and condition
    ntrials = 50

    # Number of total trials in each simulation
    N = ntrials*nparts*nconds

    # Set random seed
    np.random.seed(2020)

    ndt = np.random.uniform(.15, .6, size=(nparts)) # Uniform from .15 to .6 seconds
    alpha = np.random.uniform(.8, 1.4, size=(nparts)) # Uniform from .8 to 1.4 evidence units
    beta = np.random.uniform(.3, .7, size=(nparts)) # Uniform from .3 to .7 * alpha
    delta = np.random.uniform(-4, 4, size=(nparts, nconds)) # Uniform from -4 to 4 evidence units per second
    ndttrialrange = np.random.uniform(0,.1, size=(nparts)) # Uniform from 0 to .1 seconds
    deltatrialsd = np.random.uniform(0, 2, size=(nparts)) # Uniform from 0 to 2 evidence units per second
    prob_lapse = np.random.uniform(0, 10, size=(nparts)) # From 0 to 10 percent of trials
    y = np.zeros((N))
    rt = np.zeros((N))
    acc = np.zeros((N))
    participant = np.zeros((N)) #Participant index
    condition = np.zeros((N)) #Condition index
    indextrack = np.arange(ntrials)
    for p in range(nparts):
        for k in range(nconds):
            tempout = phju.simulratcliff(N=ntrials, Alpha= alpha[p], Tau= ndt[p], Beta=beta[p], 
                Nu= delta[p,k], Eta= deltatrialsd[p], rangeTau=ndttrialrange[p])
            tempx = np.sign(np.real(tempout))
            tempt = np.abs(np.real(tempout))
            mindwanderx = np.random.randint(low=0,high=2,size=ntrials)*2 -1
            mindwandert = np.random.uniform(low=0,high=2,size=ntrials) # Randomly distributed from 0 to 2 seconds

            mindwander_trials = np.random.choice(ntrials, size=np.int(np.round(ntrials*(prob_lapse[p]/100))), replace=False)
            tempx[mindwander_trials] = mindwanderx[mindwander_trials]
            tempt[mindwander_trials] = mindwandert[mindwander_trials]
            y[indextrack] = tempx*tempt
            rt[indextrack] = tempt
            acc[indextrack] = (tempx + 1)/2
            participant[indextrack] = p+1
            condition[indextrack] = k+1
            indextrack += ntrials


    indata = dict()
    indata['ndt'] = ndt
    indata['beta'] = beta
    indata['alpha'] = alpha
    indata['delta'] = delta
    indata['ndttrialrange'] = ndttrialrange
    indata['deltatrialsd'] = deltatrialsd
    indata['prob_lapse'] = prob_lapse
    indata['rt'] = rt
    indata['acc'] = acc
    indata['y'] = y
    indata['participant'] = participant
    indata['condition'] = condition
    indata['nparts'] = nparts
    indata['nconds'] = nconds
    indata['ntrials'] = ntrials
    indata['N'] = N
    sio.savemat('data/genparam_test1.mat', indata)
else:
    indata = sio.loadmat('data/genparam_test1.mat')


N = np.squeeze(indata['N'])

# Input for mixture modeling
Ones = np.ones(N)
Constant = 20


#Fit model to data
y = np.squeeze(indata['y'])
rt = np.squeeze(indata['rt'])
participant = np.squeeze(indata['participant'])
condition = np.squeeze(indata['condition'])
nparts = np.squeeze(indata['nparts'])
nconds = np.squeeze(indata['nconds'])
ntrials = np.squeeze(indata['ntrials'])

minrt = np.zeros((nparts,nconds))
for p in range(0,nparts):
    for c in range(0,nconds):
        minrt[p,c] = np.min(rt[((participant == (p+1)) & (condition == (c+1)))])

# JAGS code

# Set random seed
np.random.seed(2020)

tojags = '''
model {
    
    ##########
    #Between-condition variability priors
    ##########

    #Between-condition variability in drift rate to correct
    deltasdcond ~ dgamma(1,1)

    #Between-condition variability in non-decision time
    tersdcond ~ dgamma(.3,1)

    #Between-condition variability in speed-accuracy trade-off
    alphasdcond ~ dgamma(1,1)

    ##########
    #Between-participant variability priors
    ##########

    #Between-participant variability in drift rate to correct
    deltasd ~ dgamma(1,1)

    #Between-participant variability in non-decision time
    tersd ~ dgamma(.3,1)

    #Between-participant variability in Speed-accuracy trade-off
    alphasd ~ dgamma(1,1)

    #Between-participant variability in lapse trial probability
    problapsesd ~ dgamma(.3,1)

    ##########
    #Hierarchical DDM parameter priors
    ##########

    #Hierarchical drift rate to correct
    deltahier ~ dnorm(0, pow(2, -2))

    #Hierarchical Non-decision time
    terhier ~ dnorm(.5, pow(.25,-2))

    #Hierarchical boundary parameter (speed-accuracy tradeoff)
    alphahier ~ dnorm(1, pow(.5,-2))

    #Hierarchical lapse trial probability
    problapsehier ~ dnorm(.3, pow(.15,-2))

    ##########
    #Participant-level DDM parameter priors
    ##########
    for (p in 1:nparts) {

        #Participant-level drift rate to correct
        deltapart[p] ~ dnorm(deltahier, pow(deltasd, -2))

        #Participant-level non-decision time
        terpart[p] ~ dnorm(terhier, pow(tersd,-2))

        #Participant-level boundary parameter (speed-accuracy tradeoff)
        alphapart[p] ~ dnorm(alphahier, pow(alphasd,-2))

        #Probability of a lapse trial
        problapse[p] ~ dnorm(problapsehier, pow(problapsesd,-2))T(0, 1)
        probDDM[p] <- 1 - problapse[p]


        ##########
        #Condition-level DDM parameter priors
        ##########
        for (c in 1:nconds) {

            #Drift rate to correct
            delta[p,c] ~ dnorm(deltapart[p], pow(deltasdcond, -2))

            #Non-decision time
            ter[p,c] ~ dnorm(terpart[p], pow(tersdcond,-2))T(0, 1)

            #Boundary parameter (speed-accuracy tradeoff)
            alpha[p,c] ~ dnorm(alphapart[p], pow(alphasdcond,-2))T(0, 3)

        }

    }

    ##########
    # Wiener likelihood and uniform mixture using Ones trick
    for (i in 1:N) {

        # Log density for DDM process of rightward/leftward RT
        ld_comp[i, 1] <- dlogwiener(y[i], alpha[participant[i],condition[i]], ter[participant[i],condition[i]], .5, delta[participant[i],condition[i]])

        # Log density for lapse trials (negative max RT to positive max RT)
        ld_comp[i, 2] <- logdensity.unif(y[i], -3, 3)

        # Select one of these two densities (Mixture of nonlapse and lapse trials)
        selected_density[i] <- exp(ld_comp[i, DDMorLapse[i]] - Constant)
        
        # Generate a likelihood for the MCMC sampler using a trick to maximize density value
        Ones[i] ~ dbern(selected_density[i])

        # Probability of mind wandering trials (lapse trials)
        DDMorLapse[i] ~ dcat( c(probDDM[participant[i]], problapse[participant[i]]) )
    }
}
'''


# pyjags code

# Make sure $LD_LIBRARY_PATH sees /usr/local/lib
# Make sure that the correct JAGS/modules-4/ folder contains wiener.so and wiener.la
pyjags.modules.load_module('wiener')
pyjags.modules.load_module('dic')
pyjags.modules.list_modules()

nchains = 6
burnin = 2000  # Note that scientific notation breaks pyjags
nsamps = 10000

modelfile = 'jagscode/blocked_exp_conds.jags'
f = open(modelfile, 'w')
f.write(tojags)
f.close()

# Track these variables
trackvars = ['deltasdcond', 'tersdcond', 'alphasdcond',
            'deltasd', 'tersd', 'alphasd', 'problapsesd',
            'deltahier', 'terhier', 'alphahier', 'problapsehier',
            'deltapart', 'terpart', 'alphapart',
             'delta', 'ter', 'alpha', 'problapse', 'DDMorLapse']


initials = []
for c in range(0, nchains):
    chaininit = {
        'deltasdcond': np.random.uniform(.1, 3.),
        'tersdcond': np.random.uniform(.01, .2),
        'alphasdcond': np.random.uniform(.01, 1.),
        'deltasd': np.random.uniform(.1, 3.),
        'tersd': np.random.uniform(.01, .2),
        'alphasd': np.random.uniform(.01, 1.),
        'problapsesd': np.random.uniform(.01, .5),
        'deltahier': np.random.uniform(-4., 4.),
        'terhier': np.random.uniform(.1, .5),
        'alphahier': np.random.uniform(.5, 2.),
        'problapsehier': np.random.uniform(.01, .1),
        'deltapart': np.random.uniform(-4., 4., size=nparts),
        'terpart': np.random.uniform(.1, .5, size=nparts),
        'alphapart': np.random.uniform(.5, 2., size=nparts),
        'problapse': np.random.uniform(.01, .1, size=nparts),
        'delta': np.random.uniform(-4., 4., size=(nparts,nconds)),
        'ter': np.random.uniform(.1, .5, size=(nparts,nconds)),
        'alpha': np.random.uniform(.5, 2., size=(nparts,nconds))
    }
    for p in range(0, nparts):
        for c in range(0, nconds):
            chaininit['ter'][p,c] = np.random.uniform(0., minrt[p,c]/2)
    initials.append(chaininit)
print('Fitting model 2 ...')
threaded = pyjags.Model(file=modelfile, init=initials,
                        data=dict(y=y, N=N, nparts=nparts, nconds=nconds, condition=condition,
                                  participant=participant, Ones=Ones, Constant=Constant),
                        chains=nchains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(nsamps, vars=trackvars, thin=10)
savestring = ('modelfits/genparam_test1_model2.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)

#Diagnostics
samples = sio.loadmat(savestring)
samples_diagrelevant = samples.copy()
samples_diagrelevant.pop('DDMorLapse', None) #Remove variable DDMorLapse to obtain Rhat diagnostics
diags = phju.diagnostic(samples_diagrelevant)

sumpos = phju.summary(samples_diagrelevant)

#Parameter estimates
print('The median posterior drift-rate (evidence units / sec) for participant 1 in condition 1 is %.2f with 95%% credible interval (%.2f , %.2f)' % (sumpos['delta']['median'][0,0], sumpos['delta']['95lower'][0,0], sumpos['delta']['95upper'][0,0]))
print('The median posterior non-decision time (sec) for participant 1 in condition 1 is %.2f with 95%% credible interval (%.2f , %.2f)' % (sumpos['ter']['median'][0,0], sumpos['ter']['95lower'][0,0], sumpos['ter']['95upper'][0,0]))
print('The median posterior boundary (evidence units) for participant 1 in condition 1 is %.2f with 95%% credible interval (%.2f , %.2f)' % (sumpos['alpha']['median'][0,0], sumpos['alpha']['95lower'][0,0], sumpos['alpha']['95upper'][0,0]))

#Posterior distributions
plt.figure()
phju.jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig(('figures/delta_posteriors_model2.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ter'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig(('figures/ter_posteriors_model2.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig(('figures/alpha_posteriors_model2.png'), format='png',bbox_inches="tight")

#Find posterior probability of each trial being from a lapse process
nchains = 6
nthinsamps = 1000
allDDMorLapse = np.reshape(samples['DDMorLapse'], samples['DDMorLapse'].shape[:-2] + (nchains * nthinsamps,))
lapse_probability = (np.mean(allDDMorLapse,axis=1) - 1) #Posterior probability estimate of lapse trial
plt.figure()
plt.scatter(np.arange(lapse_probability.shape[0]),lapse_probability)
plt.xlabel('Trial number')
plt.ylabel('Lapse posterior probability')
plt.savefig(('figures/lapse_posteriors_model2.png'), format='png',bbox_inches="tight")
