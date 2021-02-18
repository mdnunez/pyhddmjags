# recovery_test.py - Testing JAGS fits of HDDM models in JAGS using pyjags in Python 3, 
#                    This hierarchical model is usuaully appropriate for trials with intermixed experimental conditions
#
# Copyright (C) 2020 Michael D. Nunez, <mdnunez1@uci.edu>
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
# 05/14/20      Michael Nunez                             Original code
# 06/14/20      Michael Nunez                          Fixes and updates
# 06/15/20      Michael Nunez                         Fix to beta simulation
# 06/16/20      Michael Nunez                Generate new simulation data each time
# 06/17/20      Michael Nunez                       Fixes to recovery plots
# 06/18/20      Michael Nunez    Generate only one simulation and remove simulation index
# 07/06/20      Michael Nunez                Add summary function for parameter estimates
# 12/04/20      Michael Nunez               Call definitions from pyhddmjagsutils.py


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
#Note you could remove this if statement and replace with loading your own data to dictionary "gendata"

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


    genparam = dict()
    genparam['ndt'] = ndt
    genparam['beta'] = beta
    genparam['alpha'] = alpha
    genparam['delta'] = delta
    genparam['ndttrialrange'] = ndttrialrange
    genparam['deltatrialsd'] = deltatrialsd
    genparam['prob_lapse'] = prob_lapse
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['participant'] = participant
    genparam['condition'] = condition
    genparam['nparts'] = nparts
    genparam['nconds'] = nconds
    genparam['ntrials'] = ntrials
    genparam['N'] = N
    sio.savemat('data/genparam_test1.mat', genparam)
else:
    genparam = sio.loadmat('data/genparam_test1.mat')

# JAGS code

# Set random seed
np.random.seed(2020)

tojags = '''
model {
    
    ##########
    #Between-condition variability parameters priors
    ##########

    #Between-condition variability in drift rate to choice A
    deltasdcond ~ dgamma(1,1)

    ##########
    #Between-participant variability parameters priors
    ##########

    #Between-participant variability in non-decision time
    tersd ~ dgamma(.3,1)

    #Between-participant variability in Speed-accuracy trade-off
    alphasd ~ dgamma(1,1)

    #Between-participant variability in choice A start point bias
    betasd ~ dgamma(.3,1)

    #Between-participant variability in lapse trial probability
    problapsesd ~ dgamma(.3,1)

    #Between-participant variability in drift rate to choice A
    deltasd ~ dgamma(1,1)

    ##########
    #Hierarchical DDM parameter priors
    ##########

    #Hierarchical Non-decision time
    terhier ~ dnorm(.5, pow(.25,-2))

    #Hierarchical boundary parameter (speed-accuracy tradeoff)
    alphahier ~ dnorm(1, pow(.5,-2))

    #Hierarchical start point bias towards choice A
    betahier ~ dnorm(.5, pow(.25,-2))

    #Hierarchical lapse trial probability
    problapsehier ~ dnorm(.3, pow(.15,-2))

    #Hierarchical drift rate to choice A
    deltahier ~ dnorm(0, pow(2, -2))

    ##########
    #Participant-level DDM parameter priors
    ##########
    for (p in 1:nparts) {

        #Non-decision time
        ter[p] ~ dnorm(terhier, pow(tersd,-2))T(0, 1)

        #Boundary parameter (speed-accuracy tradeoff)
        alpha[p] ~ dnorm(alphahier, pow(alphasd,-2))T(0, 3)

        #Rightward start point bias towards choice A
        beta[p] ~ dnorm(betahier, pow(betasd,-2))T(0, 1)

        #Probability of a lapse trial
        problapse[p] ~ dnorm(problapsehier, pow(problapsesd,-2))T(0, 1)
        probDDM[p] <- 1 - problapse[p]

        #Participant-level drift rate to choice A
        deltapart[p] ~ dnorm(deltahier, pow(deltasd, -2))

        for (c in 1:nconds) {

            #Participant-level drift rate to choice A
            delta[p,c] ~ dnorm(deltapart[p], pow(deltasdcond, -2))

        }

    }

    ##########
    # Wiener likelihood and uniform mixture using Ones trick
    for (i in 1:N) {

        # Log density for DDM process of rightward/leftward RT
        ld_comp[i, 1] <- dlogwiener(y[i], alpha[participant[i]], ter[participant[i]], beta[participant[i]], delta[participant[i],condition[i]])

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

modelfile = 'jagscode/recovery_test1.jags'
f = open(modelfile, 'w')
f.write(tojags)
f.close()

# Track these variables
trackvars = ['deltasdcond', 
            'tersd', 'alphasd', 'betasd', 'problapsesd', 'deltasd',
            'terhier', 'alphahier', 'betahier', 'problapsehier','deltahier',
            'ter', 'alpha', 'beta', 'problapse', 'deltapart',
             'delta', 'DDMorLapse']

N = np.squeeze(genparam['N'])

# Input for mixture modeling
Ones = np.ones(N)
Constant = 20


#Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
participant = np.squeeze(genparam['participant'])
condition = np.squeeze(genparam['condition'])
nparts = np.squeeze(genparam['nparts'])
nconds = np.squeeze(genparam['nconds'])
ntrials = np.squeeze(genparam['ntrials'])

minrt = np.zeros(nparts)
for p in range(0,nparts):
    minrt[p] = np.min(rt[(participant == (p+1))])

initials = []
for c in range(0, nchains):
    chaininit = {
        'deltasdcond': np.random.uniform(.1, 3.),
        'tersd': np.random.uniform(.01, .2),
        'alphasd': np.random.uniform(.01, 1.),
        'betasd': np.random.uniform(.01, .2),
        'problapsesd': np.random.uniform(.01, .5),
        'deltasd': np.random.uniform(.1, 3.),
        'deltapart': np.random.uniform(-4., 4., size=nparts),
        'delta': np.random.uniform(-4., 4., size=(nparts,nconds)),
        'ter': np.random.uniform(.1, .5, size=nparts),
        'alpha': np.random.uniform(.5, 2., size=nparts),
        'beta': np.random.uniform(.2, .8, size=nparts),
        'problapse': np.random.uniform(.01, .1, size=nparts),
        'deltahier': np.random.uniform(-4., 4.),
        'terhier': np.random.uniform(.1, .5),
        'alphahier': np.random.uniform(.5, 2.),
        'betahier': np.random.uniform(.2, .8),
        'problapsehier': np.random.uniform(.01, .1)
    }
    for p in range(0, nparts):
        chaininit['ter'][p] = np.random.uniform(0., minrt[p]/2)
    initials.append(chaininit)
print('Fitting model 1 ...')
threaded = pyjags.Model(file=modelfile, init=initials,
                        data=dict(y=y, N=N, nparts=nparts, nconds=nconds, condition=condition,
                                  participant=participant, Ones=Ones, Constant=Constant),
                        chains=nchains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(nsamps, vars=trackvars, thin=10)
savestring = ('modelfits/genparam_test1_model1.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)

#Diagnostics
samples = sio.loadmat(savestring)
samples_diagrelevant = samples.copy()
samples_diagrelevant.pop('DDMorLapse', None) #Remove variable DDMorLapse to obtain Rhat diagnostics
diags = phju.diagnostic(samples_diagrelevant)

#Posterior distributions
plt.figure()
phju.jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig(('figures/delta_posteriors_model1.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ter'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig(('figures/ter_posteriors_model1.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['beta'])
plt.title('Posterior distributions of the start point parameter')
plt.savefig(('figures/beta_posteriors_model1.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig(('figures/alpha_posteriors_model1.png'), format='png',bbox_inches="tight")

#Recovery
plt.figure()
phju.recovery(samples['delta'],genparam['delta'][:, :])
plt.title('Recovery of the drift-rate')
plt.savefig(('figures/delta_recovery_model1.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['ter'],genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig(('figures/ter_recovery_model1.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['beta'],genparam['beta'])
plt.title('Recovery of the start point parameter')
plt.savefig(('figures/beta_recovery_model1.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['alpha'],genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig(('figures/alpha_recovery_model1.png'), format='png',bbox_inches="tight")

