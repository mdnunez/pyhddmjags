# regression_test.py - Testing JAGS fits of HDDM models with participant-level regressors in JAGS using pyjags in Python 3
#
# Copyright (C) 2020 Michael D. Nunez, <m.d.nunez@uva.nl>
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
# 06/29/20      Michael Nunez                             Original code
# 06/30/20      Michael Nunez                       Fix regression simulation
# 07/06/20      Michael Nunez                Add summary function for parameter estimates
# 12/04/20      Michael Nunez               Call definitions from pyhddmjagsutils.py
# 01/08/22      Michael Nunez      Change np.int() to int() based on new Deprication warning


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



### Simulations ###

# Generate samples from the joint-model of reaction time and choice
#Note you could remove this if statement and replace with loading your own data to dictionary "gendata"

if not os.path.exists('data/genparam_test3.mat'):

    # Number of simulated participants
    nparts = 40

    # Number of conditions
    nconds = 6

    # Number of trials per participant and condition
    ntrials = 50

    # Number of total trials in each simulation
    N = ntrials*nparts*nconds

    # Set random seed
    np.random.seed(2021)

    #Intercepts of linear regressions
    ndt_int = np.matlib.repmat(np.random.uniform(.4, .7, size=(1,nconds)),nparts,1) # Uniform from .4 to .7 seconds
    alpha_int = np.matlib.repmat(np.random.uniform(.8, 1.4, size=(1,nconds)),nparts,1) # Uniform from .8 to 1.4 evidence units
    delta_int = np.matlib.repmat(np.random.uniform(-2, 2, size=(1, nconds)),nparts,1) # Uniform from -2 to 2 evidence units per second

    #Slopes of linear regressions
    ndt_gamma = np.matlib.repmat(np.random.uniform(0, .1, size=(1,nconds)),nparts,1)
    alpha_gamma = np.matlib.repmat(np.random.uniform(-.1, .1, size=(1,nconds)),nparts,1)
    delta_gamma = np.matlib.repmat(np.random.uniform(-1, 1, size=(1,nconds)),nparts,1)

    #Regressors
    regressors1 = np.random.normal(size=(nparts,nconds)) #The same regressors for each parameter, from a standard normal distribution

    #True parameters
    ndt = ndt_int + ndt_gamma*regressors1
    alpha = alpha_int + alpha_gamma*regressors1
    delta = delta_int + delta_gamma*regressors1


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
            tempout = phju.simulratcliff(N=ntrials, Alpha= alpha[p,k], Tau= ndt[p,k], 
                Nu= delta[p,k], Eta= deltatrialsd[p], rangeTau=ndttrialrange[p])
            tempx = np.sign(np.real(tempout))
            tempt = np.abs(np.real(tempout))
            mindwanderx = np.random.randint(low=0,high=2,size=ntrials)*2 -1
            mindwandert = np.random.uniform(low=0,high=2,size=ntrials) # Randomly distributed from 0 to 2 seconds

            mindwander_trials = np.random.choice(ntrials, size=int(np.round(ntrials*(prob_lapse[p]/100))), replace=False)
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
    genparam['alpha'] = alpha
    genparam['delta'] = delta
    genparam['ndt_int'] = ndt_int
    genparam['alpha_int'] = alpha_int
    genparam['delta_int'] = delta_int
    genparam['ndt_gamma'] = ndt_gamma
    genparam['alpha_gamma'] = alpha_gamma
    genparam['delta_gamma'] = delta_gamma
    genparam['regressors1'] = regressors1
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
    sio.savemat('data/genparam_test3.mat', genparam)
else:
    genparam = sio.loadmat('data/genparam_test3.mat')



#Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
participant = np.squeeze(genparam['participant'])
condition = np.squeeze(genparam['condition'])
nparts = np.squeeze(genparam['nparts'])
nconds = np.squeeze(genparam['nconds'])
regressors1 = np.squeeze(genparam['regressors1'])
ntrials = np.squeeze(genparam['ntrials'])
N = np.squeeze(genparam['N'])

minrt = np.zeros((nparts,nconds))
for p in range(0,nparts):
    for c in range(0,nconds):
        minrt[p,c] = np.min(rt[((participant == (p+1)) & (condition == (c+1)))])



# Set random seed
np.random.seed(2021)

# Input for mixture modeling
Ones = np.ones(N)
Constant = 20

#JAGS code

tojags = '''
model {
    
    ##########
    #Between-condition variability priors
    ##########

    #Between-condition variability in drift rate to correct
    deltasdcond ~ dgamma(1,1)

    #Between-condition variability in non-decision time
    ndtsdcond ~ dgamma(.3,1)

    #Between-condition variability in speed-accuracy trade-off
    alphasdcond ~ dgamma(1,1)

    ##########
    #Between-participant variability priors
    ##########

    #Between-participant variability in lapse trial probability
    problapsesd ~ dgamma(.3,1)

    ##########
    #Hierarchical DDM parameter priors
    ##########

    #Hierarchical lapse trial probability
    problapsehier ~ dnorm(.3, pow(.15,-2))

    ##########
    #Condition-level DDM parameter priors
    ##########

    for (c in 1:nconds) {

        #Drift rate intercept
        delta_int[c] ~ dnorm(0, pow(6, -2))

        #Non-decision time intercept
        ndt_int[c] ~ dnorm(0, pow(2,-2))

        #Boundary parameter intercept
        alpha_int[c] ~ dnorm(0, pow(4,-2))

        #Effect of regressor1 on Drift rate 
        delta_gamma[c] ~ dnorm(0, pow(3, -2))

        #Effect of regressor1 on Non-decision time
        ndt_gamma[c] ~ dnorm(0, pow(1,-2))

        #Effect of regressor1 on boundary parameter
        alpha_gamma[c] ~ dnorm(0, pow(2,-2))


    }


    ##########
    #Participant-level DDM parameter priors
    ##########
    for (p in 1:nparts) {

        #Probability of a lapse trial
        problapse[p] ~ dnorm(problapsehier, pow(problapsesd,-2))T(0, 1)
        probDDM[p] <- 1 - problapse[p]

        for (c in 1:nconds) {

            #Participant-level drift rate to correct
            delta[p,c] ~ dnorm(delta_int[c] + delta_gamma[c]*regressors1[p,c], pow(deltasdcond, -2))

            #Non-decision time
            ndt[p,c] ~ dnorm(ndt_int[c] + ndt_gamma[c]*regressors1[p,c], pow(ndtsdcond,-2))T(0, 1)

            #Boundary parameter (speed-accuracy tradeoff)
            alpha[p,c] ~ dnorm(alpha_int[c] + alpha_gamma[c]*regressors1[p,c], pow(alphasdcond,-2))T(0, 3)

        }

    }

    ##########
    # Wiener likelihood and uniform mixture using Ones trick
    for (i in 1:N) {

        # Log density for DDM process of rightward/leftward RT
        ld_comp[i, 1] <- dlogwiener(y[i], alpha[participant[i],condition[i]], ndt[participant[i],condition[i]], .5, delta[participant[i],condition[i]])

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
burnin = 4000  # Note that scientific notation breaks pyjags
nsamps = 20000

modelfile = 'jagscode/regression_test3.jags'
f = open(modelfile, 'w')
f.write(tojags)
f.close()

# Track these variables
trackvars = ['deltasdcond', 'ndtsdcond', 'alphasdcond', 'problapsesd',
            'problapsehier', 'delta_int', 'ndt_int', 'alpha_int',
            'delta_gamma', 'ndt_gamma', 'alpha_gamma',
             'delta', 'ndt', 'alpha', 'problapse', 'DDMorLapse']


initials = []
for c in range(0, nchains):
    chaininit = {
        'deltasdcond': np.random.uniform(.1, 3.),
        'ndtsdcond': np.random.uniform(.01, .2),
        'alphasdcond': np.random.uniform(.01, 1.),
        'problapsesd': np.random.uniform(.01, .5),
        'problapsehier': np.random.uniform(.01, .1),
        'delta_int': np.random.uniform(-4., 4., size=nconds),
        'ndt_int': np.random.uniform(.1, .5, size=nconds),
        'alpha_int': np.random.uniform(.5, 2., size=nconds),
        'delta_gamma': np.random.uniform(-1., 1., size=nconds),
        'ndt_gamma': np.random.uniform(-.1, .1, size=nconds),
        'alpha_gamma': np.random.uniform(-.1, .1, size=nconds),
        'delta': np.random.uniform(-4., 4., size=(nparts,nconds)),
        'ndt': np.random.uniform(.1, .5, size=(nparts,nconds)),
        'alpha': np.random.uniform(.5, 2., size=(nparts,nconds)),
        'problapse': np.random.uniform(.01, .1, size=nparts)
    }
    for p in range(0, nparts):
        for c in range(0, nconds):
            chaininit['ndt'][p,c] = np.random.uniform(0., minrt[p,c]/2)
    initials.append(chaininit)
print('Fitting model 3 ...')
threaded = pyjags.Model(file=modelfile, init=initials,
                        data=dict(y=y, N=N, regressors1=regressors1, nparts=nparts, nconds=nconds, condition=condition,
                                  participant=participant, Ones=Ones, Constant=Constant),
                        chains=nchains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(nsamps, vars=trackvars, thin=10)
savestring = ('modelfits/genparam_test1_model3.mat')
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
plt.savefig(('figures/delta_posteriors_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ndt'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig(('figures/ndt_posteriors_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig(('figures/alpha_posteriors_model3.png'), format='png',bbox_inches="tight")

#Recovery
plt.figure()
phju.recovery(samples['delta'],genparam['delta'][:, :])
plt.title('Recovery of the drift-rate')
plt.savefig(('figures/delta_recovery_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['ndt'],genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig(('figures/ndt_recovery_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['alpha'],genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig(('figures/alpha_recovery_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['delta_int'],genparam['delta_int'][0,:])
plt.title('Recovery of the drift-rate intercept')
plt.savefig(('figures/delta_int_recovery_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['ndt_int'],genparam['ndt_int'][0,:])
plt.title('Recovery of the non-decision time intercept')
plt.savefig(('figures/ndt_int_recovery_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['alpha_int'],genparam['alpha_int'][0,:])
plt.title('Recovery of boundary parameter intercept')
plt.savefig(('figures/alpha_int_recovery_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['delta_gamma'],genparam['delta_gamma'][0,:])
plt.title('Recovery of the drift-rate slope')
plt.savefig(('figures/delta_gamma_recovery_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['ndt_gamma'],genparam['ndt_gamma'][0,:])
plt.title('Recovery of the non-decision time slope')
plt.savefig(('figures/ndt_gamma_recovery_model3.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['alpha_gamma'],genparam['alpha_gamma'][0,:])
plt.title('Recovery of boundary parameter slope')
plt.savefig(('figures/alpha_gamma_recovery_model3.png'), format='png',bbox_inches="tight")


