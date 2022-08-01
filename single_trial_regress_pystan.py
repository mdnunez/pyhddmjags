# single_trial_regress_pystan.py - Testing fits of HDDM models without lapse process in Stan using pystan in Python 3
#
# Copyright (C) 2022 Michael D. Nunez, <m.d.nunez@uva.nl>
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
# 01/15/21      Michael Nunez                       Converted from nolapse_test.py


# Modules
import numpy as np
import pystan
import scipy.io as sio
from scipy import stats
import warnings
import os
import matplotlib.pyplot as plt
import pyhddmjagsutils as phju


### Simulations ###

# Generate samples from the joint-model of reaction time and choice
#Note you could remove this if statement and replace with loading your own data to dictionary "gendata"

if not os.path.exists('data/genparam_test4.mat'):
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

    ndt = np.random.uniform(.15, .6, size=(nparts)) # Uniform from .15 to .6 seconds
    alpha = np.random.uniform(.8, 1.4, size=(nparts)) # Uniform from .8 to 1.4 evidence units
    beta = np.random.uniform(.3, .7, size=(nparts)) # Uniform from .3 to .7 * alpha
    delta = np.random.uniform(-4, 4, size=(nparts, nconds)) # Uniform from -4 to 4 evidence units per second
    ndttrialrange = np.random.uniform(0,.1, size=(nparts)) # Uniform from 0 to .1 seconds
    deltatrialsd = np.random.uniform(0, 2, size=(nparts)) # Uniform from 0 to 2 evidence units per second
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
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['participant'] = participant
    genparam['condition'] = condition
    genparam['nparts'] = nparts
    genparam['nconds'] = nconds
    genparam['ntrials'] = ntrials
    genparam['N'] = N
    sio.savemat('data/genparam_test4.mat', genparam)
else:
    genparam = sio.loadmat('data/genparam_test4.mat')

# Stan code

tostan = '''
functions { 
  /* Wiener diffusion log-PDF for a single response (adapted from brms 1.10.2)
   * Arguments: 
   *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
   *   boundary: boundary separation parameter > 0
   *   ndt: non-decision time parameter > 0
   *   bias: initial bias parameter in [0, 1]
   *   drift: drift rate parameter
   * Returns:  
   *   a scalar to be added to the log posterior 
   */ 
   real diffusion_lpdf(real Y, real boundary, 
                              real ndt, real bias, real drift) { 
     
    if (Y >= 0) {
        return wiener_lpdf( fabs(Y) | boundary, ndt, bias, drift );
    } else {
        return wiener_lpdf( fabs(Y) | boundary, ndt, 1-bias, -drift );
    }

   }
} 
data {
    int<lower=1> N; // Number of trial-level observations
    int<lower=1> nconds; // Number of conditions
    int<lower=1> nparts; // Number of participants
    real y[N]; // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    int<lower=1> participant[N]; // Participant index
    int<lower=1> condition[N]; // Condition index
}
parameters {
    real<lower=0> deltasdcond; // Between-condition variability in drift rate to choice A
    real<lower=0> tersd; // Between-participant variability in non-decision time 
    real<lower=0> alphasd; // Between-participant variability in Speed-accuracy trade-off
    real<lower=0> betasd; // Between-participant variability in choice A start point bias
    real<lower=0> deltasd; // Between-participant variability in drift rate to choice A
    real terhier; // Hierarchical Non-decision time
    real alphahier; // Hierarchical boundary parameter (speed-accuracy tradeoff)
    real betahier; // Hierarchical start point bias towards choice A
    real deltahier; // Hierarchical drift rate to choice A
    vector<lower=0, upper=1>[nparts] ter; // Non-decision time
    vector<lower=0, upper=3>[nparts] alpha; // Boundary parameter (speed-accuracy tradeoff)
    vector<lower=0, upper=1>[nparts] beta; // Start point bias towards choice A
    vector[nparts] deltapart; // Participant-level drift rate to choice A
    matrix[nparts,nconds] delta; // Drift rate to choice A

}
model {
    
    // ##########
    // Between-condition variability priors
    // ##########

    // Between-condition variability in drift rate to choice A
    deltasdcond ~ gamma(1,1);

    // ##########
    // Between-participant variability priors
    // ##########

    // Between-participant variability in non-decision time
    tersd ~ gamma(.3,1);

    // Between-participant variability in Speed-accuracy trade-off
    alphasd ~ gamma(1,1);

    //Between-participant variability in choice A start point bias
    betasd ~ gamma(.3,1);

    // Between-participant variability in drift rate to choice A
    deltasd ~ gamma(1,1);


    // ##########
    // Hierarchical DDM parameter priors
    // ##########

    // Hierarchical Non-decision time
    terhier ~ normal(.5,.25);

    // Hierarchical boundary parameter (speed-accuracy tradeoff)
    alphahier ~ normal(1, .5);

    // Hierarchical start point bias towards choice A
    betahier ~ normal(.5, .25);

    // Hierarchical drift rate to choice A
    deltahier ~ normal(0, 2);


    // ##########
    // Participant-level DDM parameter priors
    // ##########
    for (p in 1:nparts) {

        // Participant-level non-decision time
        ter[p] ~ normal(terhier, tersd) T[0, 1];

        // Participant-level boundary parameter (speed-accuracy tradeoff)
        alpha[p] ~ normal(alphahier, alphasd) T[0, 3];

        //Start point bias towards choice A
        beta[p] ~ normal(betahier, betasd) T[0, 1];

        // Participant-level drift rate to correct
        deltapart[p] ~ normal(deltahier, deltasd);

        // ##########
        // Condition-level DDM parameter priors
        // ##########
        for (c in 1:nconds) {

            // Drift rate to correct
            delta[p,c] ~ normal(deltapart[p], deltasdcond);

        }

    }
    // Wiener likelihood
    for (i in 1:N) {

        target += diffusion_lpdf( y[i] | alpha[participant[i]], ter[participant[i]], beta[participant[i]], delta[participant[i],condition[i]]);
    }
}
'''

# pystan code

nchains = 6
burnin = 2000
nsamps = 10000

modelfile = f'stancode/nolapse_test.stan'
f = open(modelfile, 'w')
f.write(tostan)
f.close()

# Track these variables
trackvars = ['deltasdcond', 
            'tersd', 'alphasd', 'betasd', 'deltasd',
            'terhier', 'alphahier', 'betahier', 'deltahier',
            'ter', 'alpha', 'beta', 'deltapart',
             'delta']


N = np.squeeze(genparam['N'])

#Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
participant = np.array(np.squeeze(genparam['participant']),dtype=int)
condition = np.array(np.squeeze(genparam['condition']),dtype=int)
nparts = np.squeeze(genparam['nparts'])
nconds = np.squeeze(genparam['nconds'])
ntrials = np.squeeze(genparam['ntrials'])


#Fit model to data
data = {'y': y, 'N':N, 'nparts': nparts, 'nconds': nconds, 'condition': condition, 'participant': participant};

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
        'deltasd': np.random.uniform(.1, 3.),
        'terhier': np.random.uniform(.1, .5),
        'alphahier': np.random.uniform(.5, 2.),
        'betahier': np.random.uniform(.2, .8),
        'deltahier': np.random.uniform(-4., 4.),
        'ter': np.random.uniform(.1, .5, size=nparts),
        'alpha': np.random.uniform(.5, 2., size=nparts),
        'beta': np.random.uniform(.2, .8, size=nparts),
        'deltapart': np.random.uniform(-4., 4., size=nparts),
        'delta': np.random.uniform(-4., 4., size=(nparts,nconds))
    }
    for p in range(0, nparts):
        chaininit['ter'][p] = np.random.uniform(0., minrt[p]/2)
    initials.append(chaininit)


print('Fitting ''nolapse'' model in Stan...')

sm = pystan.StanModel(model_code=tostan)

fit = sm.sampling(data=data, pars=trackvars, iter=nsamps+burnin, warmup=burnin, thin=10, init=initials, chains=nchains,  n_jobs=nchains, seed=2020)
# fit = sm.sampling(data=data, pars=trackvars, iter=nsamps+burnin, warmup=burnin, thin=10, init='0', chains=nchains, n_jobs=nchains, seed=2020)
# fit = sm.sampling(data=data, pars=trackvars, iter=nsamps+burnin, warmup=burnin, thin=10, init='random', chains=nchains, n_jobs=nchains, seed=2022)

extractedsamps = fit.extract(permuted=False, pars=trackvars)

samples = phju.flipstanout(extractedsamps)

savestring = ('modelfits/genparam_test4_nolapse_stan.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)


#Diagnostics
samples = sio.loadmat(savestring)
diags = phju.diagnostic(samples)

#Posterior distributions
plt.figure()
phju.jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig(('figures/delta_posteriors_nolapse_stan.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ter'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig(('figures/ter_posteriors_nolapse_stan.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['beta'])
plt.title('Posterior distributions of the start point parameter')
plt.savefig(('figures/beta_posteriors_nolapse_stan.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig(('figures/alpha_posteriors_nolapse_stan.png'), format='png',bbox_inches="tight")

#Recovery
plt.figure()
phju.recovery(samples['delta'],genparam['delta'][:, :])
plt.title('Recovery of the drift-rate')
plt.savefig(('figures/delta_recovery_nolapse_stan.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['ter'],genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig(('figures/ter_recovery_nolapse_stan.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['beta'],genparam['beta'])
plt.title('Recovery of the start point parameter')
plt.savefig(('figures/beta_recovery_nolapse_stan.png'), format='png',bbox_inches="tight")

plt.figure()
phju.recovery(samples['alpha'],genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig(('figures/alpha_recovery_nolapse_stan.png'), format='png',bbox_inches="tight")

