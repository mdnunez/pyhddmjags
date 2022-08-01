# simple_test.py - Testing JAGS fits of a non-hierarchical DDM model without lapse process in JAGS using pyjags in Python 3
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
# 08/12/21      Michael Nunez                             Original code


# Modules
import numpy as np
import pyjags
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pyhddmjagsutils as phju

### Simulations ###

# Generate samples from the joint-model of reaction time and choice
# Note you could remove this if statement and replace with loading your own data to dictionary "gendata"

if not os.path.exists('data/simpleparam_test1.mat'):
    # Number of simulated participants
    nparts = 100

    # Number of trials for one participant
    ntrials = 100

    # Number of total trials in each simulation
    N = ntrials * nparts

    # Set random seed
    np.random.seed(2021)

    ndt = np.random.uniform(.15, .6, size=nparts)  # Uniform from .15 to .6 seconds
    alpha = np.random.uniform(.8, 1.4, size=nparts)  # Uniform from .8 to 1.4 evidence units
    beta = np.random.uniform(.3, .7, size=nparts)  # Uniform from .3 to .7 * alpha
    delta = np.random.uniform(-4, 4, size=nparts)  # Uniform from -4 to 4 evidence units per second
    deltatrialsd = np.random.uniform(0, 2, size=nparts)  # Uniform from 0 to 2 evidence units per second
    y = np.zeros(N)
    rt = np.zeros(N)
    acc = np.zeros(N)
    participant = np.zeros(N)  # Participant index
    indextrack = np.arange(ntrials)
    for p in range(nparts):
        tempout = phju.simulratcliff(N=ntrials, Alpha=alpha[p], Tau=ndt[p], Beta=beta[p],
                                     Nu=delta[p], Eta=deltatrialsd[p])
        tempx = np.sign(np.real(tempout))
        tempt = np.abs(np.real(tempout))
        y[indextrack] = tempx * tempt
        rt[indextrack] = tempt
        acc[indextrack] = (tempx + 1) / 2
        participant[indextrack] = p + 1
        indextrack += ntrials

    genparam = dict()
    genparam['ndt'] = ndt
    genparam['beta'] = beta
    genparam['alpha'] = alpha
    genparam['delta'] = delta
    genparam['deltatrialsd'] = deltatrialsd
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['participant'] = participant
    genparam['nparts'] = nparts
    genparam['ntrials'] = ntrials
    genparam['N'] = N
    sio.savemat('data/simpleparam_test1.mat', genparam)
else:
    genparam = sio.loadmat('data/simpleparam_test1.mat')

# JAGS code

# Set random seed
np.random.seed(2020)

tojags = '''
model {
    
    ##########
    #Simple DDM parameter priors
    ##########
    for (p in 1:nparts) {
    
        #Boundary parameter (speed-accuracy tradeoff) per participant
        alpha[p] ~ dnorm(1, pow(.5,-2))T(0, 3)

        #Non-decision time per participant
        ndt[p] ~ dnorm(.5, pow(.25,-2))T(0, 1)

        #Start point bias towards choice A per participant
        beta[p] ~ dnorm(.5, pow(.25,-2))T(0, 1)

        #Drift rate to choice A per participant
        delta[p] ~ dnorm(0, pow(2, -2))

    }

    ##########
    # Wiener likelihood
    for (i in 1:N) {

        # Observations of accuracy*RT for DDM process of rightward/leftward RT
        y[i] ~ dwiener(alpha[participant[i]], ndt[participant[i]], beta[participant[i]], delta[participant[i]])

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

modelfile = 'jagscode/simple_test1.jags'
f = open(modelfile, 'w')
f.write(tojags)
f.close()

# Track these variables
trackvars = ['alpha', 'ndt', 'beta', 'delta']

N = np.squeeze(genparam['N'])

# Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
participant = np.squeeze(genparam['participant'])
nparts = np.squeeze(genparam['nparts'])
ntrials = np.squeeze(genparam['ntrials'])

minrt = np.zeros(nparts)
for p in range(0, nparts):
    minrt[p] = np.min(rt[(participant == (p + 1))])

initials = []
for c in range(0, nchains):
    chaininit = {
        'alpha': np.random.uniform(.5, 2., size=nparts),
        'ndt': np.random.uniform(.1, .5, size=nparts),
        'beta': np.random.uniform(.2, .8, size=nparts),
        'delta': np.random.uniform(-4., 4., size=nparts)
    }
    for p in range(0, nparts):
        chaininit['ndt'][p] = np.random.uniform(0., minrt[p] / 2)
    initials.append(chaininit)
print('Fitting ''simple'' model ...')
threaded = pyjags.Model(file=modelfile, init=initials,
                        data=dict(y=y, N=N, nparts=nparts,
                                  participant=participant),
                        chains=nchains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(nsamps, vars=trackvars, thin=10)
savestring = ('modelfits/simple_test1_simple.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)

# Diagnostics
samples = sio.loadmat(savestring)
diags = phju.diagnostic(samples)

# Posterior distributions
plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig('figures/alpha_posteriors_simple.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ndt'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig('figures/ndt_posteriors_simple.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['beta'])
plt.title('Posterior distributions of the start point parameter')
plt.savefig('figures/beta_posteriors_simple.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig('figures/delta_posteriors_simple.png', format='png', bbox_inches="tight")


# Recovery
plt.figure()
phju.recovery(samples['alpha'], genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig('figures/alpha_recovery_simple.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['ndt'], genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig('figures/ndt_recovery_simple.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['beta'], genparam['beta'])
plt.title('Recovery of the start point parameter')
plt.savefig('figures/beta_recovery_simple.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['delta'], genparam['delta'])
plt.title('Recovery of the drift-rate')
plt.savefig('figures/delta_recovery_simple.png', format='png', bbox_inches="tight")


