# simpleCPP_test.py - Testing JAGS fits of a non-hierarchical Neural-DDM model
# in JAGS using pyjags in Python 3,
# assumes CPP slopes are generated from drift rates
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
# 08/12/21      Michael Nunez                             Original code
# 2021-09-08    Michael Nunez           Plot recovery figure for tutorial paper


# Internet resources:
# https://stackoverflow.com/questions/18939484/matplotlib-subplot-that-takes-the-space-of-two-plots
# https://stackoverflow.com/questions/34799488/how-to-manually-position-one-subplot-graph-in-matplotlib-pyplot

# Modules
import numpy as np
import pyjags
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from scipy import stats
import pyhddmjagsutils as phju

### Simulations ###

# Generate samples from the joint-model of reaction time and choice
# Note you could remove this if statement and replace with loading your own data to dictionary "gendata"

if not os.path.exists('data/simpleEEG_test1.mat'):
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
    CPPnoise = np.random.uniform(0, 1, size=nparts) # Uniform from 0 to 1 evidence units per second
    y = np.zeros(N)
    rt = np.zeros(N)
    acc = np.zeros(N)
    CPP = np.zeros(N)
    participant = np.zeros(N)  # Participant index
    indextrack = np.arange(ntrials)
    for p in range(nparts):
        tempout = phju.simulratcliff(N=ntrials, Alpha=alpha[p], Tau=ndt[p], Beta=beta[p],
                                     Nu=delta[p], Eta=deltatrialsd[p])
        tempx = np.sign(np.real(tempout))
        tempt = np.abs(np.real(tempout))
        CPP[indextrack] = np.random.normal(loc=delta[p],scale=CPPnoise[p],size=ntrials)
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
    genparam['CPPnoise'] = CPPnoise
    genparam['CPP'] = CPP
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['participant'] = participant
    genparam['nparts'] = nparts
    genparam['ntrials'] = ntrials
    genparam['N'] = N
    sio.savemat('data/simpleEEG_test1.mat', genparam)
else:
    genparam = sio.loadmat('data/simpleEEG_test1.mat')

# JAGS code

# Set random seed
np.random.seed(2020)

tojags = '''
model {
    
    ##########
    #Simple NDDM parameter priors
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

        #Noise in observed EEG measure, the CentroParietal Positivity (CPP) slope per participant
        CPPnoise[p] ~ dnorm(1, pow(.5,-2))T(0, 3)

    }

    ##########
    # Wiener likelihood
    for (i in 1:N) {

        # Observations of accuracy*RT for DDM process of rightward/leftward RT
        y[i] ~ dwiener(alpha[participant[i]], ndt[participant[i]], beta[participant[i]], delta[participant[i]])

        # Observations of CentroParietal Positivity (CPP) slope per trial
        CPP[i] ~ dnorm(delta[participant[i]],pow(CPPnoise[participant[i]],-2))

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

modelfile = 'jagscode/simpleCPP_test1.jags'
f = open(modelfile, 'w')
f.write(tojags)
f.close()

# Track these variables
trackvars = ['alpha', 'ndt', 'beta', 'delta', 'CPPnoise']

N = np.squeeze(genparam['N'])

# Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
CPP = np.squeeze(genparam['CPP'])
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
        'delta': np.random.uniform(-4., 4., size=nparts),
        'CPPnoise': np.random.uniform(.5, 2., size=nparts)
    }
    for p in range(0, nparts):
        chaininit['ndt'][p] = np.random.uniform(0., minrt[p] / 2)
    initials.append(chaininit)
print('Fitting ''simpleEEG'' model ...')
threaded = pyjags.Model(file=modelfile, init=initials,
                        data=dict(y=y, CPP=CPP, N=N, nparts=nparts,
                                  participant=participant),
                        chains=nchains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(nsamps, vars=trackvars, thin=10)
savestring = ('modelfits/simpleEEG_test1_simpleCPP.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)

# Diagnostics
samples = sio.loadmat(savestring)
diags = phju.diagnostic(samples)

# Posterior distributions
plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig('figures/alpha_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ndt'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig('figures/ndt_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['beta'])
plt.title('Posterior distributions of the start point parameter')
plt.savefig('figures/beta_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig('figures/delta_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['CPPnoise'])
plt.title('Posterior distributions of the noise in the observed CPP slope')
plt.savefig('figures/CPPnoise_posteriors_simpleCPP.png', format='png', bbox_inches="tight")



# Recovery
plt.figure()
phju.recovery(samples['alpha'], genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig('figures/alpha_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['ndt'], genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig('figures/ndt_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['beta'], genparam['beta'])
plt.title('Recovery of the start point parameter')
plt.savefig('figures/beta_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['delta'], genparam['delta'])
plt.title('Recovery of the drift-rate')
plt.savefig('figures/delta_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['CPPnoise'], genparam['CPPnoise'])
plt.title('Recovery of the noise in the observed CPP slope')
plt.savefig('figures/CPPnoise_recovery_simpleCPP.png', format='png', bbox_inches="tight")


# Recovery plots nicely formatting for tutorial
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True) #sudo apt install texlive-latex-extra cm-super dvipng


def recoverysub(possamps, truevals, ax):  # Parameter recovery subplots
    """Plots true parameters versus 99% and 95% credible intervals of recovered
    parameters. Also plotted are the median (circles) and mean (stars) of the posterior
    distributions.

    Parameters
    ----------
    possamps : ndarray of posterior chains where the last dimension is the
    number of chains, the second to last dimension is the number of samples in
    each chain, all other dimensions must match the dimensions of truevals

    truevals : ndarray of true parameter values
    """

    # Number of chains
    nchains = possamps.shape[-1]

    # Number of samples per chain
    nsamps = possamps.shape[-2]

    # Number of variables to plot
    nvars = np.prod(possamps.shape[0:-2])

    # Reshape data
    alldata = np.reshape(possamps, (nvars, nchains, nsamps))
    alldata = np.reshape(alldata, (nvars, nchains * nsamps))
    truevals = np.reshape(truevals, (nvars))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    for v in range(0, nvars):
        # Compute percentiles
        bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
        for b in range(0, 2):
            # Plot credible intervals
            credint = np.ones(100) * truevals[v]
            y = np.linspace(bounds[b], bounds[-1 - b], 100)
            lines = ax.plot(credint, y, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark median
                mmedian = ax.plot(truevals[v], np.median(alldata[v, :]), 'o', markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = ax.plot(truevals[v], np.mean(alldata[v, :]), '*', markersize=10, color=teal)
    # Plot line y = x
    tempx = np.linspace(np.min(truevals), np.max(
        truevals), num=100)
    recoverline = ax.plot(tempx, tempx, linewidth=3, color=orange)



fontsize = 12

fig = plt.figure(figsize=(9,10),dpi=300)
gs = gridspec.GridSpec(3, 2)

ax1 = plt.subplot(gs[0, 0:2])
recoverysub(samples['delta'], genparam['delta'],ax1)
ax1.set_xlabel('Simulated $\\delta_{p}$ ($\\mu V$ / sec)', fontsize=fontsize)
ax1.set_ylabel('Posterior of $\\delta_{p}$ ($\\mu V$ / sec)', fontsize=fontsize)

ax2 = plt.subplot(gs[1, 0])
recoverysub(samples['ndt'], genparam['ndt'],ax2)
ax2.set_xlabel('Simulated $\\tau_{p}$ (secs)', fontsize=fontsize)
ax2.set_ylabel('Posterior of $\\tau_{p}$ (secs)', fontsize=fontsize)

ax3 = plt.subplot(gs[1, 1])
recoverysub(samples['alpha'], genparam['alpha'],ax3)
ax3.set_xlabel('Simulated $\\alpha_{p}$ ($\\mu V$)', fontsize=fontsize)
ax3.set_ylabel('Posterior of $\\alpha_{p}$ ($\\mu V$)', fontsize=fontsize)

ax4 = plt.subplot(gs[2, 0])
recoverysub(samples['beta'], genparam['beta'],ax4)
ax4.set_xlabel('Simulated $\\beta_{p}$ ($\\mu V$)', fontsize=fontsize)
ax4.set_ylabel('Posterior of $\\beta_{p}$ ($\\mu V$)', fontsize=fontsize)

ax5 = plt.subplot(gs[2, 1])
recoverysub(samples['CPPnoise'], genparam['CPPnoise'],ax5)
ax5.set_xlabel('Simulated $\\sigma_{p}$ ($\\mu V$)', fontsize=fontsize)
ax5.set_ylabel('Posterior of $\\sigma_{p}$ ($\\mu V$)', fontsize=fontsize)

plt.savefig('figures/All_recovery_simpleCPP.png', dpi=300, format='png', bbox_inches="tight")