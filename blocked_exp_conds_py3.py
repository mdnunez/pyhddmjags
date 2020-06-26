# blocked_exp_conds_py3.py - Code to fit a HDDM with fixed start point from a blocked condition experiment using JAGS in Python 3
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
# 06/24/20      Michael Nunez                             Original code
# 06/25/20      Michael NUnez             Removal of false plots, reduce problapse parameters


# Modules
import numpy as np
import pyjags
import scipy.io as sio
from scipy import stats
import warnings
import random
import os
import matplotlib.pyplot as plt

### Definitions ###
def diagnostic(insamples):
    """
    Returns Rhat (measure of convergence, less is better with an approximate
    1.10 cutoff) and Neff, number of effective samples).

    Reference: Gelman, A., Carlin, J., Stern, H., & Rubin D., (2004).
              Bayesian Data Analysis (Second Edition). Chapman & Hall/CRC:
              Boca Raton, FL.


    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.

    Returns
    -------
    dict:
        Rhat for each variable. Prints Maximum Rhat
    """

    result = {}  # Initialize dictionary
    maxrhats = np.zeros((len(insamples.keys())), dtype=float)
    maxrhatsnew = np.zeros((len(insamples.keys())), dtype=float)
    minneff = np.ones((len(insamples.keys())), dtype=float)*np.inf
    allkeys ={} # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}
            
            possamps = insamples[key]
            
            # Number of chains
            nchains = possamps.shape[-1]
            
            # Number of samples per chain
            nsamps = possamps.shape[-2]
            
            # Number of variables per key
            nvars = np.prod(possamps.shape[0:-2])
            
            # Reshape data
            allsamps = np.reshape(possamps, possamps.shape[:-2] + (nchains * nsamps,))

            # Reshape data to preduce R_hatnew
            possampsnew = np.empty(possamps.shape[:-2] + (int(nsamps/2), nchains * 2,))
            newc=0
            for c in range(nchains):
                possampsnew[...,newc] = np.take(np.take(possamps,np.arange(0,int(nsamps/2)),axis=-2),c,axis=-1)
                possampsnew[...,newc+1] = np.take(np.take(possamps,np.arange(int(nsamps/2),nsamps),axis=-2),c,axis=-1)
                newc += 2

            # Index of variables
            varindx = np.arange(nvars).reshape(possamps.shape[0:-2])
            
            # Reshape data
            alldata = np.reshape(possamps, (nvars, nsamps, nchains))
                    
            # Mean of each chain for rhat
            chainmeans = np.mean(possamps, axis=-2)
            # Mean of each chain for rhatnew
            chainmeansnew = np.mean(possampsnew, axis=-2)
            # Global mean of each parameter for rhat
            globalmean = np.mean(chainmeans, axis=-1)
            globalmeannew = np.mean(chainmeansnew, axis=-1)
            result[key]['mean'] = globalmean
            result[key]['std'] = np.std(allsamps, axis=-1)
            globalmeanext = np.expand_dims(
                globalmean, axis=-1)  # Expand the last dimension
            globalmeanext = np.repeat(
                globalmeanext, nchains, axis=-1)  # For differencing
            globalmeanextnew = np.expand_dims(
                globalmeannew, axis=-1)  # Expand the last dimension
            globalmeanextnew = np.repeat(
                globalmeanextnew, nchains*2, axis=-1)  # For differencing
            # Between-chain variance for rhat
            between = np.sum(np.square(chainmeans - globalmeanext),
                             axis=-1) * nsamps / (nchains - 1.)
            # Mean of the variances of each chain for rhat
            within = np.mean(np.var(possamps, axis=-2), axis=-1)
            # Total estimated variance for rhat
            totalestvar = (1. - (1. / nsamps)) * \
                within + (1. / nsamps) * between
            # Rhat (original Gelman-Rubin statistic)
            temprhat = np.sqrt(totalestvar / within)
            maxrhats[keyindx] = np.nanmax(temprhat) # Ignore NANs
            allkeys[keyindx] = key
            result[key]['rhat'] = temprhat
            # Between-chain variance for rhatnew
            betweennew = np.sum(np.square(chainmeansnew - globalmeanextnew),
                             axis=-1) * (nsamps/2) / ((nchains*2) - 1.)
            # Mean of the variances of each chain for rhatnew
            withinnew = np.mean(np.var(possampsnew, axis=-2), axis=-1)
            # Total estimated variance
            totalestvarnew = (1. - (1. / (nsamps/2))) * \
                withinnew + (1. / (nsamps/2)) * betweennew
            # Rhatnew (Gelman-Rubin statistic from Gelman et al., 2013)
            temprhatnew = np.sqrt(totalestvarnew / withinnew)
            maxrhatsnew[keyindx] = np.nanmax(temprhatnew) # Ignore NANs
            result[key]['rhatnew'] = temprhatnew
            # Number of effective samples from Gelman et al. (2013) 286-288
            neff = np.empty(possamps.shape[0:-2])
            for v in range(0, nvars):
                whereis = np.where(varindx == v)
                rho_hat = []
                rho_hat_even = 0
                rho_hat_odd = 0
                t = 2
                while ((t < nsamps - 2) & (float(rho_hat_even) + float(rho_hat_odd) >= 0)):
                    variogram_odd = np.mean(np.mean(np.power(alldata[v,(t-1):nsamps,:] - alldata[v,0:(nsamps-t+1),:],2),axis=0)) # above equation (11.7) in Gelman et al., 2013
                    rho_hat_odd = 1 - np.divide(variogram_odd, 2*totalestvar[whereis]) # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_odd)
                    variogram_even = np.mean(np.mean(np.power(alldata[v,t:nsamps,:] - alldata[v,0:(nsamps-t),:],2),axis=0)) # above equation (11.7) in Gelman et al., 2013
                    rho_hat_even = 1 - np.divide(variogram_even, 2*totalestvar[whereis]) # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_even)
                    t += 2
                rho_hat = np.asarray(rho_hat)
                neff[whereis] = np.divide(nchains*nsamps, 1 + 2*np.sum(rho_hat)) # Equation (11.8) in Gelman et al., 2013
            result[key]['neff'] = np.round(neff) 
            minneff[keyindx] = np.nanmin(np.round(neff))
            keyindx += 1

            # Geweke statistic?
    print("Maximum Rhat was %3.2f for variable %s" % (np.max(maxrhats),allkeys[np.argmax(maxrhats)]))
    print("Maximum Rhatnew was %3.2f for variable %s" % (np.max(maxrhatsnew),allkeys[np.argmax(maxrhatsnew)]))
    print("Minimum number of effective samples was %d for variable %s" % (np.min(minneff),allkeys[np.argmin(minneff)]))
    return result


def jellyfish(possamps):  # jellyfish plots
    """Plots posterior distributions of given posterior samples in a jellyfish
    plot. Jellyfish plots are posterior distributions (mirrored over their
    horizontal axes) with 99% and 95% credible intervals (currently plotted
    from the .5% and 99.5% & 2.5% and 97.5% percentiles respectively.
    Also plotted are the median and mean of the posterior distributions"

    Parameters
    ----------
    possamps : ndarray of posterior chains where the last dimension is
    the number of chains, the second to last dimension is the number of samples
    in each chain, all other dimensions describe the shape of the parameter
    """

    # Number of chains
    nchains = possamps.shape[-1]

    # Number of samples per chain
    nsamps = possamps.shape[-2]

    # Number of dimensions
    ndims = possamps.ndim - 2

    # Number of variables to plot
    nvars = np.prod(possamps.shape[0:-2])

    # Index of variables
    varindx = np.arange(nvars).reshape(possamps.shape[0:-2])

    # Reshape data
    alldata = np.reshape(possamps, (nvars, nchains, nsamps))
    alldata = np.reshape(alldata, (nvars, nchains * nsamps))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    # Initialize ylabels list
    ylabels = ['']

    for v in range(0, nvars):
        # Create ylabel
        whereis = np.where(varindx == v)
        newlabel = ''
        for l in range(0, ndims):
            newlabel = newlabel + ('_%i' % whereis[l][0])

        ylabels.append(newlabel)

        # Compute posterior density curves
        kde = stats.gaussian_kde(alldata[v, :])
        bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
        for b in range(0, 2):
            # Bound by .5th percentile and 99.5th percentile
            x = np.linspace(bounds[b], bounds[-1 - b], 100)
            p = kde(x)

            # Scale distributions down
            maxp = np.max(p)

            # Plot jellyfish
            upper = .25 * p / maxp + v + 1
            lower = -.25 * p / maxp + v + 1
            lines = plt.plot(x, upper, x, lower)
            plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark mode
                wheremaxp = np.argmax(p)
                mmode = plt.plot(np.array([1., 1.]) * x[wheremaxp],
                                 np.array([lower[wheremaxp], upper[wheremaxp]]))
                plt.setp(mmode, linewidth=3, color=orange)
                # Mark median
                mmedian = plt.plot(np.median(alldata[v, :]), v + 1, 'ko')
                plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = plt.plot(np.mean(alldata[v, :]), v + 1, '*')
                plt.setp(mmean, markersize=10, color=teal)

    # Display plot
    plt.setp(plt.gca(), yticklabels=ylabels, yticks=np.arange(0, nvars + 1))


# Load and extract data

indata = sio.loadmat('data/genparam_test1.mat') # Change this to your data location

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
random.seed(2020)

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

    #Hierarchical drift rate to choice A
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

        for (c in 1:nconds) {

            #Participant-level drift rate to correct
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

modelfile = 'jagscode/recovery_test1.jags'
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
diags = diagnostic(samples_diagrelevant)

#Posterior distributions
plt.figure()
jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig(('figures/delta_posteriors_model2.png'), format='png',bbox_inches="tight")

plt.figure()
jellyfish(samples['ter'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig(('figures/ter_posteriors_model2.png'), format='png',bbox_inches="tight")

plt.figure()
jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig(('figures/alpha_posteriors_model2.png'), format='png',bbox_inches="tight")