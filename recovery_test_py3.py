# recovery_test_py3.py - Testing JAGS fits of HDDM models in JAGS using pyjags in Python 3
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

# Simulate diffusion models
def simulratcliff(N=100,Alpha=1,Tau=.4,Nu=1,Beta=.5,rangeTau=0,rangeBeta=0,Eta=.3,Varsigma=1):
    """
    SIMULRATCLIFF  Generates data according to a drift diffusion model with optional trial-to-trial variability


    Reference:
    Tuerlinckx, F., Maris, E.,
    Ratcliff, R., & De Boeck, P. (2001). A comparison of four methods for
    simulating the diffusion process. Behavior Research Methods,
    Instruments, & Computers, 33, 443-456.

    Parameters
    ----------
    N: a integer denoting the size of the output vector
    (defaults to 100 experimental trials)

    Alpha: the mean boundary separation across trials  in evidence units
    (defaults to 1 evidence unit)

    Tau: the mean non-decision time across trials in seconds
    (defaults to .4 seconds)

    Nu: the mean drift rate across trials in evidence units per second
    (defaults to 1 evidence units per second, restricted to -5 to 5 units)

    Beta: the initial bias in the evidence process for choice A as a proportion of boundary Alpha
    (defaults to .5 or 50% of total evidence units given by Alpha)

    rangeTau: Non-decision time across trials is generated from a uniform
    distribution of Tau - rangeTau/2 to  Tau + rangeTau/2 across trials
    (defaults to 0 seconds)

    rangeZeta: Bias across trials is generated from a uniform distribution
    of Zeta - rangeZeta/2 to Zeta + rangeZeta/2 across trials
    (defaults to 0 evidence units)

    Eta: Standard deviation of the drift rate across trials
    (defaults to 3 evidence units per second, restricted to less than 3 evidence units)

    Varsigma: The diffusion coefficient, the standard deviation of the
    evidence accumulation process within one trial. It is recommended that
    this parameter be kept fixed unless you have reason to explore this parameter
    (defaults to 1 evidence unit per second)

    Returns
    -------
    Numpy array with reaction times (in seconds) multiplied by the response vector
    such that negative reaction times encode response B and positive reaction times
    encode response A 
    
    
    Converted from simuldiff.m MATLAB script by Joachim Vandekerckhove
    See also http://ppw.kuleuven.be/okp/dmatoolbox.
    """

    if (Nu < -5) or (Nu > 5):
        Nu = np.sign(Nu)*5
        warnings.warn('Nu is not in the range [-5 5], bounding drift rate to %.1f...' % (Nu))

    if (Eta > 3):
        warning.warn('Standard deviation of drift rate is out of bounds, bounding drift rate to 3')
        eta = 3

    if (Eta == 0):
        Eta = 1e-16

    #Initialize output vectors
    result = np.zeros(N)
    T = np.zeros(N)
    XX = np.zeros(N)
    N200 = np.zeros(N)

    #Called sigma in 2001 paper
    D = np.power(Varsigma,2)/2

    #Program specifications
    eps = 2.220446049250313e-16 #precision from 1.0 to next double-precision number
    delta=eps

    for n in range(0,N):
        r1 = np.random.normal()
        mu = Nu + r1*Eta
        bb = Beta - rangeBeta/2 + rangeBeta*np.random.uniform()
        zz = bb*Alpha
        finish = 0
        totaltime = 0
        startpos = 0
        Aupper = Alpha - zz
        Alower = -zz
        radius = np.min(np.array([np.abs(Aupper), np.abs(Alower)]))
        while (finish==0):
            lambda_ = 0.25*np.power(mu,2)/D + 0.25*D*np.power(np.pi,2)/np.power(radius,2)
            # eq. formula (13) in 2001 paper with D = sigma^2/2 and radius = Alpha/2
            F = D*np.pi/(radius*mu)
            F = np.power(F,2)/(1 + np.power(F,2) )
            # formula p447 in 2001 paper
            prob = np.exp(radius*mu/D)
            prob = prob/(1 + prob)
            dir_ = 2*(np.random.uniform() < prob) - 1
            l = -1
            s2 = 0
            while (s2>l):
                s2=np.random.uniform()
                s1=np.random.uniform()
                tnew=0
                told=0
                uu=0
                while (np.abs(tnew-told)>eps) or (uu==0):
                    told=tnew
                    uu=uu+1
                    tnew = told + (2*uu+1) * np.power(-1,uu) * np.power(s1,(F*np.power(2*uu+1,2)));
                    # infinite sum in formula (16) in BRMIC,2001
                l = 1 + np.power(s1,(-F)) * tnew;
            # rest of formula (16)
            t = np.abs(np.log(s1))/lambda_;
            # is the negative of t* in (14) in BRMIC,2001
            totaltime=totaltime+t
            dir_=startpos+dir_*radius
            ndt = Tau - rangeTau/2 + rangeTau*np.random.uniform()
            if ( (dir_ + delta) > Aupper):
                T[n]=ndt+totaltime
                XX[n]=1
                finish=1
            elif ( (dir_-delta) < Alower ):
                T[n]=ndt+totaltime
                XX[n]=-1
                finish=1
            else:
                startpos=dir_
                radius=np.min(np.abs([Aupper, Alower]-startpos))

    result = T*XX
    return result


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
        Rhat, Rhatnew, Neff, posterior mean, and posterior std for each variable. Prints maximum Rhat, maximum Rhatnew, and minimum Neff across all variables
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


def summary(insamples):
    """
    Returns parameter estimates for each posterior distribution (mean and median posteriors) as well as 95% and 99% credible intervals (.5th, 2.5th, 97.5th, 99.5th percentiles)

    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.
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

            result[key]['mean'] = np.mean(allsamps, axis=-1)
            result[key]['std'] = np.std(allsamps, axis=-1)
            result[key]['median'] = np.quantile(allsamps,0.5, axis=-1)
            result[key]['95lower'] = np.quantile(allsamps,0.025, axis=-1)
            result[key]['95upper'] = np.quantile(allsamps,0.975, axis=-1)
            result[key]['99lower'] = np.quantile(allsamps,0.005, axis=-1)
            result[key]['99upper'] = np.quantile(allsamps,0.995, axis=-1)
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


def recovery(possamps, truevals):  # Parameter recovery plots
    """Plots true parameters versus 99% and 95% credible intervals of recovered
    parameters. Also plotted are the median and mean of the posterior
    distributions

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
            lines = plt.plot(credint, y)
            plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark median
                mmedian = plt.plot(truevals[v], np.median(alldata[v, :]), 'o')
                plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = plt.plot(truevals[v], np.mean(alldata[v, :]), '*')
                plt.setp(mmean, markersize=10, color=teal)
    # Plot line y = x
    tempx = np.linspace(np.min(truevals), np.max(
        truevals), num=100)
    recoverline = plt.plot(tempx, tempx)
    plt.setp(recoverline, linewidth=3, color=orange)



### Simulations ###

# Generate samples from the joint-model of reaction time and choice

# if not os.path.exists('data/genparam_test1.mat'):

# Number of simulated participants
nparts = 40

# Number of conditions
nconds = 6

# Number of trials per participant and condition
ntrials = 50

# Number of total trials in each simulation
N = ntrials*nparts*nconds

# Set random seed
random.seed(2020)

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
        tempout = simulratcliff(N=ntrials, Alpha= alpha[p], Tau= ndt[p], Beta=beta[p], 
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
# else:
#     genparam = sio.loadmat('data/genparam_test1.mat')

# JAGS code

# Set random seed
random.seed(2020)

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
diags = diagnostic(samples_diagrelevant)

#Posterior distributions
plt.figure()
jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig(('figures/delta_posteriors_model1.png'), format='png',bbox_inches="tight")

plt.figure()
jellyfish(samples['ter'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig(('figures/ter_posteriors_model1.png'), format='png',bbox_inches="tight")

plt.figure()
jellyfish(samples['beta'])
plt.title('Posterior distributions of the start point parameter')
plt.savefig(('figures/beta_posteriors_model1.png'), format='png',bbox_inches="tight")

plt.figure()
jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig(('figures/alpha_posteriors_model1.png'), format='png',bbox_inches="tight")

#Recovery
plt.figure()
recovery(samples['delta'],genparam['delta'][:, :])
plt.title('Recovery of the drift-rate')
plt.savefig(('figures/delta_recovery_model1.png'), format='png',bbox_inches="tight")

plt.figure()
recovery(samples['ter'],genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig(('figures/ter_recovery_model1.png'), format='png',bbox_inches="tight")

plt.figure()
recovery(samples['beta'],genparam['beta'])
plt.title('Recovery of the start point parameter')
plt.savefig(('figures/beta_recovery_model1.png'), format='png',bbox_inches="tight")

plt.figure()
recovery(samples['alpha'],genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig(('figures/alpha_recovery_model1.png'), format='png',bbox_inches="tight")

