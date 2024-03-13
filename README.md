# pyhddmjags
#### (Repository version 0.7.1)
Repository for example Hierarchical Drift Diffusion Model (HDDM) code using JAGS in Python

**Authors: Michael D. Nunez from the Psychological Methods group at the University of Amsterdam**

### Possible citation
Nunez, M. D., Fernandez, K., Srinivasan, R., & Vandekerckhove, J. (2024). [A tutorial on fitting joint models of M/EEG and behavior to understand cognition.](https://link.springer.com/article/10.3758/s13428-023-02331-x) Behavior Research Methods. https://doi.org/10.3758/s13428-023-02331-x

### Prerequisites

[Python 3 and Scientific Python libraries](https://www.anaconda.com/products/individual)

For these next install steps in Ubuntu, see jags_wiener_ubuntu.md in this repository.

[MCMC Sampling Program: JAGS](http://mcmc-jags.sourceforge.net/)

[Program: JAGS Wiener module](https://sourceforge.net/projects/jags-wiener/)

[Python Repository: pyjags](https://github.com/michaelnowotny/pyjags), can use pip:
```bash
pip install pyjags
```
See also yaml/pyjags.yml

#### Optional Stan + PyStan

The newest version of PyStan was PyStan 3, but I didn't find PyStan 3 as easy to use with custom diagnostic and plotting scripts as PyStan 2.

[Here are the docs for PyStan 2](https://pystan2.readthedocs.io)

See also yaml/pystan.yml


### Downloading

The repository can be cloned with `git clone https://github.com/mdnunez/pyhddmjags.git`

The repository can also be downloaded via the Code -> _Download zip_ buttons above on this Github page.

### Getting started

At the moment each script can be run individually to simulate from hierarchical drift-diffusion models (HDDMs) and then find and recover parameter estimates from those models. The most simple HDDM lives in nolapse_test.py. See other scripts: recovery_test.py, blocked_exp_conds.py, and regression_test.py. These scripts provide useful examples for using JAGS with pyjags, the JAGS Wiener module, mixture modeling in JAGS, and Bayesian diagnostics in Python. 

The script nolapse_test_pystan.py contains pystan and Stan code to find and recover parameters from the exact same HDDM written in JAGS within nolapse_test.py. Note that this script will not work with the newest versions of pystan (pystan 3). One solution is to create a Python environment using `conda`:
```bash
conda create -n mcmc python=3 numpy scipy pystan==2.19
conda activate mcmc
```

Note that you may need or want to create a separate environment for pyjags using `conda`:
```bash
conda create -n jags python=3.8 numpy
conda activate jags
pip install pyjags
```

### License

pyhddmjags is licensed under the GNU General Public License v3.0 and written by Michael D. Nunez from the Psychological Methods group at the University of Amsterdam.

### Usage examples

Nunez, M. D., Gosai, A., Vandekerckhove, J., & Srinivasan, R. (2019).
[The latency of a visual evoked potential tracks the onset of decision making.](https://sci-hub.st/https://www.sciencedirect.com/science/article/pii/S1053811919303386) NeuroImage. doi: 10.1016/j.neuroimage.2019.04.052
(See repository [encodingN200](https://github.com/mdnunez/encodingN200) associated with this paper for specific pyjags examples.)

Lui, K. K., Nunez, M. D., Cassidy, J. M., Vandekerckhove, J., Cramer, S. C., & Srinivasan, R. (2021).
[Timing of readiness potentials reflect a decision-making process in the human brain.](https://sci-hub.st/https://link.springer.com/article/10.1007/s42113-020-00097-5) Computational Brain & Behavior.
(See repository [RPDecision](https://github.com/mdnunez/RPDecision) associated with this paper for a specific HDDM in JAGS using MATLAB.)

Nunez, M. D., Vandekerckhove, J., & Srinivasan, R. (2017).
[How attention influences perceptual decision making: Single-trial EEG correlates of drift-diffusion model parameters.](https://sci-hub.st/https://www.sciencedirect.com/science/article/abs/pii/S0022249616000316)
Journal of Mathematical Psychology, 76, 117-130.
(See repository [mcntoolbox](https://github.com/mdnunez/mcntoolbox) associated with this paper for a specific HDDM in JAGS using MATLAB.)

Nunez, M. D., Srinivasan, R., & Vandekerckhove, J. (2015). 
[Individual differences in attention influence perceptual decision making.](https://www.frontiersin.org/articles/10.3389/fpsyg.2015.00018/full) 
Frontiers in Psychology, 8.
(This paper fit HDDMs in JAGS like that in regression_test_py3.py using MATLAB and [Trinity](https://github.com/joachimvandekerckhove/trinity).)
