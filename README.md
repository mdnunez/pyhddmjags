# pyhddmjags
#### (Repository version 0.2.10)
Repository for example Hierarchical Drift Diffusion Model (HDDM) code using JAGS in Python

**Authors: Michael D. Nunez from the Cognitive Sciences Department at the University of California, Irvine**

### Prerequisites

[Python 3 and Scientific Python libraries](https://www.anaconda.com/products/individual)

For these next install steps in Ubuntu, see jags_wiener_ubuntu.md in this repository.

[MCMC Sampling Program: JAGS](http://mcmc-jags.sourceforge.net/)

[Program: JAGS Wiener module](https://sourceforge.net/projects/jags-wiener/)

[Python Repository: pyjags](https://github.com/michaelnowotny/pyjags), can use pip:
```bash
pip install pyjags
```


### Downloading

The repository can be cloned with `git clone https://github.com/mdnunez/pyhddmjags.git`

The repository can also be may download via the _Download zip_ button above.

### Getting started

At the moment each script can be run individually to simulate from hierarchical drift-diffusion models and then find (and recover) parameter estimates from those models. See scripts blocked_exp_conds_py3.py, recovery_test_py3.py, and regression_test_py3.py. These scripts also provide useful examples for using JAGS with pyjags, the JAGS Wiener module, mixture modeling in JAGS, and Bayesian diagnostics in Python.

In the future this package will be more user-friendly.

### License

pyhddmjags is licensed under the GNU General Public License v3.0 and written by Michael D. Nunez from the Cognitive Sciences Department at the University of California, Irvine.

### Usage examples

Nunez, M. D., Gosai, A., Vandekerckhove, J., & Srinivasan, R. (2019).
[The latency of a visual evoked potential tracks the onset of decision making.](https://sci-hub.st/https://www.sciencedirect.com/science/article/pii/S1053811919303386) NeuroImage. doi: 10.1016/j.neuroimage.2019.04.052

Nunez, M. D., Vandekerckhove, J., & Srinivasan, R. (2017).
[How attention influences perceptual decision making: Single-trial EEG correlates of drift-diffusion model parameters.](https://sci-hub.st/https://www.sciencedirect.com/science/article/abs/pii/S0022249616000316)
Journal of Mathematical Psychology, 76, 117-130.

Nunez, M. D., Srinivasan, R., & Vandekerckhove, J. (2015). 
[Individual differences in attention influence perceptual decision making.](https://www.frontiersin.org/articles/10.3389/fpsyg.2015.00018/full) 
Frontiers in Psychology, 8.