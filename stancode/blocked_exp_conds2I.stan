
functions { 
  /* Mixture of Wiener diffusion log-PDF and uniform distribution log-PDF for a single response (adapted from brms 1.10.2)
   * Arguments: 
   *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
   *   boundary: boundary separation parameter > 0
   *   ndt: non-decision time parameter > 0
   *   bias: initial bias parameter in [0, 1]
   *   drift: drift rate parameter
   *   theta: probability of lapse trial
   *   seccutoff: larger parameter in seconds for uniform lapse process
   * Returns:  
   *   a scalar to be added to the log posterior 
   */ 
   real diffusionmix_lpdf(real Y, real boundary, 
                              real ndt, real bias, real drift, real theta, real seccutoff) { 
     
    if (fabs(Y) < ndt) {
        return uniform_lpdf(fabs(Y) | 0 , seccutoff); // This trial likely came from a lapse process
    } else {
        if (Y >= 0) {
            return wiener_lpdf( fabs(Y) | boundary, ndt, bias, drift );
        } else {
            return wiener_lpdf( fabs(Y) | boundary, ndt, 1-bias, -drift );
        }
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
    real<lower=0,upper=3> deltasdcond; // Between-condition variability in drift rate to correct
    real<lower=0,upper=1> tersdcond; // Between-condition variability in non-decision time
    real<lower=0,upper=1> alphasdcond; // Between-condition variability in speed-accuracy trade-off
    real<lower=0,upper=3> deltasd; // Between-participant variability in drift rate to correct
    real<lower=0,upper=1> tersd; // Between-participant variability in non-decision time 
    real<lower=0,upper=1> alphasd; // Between-participant variability in Speed-accuracy trade-off
    real<lower=0,upper=.5> problapsesd; // Between-participant variability in lapse trial probability
    real deltahier; // Hierarchical drift rate to correct
    real terhier; // Hierarchical Non-decision time
    real alphahier; // Hierarchical boundary parameter (speed-accuracy tradeoff)
    real problapsehier; // Hierarchical lapse trial probability
    vector[nparts] deltapart; // Participant-level drift rate to correct
    vector[nparts] terpart; // Participant-level non-decision time
    vector[nparts] alphapart; // Participant-level boundary parameter (speed-accuracy tradeoff)
    vector<lower=0, upper=1>[nparts] problapse; // Probability of a lapse trial
    matrix<lower=-4, upper=4>[nparts,nconds] delta; // Participant-level drift rate to correct
    matrix<lower=0, upper=1>[nparts,nconds] ter; // Non-decision time
    matrix<lower=0, upper=3>[nparts,nconds] alpha; // Boundary parameter (speed-accuracy tradeoff)
}
transformed parameters {
    vector<lower=0, upper=1>[nparts] probDDM;

    for (p in 1:nparts) {

        // Probability of a DDM trial
        probDDM[p] = 1 - problapse[p];

    }
}
model {
    
    // ##########
    // Between-condition variability priors
    // ##########

    // Between-condition variability in drift rate to correct
    deltasdcond ~ gamma(1,1);

    // Between-condition variability in non-decision time
    tersdcond ~ gamma(.3,1);

    // Between-condition variability in speed-accuracy trade-off
    alphasdcond ~ gamma(1,1);

    // ##########
    // Between-participant variability priors
    // ##########

    // Between-participant variability in drift rate to correct
    deltasd ~ gamma(1,1);

    // Between-participant variability in non-decision time
    tersd ~ gamma(.3,1);

    // Between-participant variability in Speed-accuracy trade-off
    alphasd ~ gamma(1,1);

    // Between-participant variability in lapse trial probability
    problapsesd ~ gamma(.3,1);

    // ##########
    // Hierarchical DDM parameter priors
    // ##########

    // Hierarchical drift rate to correct
    deltahier ~ normal(0, 2);

    // Hierarchical Non-decision time
    terhier ~ normal(.5,.25);

    // Hierarchical boundary parameter (speed-accuracy tradeoff)
    alphahier ~ normal(1, .5);

    // Hierarchical lapse trial probability
    problapsehier ~ normal(.3, .15);

    // ##########
    // Participant-level DDM parameter priors
    // ##########
    for (p in 1:nparts) {

        // Participant-level drift rate to correct
        deltapart[p] ~ normal(deltahier, deltasd);

        // Participant-level non-decision time
        terpart[p] ~ normal(terhier, tersd);

        // Participant-level boundary parameter (speed-accuracy tradeoff)
        alphapart[p] ~ normal(alphahier, alphasd);

        // Probability of a lapse trial
        problapse[p] ~ normal(problapsehier, problapsesd);

        // ##########
        // Condition-level DDM parameter priors
        // ##########
        for (c in 1:nconds) {

            // Drift rate to correct
            delta[p,c] ~ normal(deltapart[p], deltasdcond) T[-4, 4];

            // Non-decision time
            ter[p,c] ~ normal(terpart[p], tersdcond) T[0, 1];

            // Boundary parameter (speed-accuracy tradeoff)
            alpha[p,c] ~ normal(alphapart[p], alphasdcond) T[0, 3];

        }

    }
    // Wiener likelihood and uniform mixture
    for (i in 1:N) {

        target += diffusionmix_lpdf( y[i] | alpha[participant[i],condition[i]], ter[participant[i],condition[i]], .5, delta[participant[i],condition[i]], problapse[participant[i]], 3 );
    }
}
