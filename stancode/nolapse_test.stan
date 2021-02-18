
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
