data {
    int<lower=1> N;                         // number of nodes
    int<lower=1> E;                         // number of edges
    int<lower=0,upper=1> A[E];              // adjacency matrix
    int<lower=1,upper=N> i0[E];             // node index 0
    int<lower=1,upper=N> i1[E];             // node index 1
    int<lower=1> K;                         // latent dimension
    real mu_alpha;                          // prior mean for node heterogeneity
    real<lower=0> sig2_alpha;               // prior variance for node heterogeneity
    real mu_Z;                              // prior mean for latent positions
    real<lower=0> sig2_Z;                   // prior variance for latent positions
}
parameters {
    matrix[N, K] Z;                         // latent positions
    vector[N] alpha;                        // latent node heterogeneity
}
transformed parameters {
    vector[E] Theta_A;                      // logit link probability
    Theta_A = alpha[i0] + alpha[i1] + rows_dot_product(Z[i0], Z[i1]);
    vector[E] proba = inv_logit(Theta_A);
    matrix[N, N] ZZt = tcrossprod(Z);
}
model {
    // priors
    alpha ~ normal(mu_alpha, sqrt(sig2_alpha));
    to_vector(Z) ~ normal(mu_Z, sqrt(sig2_Z));
    // adjacency model
    A ~ bernoulli_logit(Theta_A);
}