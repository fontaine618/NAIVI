model_none = """
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
"""

model_both = """
data {
    int<lower=1> N;                         // number of nodes
    int<lower=1> E;                         // number of edges
    int<lower=0> p;                         // number of attributes
    int<lower=0> p_cts;                     // number of continuous node attributes
    int<lower=0> p_bin;                     // number of binary node attributes
    real X_cts[N, p_cts];                   // observed continuous attributes
    int<lower=0,upper=1> X_bin[N, p_bin];   // observed binary attributes
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
    real<lower=0> sig2_X[p_cts];            // noise for continuous attributes
    matrix[K, p] B;                         // regression weights
    row_vector[p] B0;                       // regression bias
    matrix[N, K] Z;                         // latent positions
    vector[N] alpha;                        // latent node heterogeneity
}
transformed parameters {
    row_vector[p] Theta_X[N];               // GLM parameter
    for (u in 1:N) Theta_X[u] = B0 + Z[u] * B;
    vector[E] Theta_A;                      // logit link probability
    Theta_A = alpha[i0] + alpha[i1] + rows_dot_product(Z[i0], Z[i1]);
    vector[E] proba = inv_logit(Theta_A);
    matrix[N, N] ZZt = tcrossprod(Z);
}
model {
    // priors
    alpha ~ normal(mu_alpha, sqrt(sig2_alpha));
    to_vector(Z) ~ normal(mu_Z, sqrt(sig2_Z));
    sig2_X ~ inv_gamma(1., 1.);
    to_vector(B) ~ normal(0., 100.);
    B0 ~ normal(0., 100.);
    // adjacency model
    A ~ bernoulli_logit(Theta_A);
    for (u in 1:N) {
        // continuous covariates model
        for (j in 1:p_cts) X_cts[u, j] ~ normal(Theta_X[u, j], sqrt(sig2_X[j]));
        // binary attributes model
        for (j in 1:p_bin) X_bin[u, j] ~ bernoulli_logit(Theta_X[u, j+p_cts]);
    }
}
"""

model_cts = """
data {
    int<lower=1> N;                         // number of nodes
    int<lower=1> E;                         // number of edges
    int<lower=0> p;                         // number of attributes
    int<lower=0> p_cts;                     // number of continuous node attributes
    real X_cts[N, p_cts];                   // observed continuous attributes
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
    real<lower=0> sig2_X[p_cts];            // noise for continuous attributes
    matrix[K, p] B;                         // regression weights
    row_vector[p] B0;                       // regression bias
    matrix[N, K] Z;                         // latent positions
    vector[N] alpha;                        // latent node heterogeneity
}
transformed parameters {
    row_vector[p] Theta_X[N];               // GLM parameter
    for (u in 1:N) Theta_X[u] = B0 + Z[u] * B;
    vector[E] Theta_A;                      // logit link probability
    Theta_A = alpha[i0] + alpha[i1] + rows_dot_product(Z[i0], Z[i1]);
    vector[E] proba = inv_logit(Theta_A);
    matrix[N, N] ZZt = tcrossprod(Z);
}
model {
    // priors
    alpha ~ normal(mu_alpha, sqrt(sig2_alpha));
    to_vector(Z) ~ normal(mu_Z, sqrt(sig2_Z));
    sig2_X ~ inv_gamma(1., 1.);
    to_vector(B) ~ normal(0., 10.);
    B0 ~ normal(0., 10.);
    // adjacency model
    A ~ bernoulli_logit(Theta_A);
    for (u in 1:N) {
        // continuous covariates model
        for (j in 1:p_cts) X_cts[u, j] ~ normal(Theta_X[u, j], sqrt(sig2_X[j]));
    }
}
"""

model_bin = """
data {
    int<lower=1> N;                         // number of nodes
    int<lower=1> E;                         // number of edges
    int<lower=0> p;                         // number of attributes
    int<lower=0> p_bin;                     // number of binary node attributes
    int<lower=0,upper=1> X_bin[N, p_bin];   // observed binary attributes
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
    matrix[K, p] B;                         // regression weights
    row_vector[p] B0;                       // regression bias
    matrix[N, K] Z;                         // latent positions
    vector[N] alpha;                        // latent node heterogeneity
}
transformed parameters {
    row_vector[p] Theta_X[N];               // GLM parameter
    for (u in 1:N) Theta_X[u] = B0 + Z[u] * B;
    vector[E] Theta_A;                      // logit link probability
    Theta_A = alpha[i0] + alpha[i1] + rows_dot_product(Z[i0], Z[i1]);
    vector[E] proba = inv_logit(Theta_A);
    matrix[N, N] ZZt = tcrossprod(Z);
}
model {
    // priors
    alpha ~ normal(mu_alpha, sqrt(sig2_alpha));
    to_vector(Z) ~ normal(mu_Z, sqrt(sig2_Z));
    to_vector(B) ~ normal(0., 10.);
    B0 ~ normal(0., 10.);
    // adjacency model
    A ~ bernoulli_logit(Theta_A);
    for (u in 1:N) {
        // binary attributes model
        for (j in 1:p_bin) X_bin[u, j] ~ bernoulli_logit(Theta_X[u, j]);
    }
}
"""