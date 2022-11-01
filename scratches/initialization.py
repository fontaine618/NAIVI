from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.initialization import initialize
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, VIMC, MLE, GLM
from NAIVI.initialization.usvt import initialize_latent_variables
from NAIVI.utils.metrics import projection_distance, invariant_distance
import torch
import httpstan

httpstan

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 500
K = 5
p_cts = 10
p_bin = 0
var_cts = 1.
missing_mean = -2.50 #-3 ~ 7%, -2 ~ 15%, -1 ~ 30% , 0. ~ 50%
alpha_mean = -1.85
seed = 0

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_mean=missing_mean,
    alpha_mean=alpha_mean, seed=seed, mnar_sparsity=0.3
)
mnar = False
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)

# coefficient init
self = GLM(K, N, p_cts, p_bin, mnar, Z)
self.fit(train, test, batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.5, reg=0.27)

self.covariate_weight
B.T
C.T

# Latent variable init
alpha_hat, Z_hat = initialize_latent_variables(i0, i1, A, K, None)

projection_distance(Z, Z_hat)
invariant_distance(Z, Z_hat)



self = MLE(K, N, p_cts, p_bin)
# self = ADVI(K, N, p_cts, p_bin)
# self = VIMC(K, N, p_cts, p_bin)
self.init(positions=Z_hat, heterogeneity=alpha_hat)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.1)

list(self.model.hyperparameters())