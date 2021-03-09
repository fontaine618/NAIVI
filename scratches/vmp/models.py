import torch
from NAIVI.utils.gen_data import generate_dataset
torch.set_default_dtype(torch.float64)

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------
N = 200
K = 5
p_cts = 20
p_bin = 20
var_cts = 1.0
missing_rate = 0.5
alpha_mean = -1.85

Z, a, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=1
)

if X_cts is None:
    X_cts = torch.zeros((N, 0))
if X_bin is None:
    X_bin = torch.zeros((N, 0))

A.mean()


# -----------------------------------------------------------------------------
# AdjacencyModel
from NAIVI.vmp.gaussian import Gaussian
from NAIVI.vmp.models import AdjacencyModel


positions = Gaussian.from_array(torch.randn_like(Z), torch.full_like(Z, 2.))
heterogeneity = Gaussian.from_array(torch.randn_like(a)-1.85, torch.full_like(a, 1.))
indices = (i0, i1)
links = A

self = AdjacencyModel(positions, heterogeneity, indices, links)
self.forward()



for _ in range(10):
    with torch.no_grad():
        self.backward()
        elbo = self.to_elbo()
        print(elbo.item())

for _ in range(1):
    with torch.no_grad():
        self.forward()
        elbo = self.to_elbo()
        print(elbo.item())




torch.cat([heterogeneity.mean, a], 1)
torch.cat([positions.mean, Z], 1)

torch.cat([A, self._logistic.message_to_child.proba, self._sum.message_to_child.mean], 1)

self._logistic.message_to_child.proba[A==1.].mean()
self._logistic.message_to_child.proba[A==0.].mean()
self._sum.message_to_child.mean[A==1.].mean()
self._sum.message_to_child.mean[A==0.].mean()

# -----------------------------------------------------------------------------
# CovariateModel
from NAIVI.vmp.gaussian import Gaussian
from NAIVI.vmp.models import CovariateModel


positions = Gaussian.from_array(torch.randn_like(Z), torch.full_like(Z, 1.))
index = torch.arange(0, N)

self = CovariateModel(positions, index, X_cts, X_bin)

self.set_weight_and_bias(B, B0)

for _ in range(1000):
    with torch.no_grad():
        self.forward()
        self.backward()

    self.zero_grad()
    with torch.enable_grad():
        self.forward()
        elbo = self.to_elbo()
        print(elbo.item())
        elbo.backward()

        for n, p in self.named_parameters():
            p.data = p.data + 1.e-2 * p.grad

torch.cat([positions.mean, Z], 1)


for n, p in self.named_parameters():
    print(n)
    print(p)

self._gaussian.log_var.exp()

