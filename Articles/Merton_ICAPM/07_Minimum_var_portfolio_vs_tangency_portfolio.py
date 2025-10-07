# Compute and plot the portfolio mix curve for two assets,
# mark the global minimum-variance portfolio and the tangency (max-Sharpe) portfolio.
import numpy as np
import matplotlib.pyplot as plt

# Parameters (same as before)
r = 0.02
alpha = np.array([0.08, 0.12])
sigma = np.array([0.20, 0.30])
rho = 0.2

# Covariance matrix
Sigma = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]],
                  [rho*sigma[0]*sigma[1], sigma[1]**2]])

# Excess returns
excess = alpha - r

# Sweep w1 from 0 to 1 (weight on Asset 1); w2 = 1 - w1
w1_vals = np.linspace(0, 1, 501)
means = []
stds = []
for w1 in w1_vals:
    w = np.array([w1, 1-w1])
    mean = w @ alpha
    var = w @ Sigma @ w
    means.append(mean)
    stds.append(np.sqrt(var))
means = np.array(means)
stds = np.array(stds)

# Global minimum-variance portfolio among the two assets (no constraint on sum=1? 
# For two-asset unconstrained, global min var with weights that sum to 1 can be computed analytically)
# Here we restrict weights to sum 1 (so it's on the segment w1 in [0,1]); but the true unconstrained min-var may lie outside [0,1].
# Compute the analytical min-variance weight (sum to 1)
# w1_minvar = (sigma2^2 - cov) / (sigma1^2 + sigma2^2 - 2 cov)
cov = Sigma[0,1]
s1sq = Sigma[0,0]
s2sq = Sigma[1,1]
w1_minvar = (s2sq - cov) / (s1sq + s2sq - 2*cov)
w_minvar = np.array([w1_minvar, 1-w1_minvar])
mean_minvar = w_minvar @ alpha
std_minvar = np.sqrt(w_minvar @ Sigma @ w_minvar)

# Tangency (maximum Sharpe) portfolio among risky assets (proportional to Sigma^{-1} * excess)
invSigma = np.linalg.inv(Sigma)
tangency_raw = invSigma @ excess
# Normalize to sum=1 to get composition of the risky fund (as in previous plot)
w_tangency = tangency_raw / tangency_raw.sum()
mean_tangency = w_tangency @ alpha
std_tangency = np.sqrt(w_tangency @ Sigma @ w_tangency)
# Also find the w1 value nearest to w_tangency[0] on the sweep
idx_near_tang = np.argmin(np.abs(w1_vals - w_tangency[0]))

# Plot the mean-std curve (parametric in w1)
plt.figure(figsize=(8,6))
plt.plot(stds, means, label='Portfolio curve (mix Asset1/Asset2)', lw=2)
plt.scatter(std_minvar, mean_minvar, color='red', label=f'Min-Var (w1={w1_minvar:.3f})', zorder=5)
plt.scatter(std_tangency, mean_tangency, color='green', label=f'Tangency risky fund (w1={w_tangency[0]:.3f})', zorder=5)
# point for w1 from earlier graph1 (w_risky normalized)
plt.scatter(stds[idx_near_tang], means[idx_near_tang], marker='x', color='black', s=80, label='nearest sweep point to tangency', zorder=6)
plt.xlabel('Portfolio risk (std dev)')
plt.ylabel('Expected return')
plt.title('Mixing Asset1 and Asset2: mean vs risk curve\n(min-variance point and tangency portfolio marked)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Print numeric comparisons
print("Analytic min-variance weight (w1):", w1_minvar)
print("Min-variance portfolio (w1,w2):", w_minvar)
print("Tangency composition (normalized to sum=1) w1:", w_tangency[0])
print("Tangency composition (w1,w2):", w_tangency)
print("Nearest grid w1 to tangency on sweep:", w1_vals[idx_near_tang])
