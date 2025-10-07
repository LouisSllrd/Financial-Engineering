import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
r = 0.02  # risk-free rate
alpha = np.array([0.08, 0.12])  # expected returns of risky assets
sigma = np.array([0.20, 0.30])  # volatilities
rho = 0.2  # correlation

# Covariance matrix
Sigma = np.array([
    [sigma[0]**2, rho*sigma[0]*sigma[1]],
    [rho*sigma[0]*sigma[1], sigma[1]**2]
])

# Excess returns
excess = alpha - r

# -----------------------------
# Risky fund weights
# -----------------------------
invSigma = np.linalg.inv(Sigma)
w_risky = invSigma @ excess
w_risky = w_risky / np.sum(w_risky)  # normalize to sum=1

# Expected return and variance of risky fund
alpha_M = np.dot(w_risky, alpha)
var_M = w_risky.T @ Sigma @ w_risky
sigma_M = np.sqrt(var_M)

# -----------------------------
# Plot 1: Risky fund composition
# -----------------------------
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.bar(["Asset 1", "Asset 2"], w_risky, color=["skyblue", "salmon"])
plt.title("Risky Fund Composition")
plt.ylabel("Weight")
plt.ylim(0, 1)

# -----------------------------
# Plot 2: Capital Allocation Line
# -----------------------------
lambdas = [0, 0.5, 1, 1.5]  # proportion in risky fund
returns = [r + l*(alpha_M - r) for l in lambdas]
risks = [l*sigma_M for l in lambdas]

plt.subplot(1, 3, 2)
plt.plot(risks, returns, label="CAL (Risky fund + Risk-free)", color="black")
plt.scatter(risks, returns, c="red")
for l, x, y in zip(lambdas, risks, returns):
    plt.text(x, y, f"{int(l*100)}%", ha="left", va="bottom")
plt.title("Capital Allocation Line")
plt.xlabel("Portfolio Risk (σ)")
plt.ylabel("Expected Return")

# -----------------------------
# Plot 3: Security Market Line
# -----------------------------
# Compute betas of assets
cov_with_M = Sigma @ w_risky
betas = cov_with_M / var_M

# SML line
beta_range = np.linspace(0, 2, 50)
sml = r + beta_range * (alpha_M - r)

plt.subplot(1, 3, 3)
plt.plot(beta_range, sml, label="SML", color="black")
plt.scatter(betas, alpha, color=["skyblue", "salmon"], s=80, zorder=5)
for i, txt in enumerate(["Asset 1", "Asset 2"]):
    plt.text(betas[i]+0.02, alpha[i], txt)
plt.title("Security Market Line")
plt.xlabel("Beta")
plt.ylabel("Expected Return")
plt.legend()

plt.tight_layout()
plt.show()
