import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Baseline parameters
# ----------------------
alpha = np.array([0.08, 0.12])   # expected returns of assets
r = 0.02                          # risk-free rate
nu = np.array([0.15, 0.25])       # volatilities
rho = 0.3                         # correlation between assets

# ----------------------
# Compute weights given parameters
# ----------------------
def compute_weights(alpha, r, nu, rho, gamma):
    Sigma = np.array([
        [nu[0]**2, rho * nu[0] * nu[1]],
        [rho * nu[0] * nu[1], nu[1]**2]
    ])
    excess = alpha - r
    invSigma = np.linalg.inv(Sigma)
    w = (1/gamma) * invSigma @ excess
    return w

# ----------------------
# Sweep parameters for gamma and rho
# ----------------------
gammas = np.linspace(1, 10, 50)
rho_values = np.linspace(0, 0.9, 50)

weights_gamma = np.array([compute_weights(alpha, r, nu, rho, g) for g in gammas])
weights_rho = np.array([compute_weights(alpha, r, nu, rh, 3) for rh in rho_values])

# ----------------------
# Efficient frontier (vary gamma)
# ----------------------
returns = []
risks = []
for g in gammas:
    w = compute_weights(alpha, r, nu, rho, g)
    port_ret = r + w @ (alpha - r)     # expected portfolio return
    Sigma = np.array([
        [nu[0]**2, rho * nu[0] * nu[1]],
        [rho * nu[0] * nu[1], nu[1]**2]
    ])
    port_var = w @ Sigma @ w          # portfolio variance
    returns.append(port_ret)
    risks.append(np.sqrt(port_var))

returns = np.array(returns)
risks = np.array(risks)

# ----------------------
# Plotting
# ----------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Left plot: effect of gamma ---
axs[0].plot(gammas, weights_gamma[:,0], label="w1")
axs[0].plot(gammas, weights_gamma[:,1], label="w2")
axs[0].plot(gammas, weights_gamma.sum(axis=1), label="w1+w2", linestyle="--")
axs[0].set_title("Effect of Risk Aversion γ on Weights")
axs[0].set_xlabel("Risk Aversion γ")
axs[0].set_ylabel("Weights")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.6)

# --- Middle plot: effect of correlation ---
axs[1].plot(rho_values, weights_rho[:,0], label="w1")
axs[1].plot(rho_values, weights_rho[:,1], label="w2")
axs[1].plot(rho_values, weights_rho.sum(axis=1), label="w1+w2", linestyle="--")
axs[1].set_title("Effect of Correlation ρ on Weights (γ=3)")
axs[1].set_xlabel("Correlation ρ")
axs[1].set_ylabel("Weights")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.6)

# --- Right plot: efficient frontier ---
axs[2].plot(risks, returns, marker="o", label="Efficient Frontier")
axs[2].scatter(nu[0], alpha[0], color="red", label="Asset 1")
axs[2].scatter(nu[1], alpha[1], color="blue", label="Asset 2")
axs[2].set_title("Efficient Frontier vs Individual Assets")
axs[2].set_xlabel("Risk (Std Dev)")
axs[2].set_ylabel("Expected Return")
axs[2].legend()
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
