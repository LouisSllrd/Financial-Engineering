import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Parameters
# -------------------
alpha = 0.08   # risky expected return
r = 0.02       # risk-free rate
sigma = 0.3    # risky volatility
cX = -0.5      # effect of state variable on consumption (negative = bad shock)
cW = 1.0       # effect of wealth on consumption
g = 0.2        # volatility of state variable (interest rate)

# -------------------
# Functions
# -------------------
def speculative_demand(alpha, r, sigma, gamma):
    return (1/gamma) * (alpha - r) / (sigma**2)

def hedging_demand(cX, cW, g, sigma, rho):
    return -(cX/cW) * (g * rho / sigma)

def total_demand(alpha, r, sigma, gamma, cX, cW, g, rho):
    return speculative_demand(alpha, r, sigma, gamma) + hedging_demand(cX, cW, g, sigma, rho)

# -------------------
# 1. Speculative vs Hedging vs Total demand
# -------------------
rhos = np.linspace(-1, 1, 200)
gamma = 3.0

spec = speculative_demand(alpha, r, sigma, gamma)
hedge = np.array([hedging_demand(cX, cW, g, sigma, rho) for rho in rhos])
total = spec + hedge

plt.figure(figsize=(9,6))
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.plot(rhos, [spec]*len(rhos), label="Speculative demand", color="red")
plt.plot(rhos, hedge, label="Hedging demand", color="blue")
plt.plot(rhos, total, label="Total demand", color="green", linewidth=2)
plt.title("Speculative vs Hedging Demand for Risky Asset")
plt.xlabel("Correlation ρ with state variable shocks")
plt.ylabel("Optimal risky investment wW")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# -------------------
# 2. Heatmap: total demand vs γ and ρ
# -------------------
gammas = np.linspace(1, 10, 50)
rhos = np.linspace(-1, 1, 50)
W = np.zeros((len(gammas), len(rhos)))

for i, gma in enumerate(gammas):
    for j, rho in enumerate(rhos):
        W[i,j] = total_demand(alpha, r, sigma, gma, cX, cW, g, rho)

plt.figure(figsize=(10,6))
plt.imshow(W, aspect="auto", origin="lower",
           extent=[rhos[0], rhos[-1], gammas[0], gammas[-1]],
           cmap="coolwarm")
plt.colorbar(label="Optimal risky investment wW")
plt.title("Heatmap of Total Demand (wW) vs Risk Aversion γ and Correlation ρ")
plt.xlabel("Correlation ρ")
plt.ylabel("Risk Aversion γ")
plt.show()

# -------------------
# 3. Portfolio return vs variance (efficient frontier intuition)
# -------------------
# Take different ρ values
rho_values = [-0.8, 0, 0.8]
colors = ["blue", "black", "orange"]

plt.figure(figsize=(9,6))
for rho, col in zip(rho_values, colors):
    w = total_demand(alpha, r, sigma, gamma, cX, cW, g, rho)
    mean_return = r + w * (alpha - r)
    variance = (w**2) * (sigma**2)
    plt.scatter(variance, mean_return, color=col, label=f"ρ={rho}, w={w:.2f}")

plt.title("Impact of Hedging on Risk-Return Tradeoff")
plt.xlabel("Portfolio Variance")
plt.ylabel("Expected Return")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# -------------------
# 4. Hedging demand vs state-variable volatility g
# -------------------
g_values = np.linspace(0, 0.5, 100)
rho = 0.5
hedges = [hedging_demand(cX, cW, gv, sigma, rho) for gv in g_values]

plt.figure(figsize=(9,6))
plt.plot(g_values, hedges, label="Hedging demand", color="purple")
plt.title("Hedging Demand vs Volatility of State Variable g")
plt.xlabel("Volatility of state variable g")
plt.ylabel("Hedging demand (wW)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
