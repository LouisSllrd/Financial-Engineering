# portfolio_decomposition.py
import numpy as np
import matplotlib.pyplot as plt

# ---- Model set-up ----
# 2 risky assets
alpha = np.array([0.08, 0.12])   # expected returns
r = 0.02
nu = np.array([0.15, 0.25])      # volatilities
rho = 0.3

# covariance matrix
Sigma = np.array([
    [nu[0]**2, rho*nu[0]*nu[1]],
    [rho*nu[0]*nu[1], nu[1]**2]
])

invSigma = np.linalg.inv(Sigma)
excess = alpha - r

# assume total wealth W = 1 (dollar units)
W = 1.0

# ---- Choose A and hedging parameters ----
# A = -J_W / J_WW  (scale for the myopic demand)
A = 0.33   # chosen risk-tolerance scalar (tune this)

# Single state variable (m = 1) for simplicity
# Gamma is n x m: covariances between asset returns and state shock (per unit time)
Gamma = np.array([[0.02],   # asset1 cov with state shock
                  [-0.01]]) # asset2 cov with state shock

# H is m x 1 (here scalar) ; H = -J_Wx / J_WW  (captures hedging need)
# We'll show effect for various H values (positive H => hedging demand direction)
H_values = np.array([-0.4, -0.2, 0.0, 0.2, 0.5])  # try several hedging strengths

# ---- Function computing decomposition ----
def compute_decomposition(A, invSigma, excess, Gamma, H_scalar, W=1.0):
    # myopic monetary holdings = A * invSigma @ excess
    myopic = A * (invSigma @ excess)            # shape (2,)
    # hedging monetary holdings = invSigma @ (Gamma * H)
    # Gamma is n x 1, H_scalar is scalar -> Gamma*H is n x 1
    hedging = (invSigma @ (Gamma.flatten() * H_scalar))
    total = myopic + hedging
    # convert to weight fractions if desired: w = total / W (here W=1)
    return myopic, hedging, total

# ---- Plot stacked bars for each H ----
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# choose a representative H to show single decomposition
H0 = 0.2
my0, hed0, tot0 = compute_decomposition(A, invSigma, excess, Gamma, H0, W=1.0)

x = np.arange(2)  # asset indices
axes[0].bar(x - 0.2, my0, width=0.4, label='Myopic (mean-variance)', color='tab:blue')
axes[0].bar(x + 0.2, hed0, width=0.4, label='Hedging', color='tab:orange')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Asset 1', 'Asset 2'])
axes[0].set_title(f'Decomposition of holdings (W={W}, A={A}, H={H0})')
axes[0].legend()
axes[0].axhline(0, color='k', linewidth=0.5)

# ---- Show how holdings change as H varies ----
totals = []
myopics = []
hedges = []
for H in H_values:
    my, hed, tot = compute_decomposition(A, invSigma, excess, Gamma, H, W=1.0)
    totals.append(tot)
    myopics.append(my)
    hedges.append(hed)

totals = np.array(totals)    # shape (len(H_values), 2)
myopics = np.array(myopics)
hedges = np.array(hedges)

# plot holdings of each asset vs H
axes[1].plot(H_values, totals[:,0], marker='o', label='Total Asset1', color='tab:green')
axes[1].plot(H_values, myopics[:,0], marker='x', linestyle='--', label='Myopic Asset1', color='tab:blue')
axes[1].plot(H_values, hedges[:,0], marker='s', linestyle=':', label='Hedging Asset1', color='tab:orange')

axes[1].plot(H_values, totals[:,1], marker='o', label='Total Asset2', color='tab:red')
axes[1].plot(H_values, myopics[:,1], marker='x', linestyle='--', label='Myopic Asset2', color='tab:purple')
axes[1].plot(H_values, hedges[:,1], marker='s', linestyle=':', label='Hedging Asset2', color='tab:brown')

axes[1].set_xlabel('H (hedging coefficient)')
axes[1].set_ylabel('Monetary holding (w_i * W)')
axes[1].set_title('How holdings vary with hedging motive H')
axes[1].legend(loc='best')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
