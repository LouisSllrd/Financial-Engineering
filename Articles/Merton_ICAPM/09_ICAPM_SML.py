import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
alpha_M_minus_r = 0.06   # 6% market premium
alpha_n_minus_r = -0.02  # -2% hedge asset premium (investors pay to hedge)
r = 0.02                 # 2% risk-free rate

# Grid of betas
beta_M = np.linspace(-0.5, 1.5, 30)
beta_r = np.linspace(-1.5, 1.5, 30)
BM, BR = np.meshgrid(beta_M, beta_r)

# Expected excess return plane (ICAPM)
ER_minus_r = BM * alpha_M_minus_r + BR * alpha_n_minus_r
ER = ER_minus_r + r

# Example assets
assets = {
    "Stock": (1.2, 0.1),
    "Long Bond": (0.1, -1.0),
    "Market": (1.0, 0.0),
    "Hedge Asset n": (0.0, 1.0)
}
asset_returns = {name: r + bm*alpha_M_minus_r + br*alpha_n_minus_r
                 for name, (bm, br) in assets.items()}

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the plane
ax.plot_surface(BM, BR, ER, alpha=0.5, cmap="viridis")

# Plot example assets
for name, (bm, br) in assets.items():
    ax.scatter(bm, br, asset_returns[name], s=80, label=name)

ax.set_xlabel(r"$\beta_M$ (Market Beta)")
ax.set_ylabel(r"$\beta_r$ (Hedge Beta)")
ax.set_zlabel("Expected Return")
ax.set_title("ICAPM: Security Market Plane")
ax.legend()

plt.show()

# ------------- 2D projection ----------------

# Parameters
r = 0.02       # risk-free rate
lam_M = 0.06   # price of market risk (lambda_M)
lam_r = -0.03  # price of interest rate risk (lambda_r)

# Define function for expected return under ICAPM plane
def expected_return(beta_M, beta_r, r, lam_M, lam_r):
    return r + lam_M * beta_M + lam_r * beta_r

# Beta ranges
beta_M = np.linspace(-0.5, 2, 100)
beta_r = np.linspace(-1, 1, 100)

# --- 2D Slice 1: Vary beta_M, fix beta_r ---
beta_r_fixed = 0.5
E_M = expected_return(beta_M, beta_r_fixed, r, lam_M, lam_r)

# --- 2D Slice 2: Vary beta_r, fix beta_M ---
beta_M_fixed = 1.0
E_r = expected_return(beta_M_fixed, beta_r, r, lam_M, lam_r)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Left: expected return vs beta_M (beta_r fixed)
axs[0].plot(beta_M, E_M, label=fr"$\beta_r={beta_r_fixed}$", color="blue")
axs[0].axhline(r, color="black", linestyle="--", label="Risk-free rate")
axs[0].set_title("Expected Return vs Market Beta")
axs[0].set_xlabel(r"Market Beta $\beta_M$")
axs[0].set_ylabel("Expected Return")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.6)

# Right: expected return vs beta_r (beta_M fixed)
axs[1].plot(beta_r, E_r, label=fr"$\beta_M={beta_M_fixed}$", color="red")
axs[1].axhline(r, color="black", linestyle="--", label="Risk-free rate")
axs[1].set_title("Expected Return vs Hedge Beta")
axs[1].set_xlabel(r"Hedge Beta $\beta_r$")
axs[1].set_ylabel("Expected Return")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()