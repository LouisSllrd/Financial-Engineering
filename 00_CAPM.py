import matplotlib.pyplot as plt
import numpy as np

# Parameters
Rf = 0.02   # risk-free rate (2%)
Rm = 0.08   # expected return on market (8%)
sigma_m = 0.18  # standard deviation of market portfolio

# Security Market Line (SML) parameters
betas = np.linspace(-0.5, 2, 100)
expected_returns = Rf + betas * (Rm - Rf)

# Capital Market Line (CML) parameters
sigmas = np.linspace(0, 0.3, 100)
cml_returns = Rf + (Rm - Rf)/sigma_m * sigmas

# Plot
fig, ax = plt.subplots(1, 2, figsize=(14,6))

# Plot SML
ax[0].plot(betas, expected_returns, label="Security Market Line (SML)", color="blue")
ax[0].scatter([1], [Rm], color="red", zorder=5, label="Market Portfolio (M)")
ax[0].axhline(y=Rf, color="gray", linestyle="--", linewidth=1)
ax[0].set_title("Security Market Line (SML)", fontsize=14)
ax[0].set_xlabel("Beta (Systematic Risk)")
ax[0].set_ylabel("Expected Return")
ax[0].legend()
ax[0].grid(True, linestyle="--", alpha=0.6)

# Plot CML
ax[1].plot(sigmas, cml_returns, label="Capital Market Line (CML)", color="green")
ax[1].scatter([sigma_m], [Rm], color="red", zorder=5, label="Market Portfolio (M)")
ax[1].scatter([0], [Rf], color="orange", zorder=5, label="Risk-Free Asset (Rf)")
ax[1].set_title("Capital Market Line (CML)", fontsize=14)
ax[1].set_xlabel("Standard Deviation (Total Risk)")
ax[1].set_ylabel("Expected Return")
ax[1].legend()
ax[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
