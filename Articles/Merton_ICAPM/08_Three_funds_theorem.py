import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
r = 0.03  # risk-free rate
alpha = np.array([0.08, 0.05])  # expected returns: [stock, bond]
sigma = np.array([0.2, 0.1])    # volatilities
rho = 0.2                       # correlation stock-bond
cov_matrix = np.array([
    [sigma[0]**2, rho*sigma[0]*sigma[1]],
    [rho*sigma[0]*sigma[1], sigma[1]**2]
])

# -------------------------------
# Fund 1: Tangency risky portfolio
# -------------------------------
excess = alpha - r
inv_cov = np.linalg.inv(cov_matrix)
w_tangency = inv_cov @ excess / (np.ones(2) @ inv_cov @ excess)
mu_tangency = w_tangency @ alpha
var_tangency = w_tangency @ cov_matrix @ w_tangency

# -------------------------------
# Fund 2: Bond asset (hedge against interest rates)
# -------------------------------
# Simply the bond itself (asset 2)
fund2 = np.array([0.0, 1.0])
mu_fund2 = alpha[1]
var_fund2 = sigma[1]**2

# -------------------------------
# Efficient frontier using Fund1 + risk-free (classic 2-fund separation)
# -------------------------------
lambdas = np.linspace(-0.5, 1.5, 100)
frontier_rf = [(1-l)*r + l*mu_tangency for l in lambdas]
frontier_risk = [np.sqrt((l**2)*var_tangency) for l in lambdas]

# -------------------------------
# Investor allocations: effect of hedging demand H
# -------------------------------
H_values = np.linspace(-0.5, 0.5, 11)   # hedging demand (signed)
s = 0.6   # total fraction allocated to risky assets (choose by risk tolerance)
allocations = []   # will store [w1, w2, w3] with w1 = Fund1, w2 = Fund2, w3 = risk-free
for H in H_values:
    w2 = H                      # direct hedging allocation (can be negative)
    w1 = s - w2                 # remainder of risky exposure goes to Fund1
    w3 = 1.0 - s                # risk-free share (fixed here)
    allocations.append([w1, w2, w3])
allocations = np.array(allocations)

# -------------------------------
# Simulation: portfolio with vs without Fund2 under rate shocks
# -------------------------------
np.random.seed(42)
T = 10

r = 0.02  # baseline risk-free rate
alpha = [0.08, 0.03]  # expected returns: stock, bond
sigma = [0.15, 0.05]  # volatilities: stock, bond

c0 = 1.0    # baseline consumption
beta = 5.0  # sensitivity to portfolio return
gamma = 10.0  # sensitivity to interest rate

# -------------------------------
# Simulate rate shocks and returns
# -------------------------------
rate_shocks = np.random.normal(0, 0.01, T)

rf_returns = r + rate_shocks
stock_returns = np.random.normal(alpha[0], sigma[0], T)
bond_returns = np.random.normal(alpha[1] - 3*rate_shocks, sigma[1], T)

# -------------------------------
# Portfolios
# -------------------------------
port_A = 0.7*stock_returns + 0.3*rf_returns
port_B = 0.6*stock_returns + 0.3*rf_returns + 0.1*bond_returns

# -------------------------------
# Consumption functions
# -------------------------------
consumption_A = c0 + beta*port_A - gamma*rf_returns
consumption_B = c0 + beta*port_B - gamma*rf_returns

# -------------------------------
# Plotting
# -------------------------------
fig, axs = plt.subplots(1, 3, figsize=(18,5))

# --- Graph 1: Efficient frontier ---
axs[0].plot(frontier_risk, frontier_rf, label="Frontier (Fund1+RF)", color="blue")
axs[0].scatter(np.sqrt(var_tangency), mu_tangency, color="red", label="Fund1 (Tangency)")
axs[0].scatter(np.sqrt(var_fund2), mu_fund2, color="green", label="Fund2 (Bond Hedge)")
axs[0].set_xlabel("Risk (Std Dev)")
axs[0].set_ylabel("Expected Return")
axs[0].set_title("Efficient Frontier & Funds")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.6)

# --- Graph 2: Allocations vs H ---
axs[1].plot(H_values, allocations[:,0], label="Fund1 (Tangency)")
axs[1].plot(H_values, allocations[:,1], label="Fund2 (Hedge)")
axs[1].plot(H_values, allocations[:,2], label="Fund3 (Risk-free)")
axs[1].set_xlabel("Hedging Demand H")
axs[1].set_ylabel("Portfolio Share")
axs[1].set_title("Investor Allocations into 3 Funds")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.6)

# --- Graph 3: Simulation of returns ---
axs[2].plot(consumption_A, label='Consumption Portfolio A (without hedge)', color='blue')
axs[2].plot(consumption_B, label='Consumption Portfolio B (with hedge)', color='green')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Consumption')
axs[2].set_title('Consumption over time under rate shocks')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
