import numpy as np
import matplotlib.pyplot as plt

# Consumption values
c = np.linspace(1, 10, 100)

# Utility function with gamma=2
gamma = 2
u = (c**(1-gamma)) / (1-gamma)

# Plot
plt.figure(figsize=(6,4))
plt.plot(c, u, color='blue', linewidth=2)
plt.title("Concave Utility Function (CRRA, γ=2)")
plt.xlabel("Consumption (c)")
plt.ylabel("Utility u(c)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
