import numpy as np
import matplotlib.pyplot as plt

# Create a grid of (x, y) points
x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

# Compute ||x - y||
distance = np.abs(X - Y)

# Define the piecewise function
Z = np.where(distance <= 1, 1, 1 / (1 + (np.abs(X - Y) - 1)**2))

# Create the 2D contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=np.linspace(0, 1, 51), cmap='viridis', vmin=0, vmax=1)

# Correct boundary labeling
plt.contour(X, Y, X - Y, levels=[1], colors='red', linewidths=2, linestyles='--')
plt.contour(X, Y, X - Y, levels=[-1], colors='red', linewidths=2, linestyles='--')

# Add a colorbar
cbar = plt.colorbar(contour, ticks=np.linspace(0, 1, 11))
cbar.set_label(r'$p$')

# Labels
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$p(x, y)$ in One Dimension')

plt.show()


'''
import numpy as np
import matplotlib.pyplot as plt

# Define a range of distances ||x - y||
d = np.linspace(0, 10, 500)

R = 1
# Define z(d) as piecewise
z = np.where(d <= R, 1, 1 / (1 + (d-R)**2))

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(d, z)
plt.axvline(x=1, color='red', linestyle='--', label=r'$d=R$ (boundary)')

# Labels
plt.xlabel(r'$d_{i j}$')
plt.ylabel(r'$p$')
plt.title(r'Probability with respect to distance $(\kappa = 1, R = 1)$')
plt.legend()
plt.grid(True)

plt.show()

'''