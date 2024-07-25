import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Laplace distribution
loc = 0.0  # Mean
scale = 0.5  # Scale (diversity)

# Generate samples from the Laplace distribution
samples = np.random.laplace(loc, scale, 1000)

# Plot the histogram of the samples
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

# Plot the Laplace distribution PDF
x = np.linspace(-10, 10, 1000)
pdf = (1/(2*scale)) * np.exp(-np.abs(x-loc)/scale)
plt.plot(x, pdf, 'k', linewidth=2)

plt.title('Laplace Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()