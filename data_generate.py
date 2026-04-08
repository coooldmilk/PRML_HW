# Generating 3D make-moons data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    rng = np.random.default_rng(random_state) #随机种子

    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += rng.normal(scale=noise, size=X.shape)

    return X, y

""" # Generate the data (1000 datapoints)
X, labels = make_moons_3d(n_samples=1000, noise=0.2) """

# train
# 总1000 → 每类500 → n_samples=500
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2, random_state=42)

# test
# 总500 → 每类250 → n_samples=250
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2, random_state=123)

train_data = np.column_stack((X_train, y_train))
test_data = np.column_stack((X_test, y_test))

np.savetxt("train.csv", train_data, delimiter=",", header="x1,x2,x3", comments="")
np.savetxt("test.csv", test_data, delimiter=",", header="x1,x2,x3", comments="")

print("train_c0:", train_data.shape)
print("test_c0:", test_data.shape)


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Make Moons')
plt.show()