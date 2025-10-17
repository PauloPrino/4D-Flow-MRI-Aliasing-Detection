import numpy as np
import matplotlib.pyplot as plt

def transversal_tube_velocity_field(R=10, num_samples=100, v_max=50): # for a transversal cross-section of a tube
    x = np.linspace(-R, R, num_samples)
    y = np.linspace(-R, R, num_samples)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2) # distance from the center of the tube

    V = np.where(r <= R, v_max * (1 - (r / R)**2), 0) # calculating the velocity flow using Poiseuille's law

    return X, Y, V

def aliasing_simulation_transversal_tube(venc, R=10, num_samples=100, v_max=50):
    R_grid, Z_grid, V = transversal_tube_velocity_field(R, num_samples, v_max)
    V_aliased = np.where(V > venc, V - 2*venc, np.where(V < -venc, V + 2*venc, V))
    return R_grid, Z_grid, V_aliased, V

def plot_transversal_tube_velocity(V_non_aliased, v_aliased, title, v_max=50, R=10):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    im1 = axes[0].imshow(V_non_aliased, cmap='coolwarm', vmin=-v_max, vmax=v_max, extent=[-R, R, -R, R])
    circle = plt.Circle((0, 0), R, color='black', fill=False, linewidth=2)
    axes[0].add_patch(circle)
    axes[0].set_title(title+" (Non-aliased)")
    fig.colorbar(im1, ax=axes[0], label="Velocity (cm/s)")

    im2 = axes[1].imshow(v_aliased, cmap='coolwarm', vmin=-v_max, vmax=v_max, extent=[-R, R, -R, R])
    circle = plt.Circle((0, 0), R, color='black', fill=False, linewidth=2)
    axes[1].add_patch(circle)
    axes[1].set_title(title+" (Aliased)")
    fig.colorbar(im1, ax=axes[1], label="Velocity (cm/s)")
    plt.tight_layout()
    plt.show()

X, Y, v_aliased, V_non_aliased = aliasing_simulation_transversal_tube(venc=40, R=10, num_samples=100, v_max=50)
plot_transversal_tube_velocity(V_non_aliased, v_aliased, "Velocity field in a tube (Poiseuille's law)", v_max=50, R=10)

def side_view_velocity_field(R=10, tube_length=100, num_samples=100, v_max=50):
    r = np.linspace(-R, R, num_samples) # distances from the center to the edge of the tube (-R to R)

    z = np.linspace(0, tube_length, num_samples) # the longitudinal direction
    R_grid, Z_grid = np.meshgrid(r, z) # go from two 1D element to building a 2D grid

    # Poiseuille flow equation for velocity (only depends on r)
    V = v_max * (1 - (R_grid / R)**2)

    return R_grid, Z_grid, V

def plot_side_view_velocity(V_non_aliased, V_aliased, R_grid, Z_grid, title):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    im1 = axes[0].imshow(V_non_aliased.T, cmap='coolwarm', vmin=-50, vmax=50,
                   extent=[Z_grid.min(), Z_grid.max(), R_grid.min(), R_grid.max()],
                   aspect='auto')
    axes[0].set_xlabel("Tube Length (z)")
    axes[0].set_ylabel("Radial Distance (r)")
    axes[0].set_title(title+" (Non-aliased)")
    fig.colorbar(im1, ax=axes[0], label="Velocity (cm/s)")
    im2 = axes[1].imshow(V_aliased.T, cmap='coolwarm', vmin=-50, vmax=50,
                   extent=[Z_grid.min(), Z_grid.max(), R_grid.min(), R_grid.max()],
                   aspect='auto')
    axes[1].set_xlabel("Tube Length (z)")
    axes[1].set_ylabel("Radial Distance (r)")
    axes[1].set_title(title+" (Aliased)")
    fig.colorbar(im2, ax=axes[1], label="Velocity (cm/s)")
    plt.tight_layout()
    plt.show()

def aliasing_simulation_tube(venc, R=10, tube_length=100, num_samples=100, v_max=50):
    R_grid, Z_grid, V = side_view_velocity_field(R, tube_length, num_samples, v_max)
    V_aliased = np.where(V > venc, V - 2*venc, np.where(V < -venc, V + 2*venc, V))
    return R_grid, Z_grid, V_aliased, V

R_grid, Z_grid, V_aliased, V_non_aliased = aliasing_simulation_tube(venc=40, R=10, tube_length=100, num_samples=100, v_max=50)
plot_side_view_velocity(V_non_aliased, V_aliased, R_grid, Z_grid, "Blood Flow Velocity in a Tube (Side View)")

# With stenosis

def side_view_velocity_field_with_stenosis(R=10, tube_length=100, num_samples=100, v_max=50, stenosis_center=50, stenosis_width=10, stenosis_depth=0.7):
    r = np.linspace(-R, R, num_samples) # distances from the center to the edge of the tube (-R to R)
    z = np.linspace(0, tube_length, num_samples) # the longitudinal direction
    R_grid, Z_grid = np.meshgrid(r, z)

    R_z = R * (1 - stenosis_depth * np.exp(-0.5 * ((Z_grid - stenosis_center) / stenosis_width)**2)) # raidus depending on z, R(z) to simulate stenosis with gaussian format

    V = np.zeros_like(R_grid)
    for i in range(num_samples): # computing the velocity field
        for j in range(num_samples):
            if abs(R_grid[i, j]) <= R_z[i, j]:
                V[i, j] = v_max * (1 - (R_grid[i, j] / R_z[i, j])**2)
            else:
                V[i, j] = 0

    return R_grid, Z_grid, V, R_z

def plot_side_view_velocity_with_stenosis(V_non_aliased, V_aliased, R_grid, Z_grid, R_z, title, num_samples=100):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    axes[0].plot(Z_grid, R_z[:, num_samples//2], color='black', linewidth=2) # plot the variating radius to see the stenosis
    axes[0].plot(Z_grid, -R_z[:, num_samples//2], color='black', linewidth=2)

    im1 = axes[0].imshow(V_non_aliased.T, cmap='coolwarm', vmin=-50, vmax=50,
                   extent=[Z_grid.min(), Z_grid.max(), R_grid.min(), R_grid.max()],
                   aspect='auto')
    axes[0].set_xlabel("Tube Length (z)")
    axes[0].set_ylabel("Radial Distance (r)")
    axes[0].set_title(title+" (Non-aliased)")
    fig.colorbar(im1, ax=axes[0], label="Velocity (cm/s)")

    axes[1].plot(Z_grid, R_z[:, num_samples//2], color='black', linewidth=2)
    axes[1].plot(Z_grid, -R_z[:, num_samples//2], color='black', linewidth=2)

    im2 = axes[1].imshow(V_aliased.T, cmap='coolwarm', vmin=-50, vmax=50,
                   extent=[Z_grid.min(), Z_grid.max(), R_grid.min(), R_grid.max()],
                   aspect='auto')
    axes[1].set_xlabel("Tube Length (z)")
    axes[1].set_ylabel("Radial Distance (r)")
    axes[1].set_title(title+" (Aliased)")
    fig.colorbar(im2, ax=axes[1], label="Velocity (cm/s)")

    plt.tight_layout()
    plt.show()

R_grid, Z_grid, V_non_aliased, R_z = side_view_velocity_field_with_stenosis(R=10, tube_length=100, num_samples=100, v_max=50, stenosis_center=50, stenosis_width=10, stenosis_depth=0.7)

V_aliased = np.where(V_non_aliased > 40, V_non_aliased - 2*40, np.where(V_non_aliased < -40, V_non_aliased + 2*40, V_non_aliased))

plot_side_view_velocity_with_stenosis(V_non_aliased, V_aliased, R_grid, Z_grid, R_z, "Blood Flow Velocity in a Tube (Side View)")
