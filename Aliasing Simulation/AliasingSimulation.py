import numpy as np
import matplotlib.pyplot as plt

def plot_1D(x, velocity_flow, velocity_flow_aliased, venc):
    plt.figure(figsize=(10, 5))
    plt.plot(x, velocity_flow, label=f"Velocity flow before applying venc={venc} (before aliasing)", color="blue", linewidth=2)
    plt.plot(x, velocity_flow_aliased, label=f"Velocity flow after applying venc={venc} (with aliasing)", color="red", linestyle="--", linewidth=2)
    plt.axhline(y=venc, color="orange", linestyle="-", label="+Venc limit")
    plt.axhline(y=-venc, color="orange", linestyle="-", label="-Venc limit")
    plt.xlabel("Position along the tube (cm)")
    plt.ylabel("Velocity (cm/s)")
    plt.title("Simulation of aliasing effect in MRI 4D-flow (1D flow)")
    plt.legend()
    plt.grid(True)
    plt.show()

def tube_1D(length, num_samples):
    return np.linspace(start=0,stop=length,num=num_samples) # a 1D array of num_samples points from 0 to length that are evenly spaced

def velocity_1D(x):
    return 20 + 10*np.sin(x) + 0.5*x

def aliasing_simulation_1D(venc):
    x = tube_1D(10, 100)
    velocity_flow = velocity_1D(x)
    velocity_flow_aliased = np.array([])

    for v in velocity_flow:
        if v > venc:
            v_aliased = v - 2*venc
        elif v < -venc:
            v_aliased = v + 2*venc
        else:
            v_aliased = v
        velocity_flow_aliased = np.append(velocity_flow_aliased, v_aliased)
    plot_1D(x, velocity_flow, velocity_flow_aliased, venc)

aliasing_simulation_1D(venc=30)


def plot_2D(V, V_aliased, venc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(V, cmap='coolwarm', vmin=-50, vmax=50)
    ax1.set_title("Velocity field before aliasing")
    fig.colorbar(im1, ax=ax1, label="Velocity (cm/s)")

    im2 = ax2.imshow(V_aliased, cmap='coolwarm', vmin=-50, vmax=50)
    ax2.set_title(f"Velocity after aliasing (Venc={venc})")
    fig.colorbar(im2, ax=ax2, label="Velocity (cm/s)")

    plt.tight_layout()
    plt.show()

def velocity_field_2D(num_samples=100):
    x = np.linspace(start=-5, stop=5, num=num_samples)
    y = np.linspace(start=-5, stop=5, num=num_samples)
    X, Y = np.meshgrid(x, y)

    V = 50 * np.exp(-(X**2 + Y**2)/10) # velocity field in cm/s
    return X, Y, V

def aliasing_simulation_2D(venc):
    X, Y, velocity_flow = velocity_field_2D(100)
    velocity_flow_aliased = np.where(
        velocity_flow > venc, velocity_flow - 2*venc,
        np.where(velocity_flow < -venc, velocity_flow + 2*venc,
        velocity_flow))
    plot_2D(velocity_flow, velocity_flow_aliased, venc)

aliasing_simulation_2D(venc=30)