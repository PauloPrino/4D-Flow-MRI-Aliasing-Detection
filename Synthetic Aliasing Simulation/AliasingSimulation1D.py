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