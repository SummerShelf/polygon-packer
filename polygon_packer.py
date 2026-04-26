import numpy as np
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as ppt
from numba import njit
from joblib import Parallel, delayed
import argparse
import time

timestart = time.time()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("inner_circles", type=int, help="Number of circles to pack")
arg_parser.add_argument("container_sides", type=int, help="Number of sides of the container polygon")
arg_parser.add_argument("--attempts", type=int, default=100, help="Number of random starts")
arg_parser.add_argument("--tolerance", type=float, default=1e-9, help="Overlap penalty tolerance")
arg_parser.add_argument("--finalstep", type=float, default=0.0001, help="Container size decrease step")
args = arg_parser.parse_args()

N = args.inner_circles
nsc = args.container_sides
attempts = args.attempts
penalty_tolerance = args.tolerance
final_step_size = args.finalstep

R = 1.0

unit_container_angles = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
unit_container_vertices = np.column_stack((np.cos(unit_container_angles), np.sin(unit_container_angles)))
unit_container_vectors = np.column_stack((np.cos(unit_container_angles + np.pi / nsc), np.sin(unit_container_angles + np.pi / nsc)))
unit_container_apothem = np.cos(np.pi / nsc)


@njit(cache=True)
def circle_bh_function(values, S):
    penalty = 0.0
    limit = unit_container_apothem * S - R
    for i in range(N):
        xi = values[i * 2]
        yi = values[i * 2 + 1]

        for c in range(nsc):
            dist_to_wall = xi * unit_container_vectors[c, 0] + yi * unit_container_vectors[c, 1]
            if dist_to_wall > limit:
                diff = dist_to_wall - limit
                penalty += diff * diff

    for i in range(N):
        xi = values[i * 2]
        yi = values[i * 2 + 1]
        for j in range(i + 1, N):

            dx = xi - values[j * 2]
            dy = yi - values[j * 2 + 1]
            dist_sq = dx*dx + dy*dy

            if dist_sq < (2.0 * R) ** 2:
                overlap = 2.0 * R - np.sqrt(dist_sq)
                penalty += overlap * overlap

    return penalty

def repetition(seed):
    np.random.seed(seed)
    dynamic_S = np.sqrt(N) * 2.5
    initial_S = dynamic_S
    x0 = np.random.uniform(-dynamic_S/2, dynamic_S/2, N * 2)
    last_valid_x = x0.copy()
    last_valid_S = dynamic_S
    while True:
        res = minimize(circle_bh_function, x0, args=(dynamic_S,), method="L-BFGS-B", tol=1e-8)
        multiplier = 1.0 - final_step_size - (dynamic_S - np.sqrt(N)) * (0.01 - final_step_size) / (initial_S - np.sqrt(N))
        if res.fun < penalty_tolerance:
            last_valid_x = res.x.copy()
            last_valid_S = dynamic_S
            x0 = res.x * multiplier
            dynamic_S *= multiplier
        else:
            bh_result = basinhopping(circle_bh_function, x0,minimizer_kwargs={'method': 'L-BFGS-B', 'args': (dynamic_S,), 'tol': 1e-8},niter=20, T=0.05, stepsize=0.2)
            if bh_result.fun < penalty_tolerance:
                last_valid_x = bh_result.x.copy()
                last_valid_S = dynamic_S
                x0 = bh_result.x * multiplier
                dynamic_S *= multiplier
            else:
                break
    return last_valid_S, last_valid_x

results = Parallel(n_jobs=-1)(delayed(repetition)(i) for i in range(attempts))
best_S, best_coords = min(results, key=lambda x: x[0])
fig, ax = ppt.subplots(figsize=(8, 8))
container_pts = np.vstack((unit_container_vertices * best_S, unit_container_vertices[0] * best_S))
ax.plot(container_pts[:, 0], container_pts[:, 1], color="black", lw=2)

for i in range(N):
    cx, cy = best_coords[i*2], best_coords[i*2+1]
    circ = ppt.Circle((cx, cy), R, facecolor="#95a5a6", edgecolor="#2c3e50", alpha=0.7)
    ax.add_patch(circ)

ax.set_aspect('equal')
ax.set_xlim(-best_S*1.1, best_S*1.1)
ax.set_ylim(-best_S*1.1, best_S*1.1)
ppt.title(f"Minimum Container Scale S: {best_S:.4f}")
ppt.savefig(f"{N}_in_{nsc}.png")

print(f"Final Scale S: {best_S}")
print(f"Runtime: {round(time.time() - timestart)} s")