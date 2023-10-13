import os

import numpy as np
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.mesh import PyEITMesh
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from tqdm import tqdm

from codes.support import rphi_to_xy

n_datapoints = 1000
n_el = 32  # number of electrodes
h0 = 0.025
r_obj = 0.3
perm_obj = 100
dist_exc = 8
step_meas = 4

s_path = f"data/{n_el=}_{r_obj=}_{dist_exc=}_{step_meas=}/"

try:
    os.mkdir(s_path)
except BaseException:
    print(f"{s_path} already exist.")


θ = np.linspace(0, 2 * np.pi, n_datapoints)
r = 0.5
X, Y = rphi_to_xy(r, θ)

# create meshes
mesh_obj = mesh.create(n_el, h0=h0)
mesh_empty = mesh.create(n_el, h0=h0)

# generate data
s_idx = 0
for x, y in tqdm(zip(X, Y)):
    anomaly = PyEITAnomaly_Circle(center=[x, y], r=r_obj, perm=perm_obj)
    mesh_obj = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    protocol_obj = protocol.create(
        n_el, dist_exc=dist_exc, step_meas=step_meas, parser_meas="std"
    )
    fwd_v = EITForward(mesh_empty, protocol_obj)
    v_empty = fwd_v.solve_eit(perm=mesh_empty.perm)
    v_obj = fwd_v.solve_eit(perm=mesh_obj.perm)

    np.savez(
        s_path + "sample_{:06d}.npz".format(s_idx),
        anomaly=anomaly,
        perm_array=mesh_obj.perm_array,
        n_el=n_el,
        h0=h0,
        v_empty=v_empty,
        v_obj=v_obj,
        dist_exc=dist_exc,
        step_meas=step_meas,
    )
    s_idx += 1
