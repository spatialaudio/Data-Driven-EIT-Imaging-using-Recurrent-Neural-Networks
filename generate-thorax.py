import os

import numpy as np
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.mesh import PyEITMesh
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.mesh.shape import thorax
from tqdm import tqdm

from codes.support import rphi_to_xy

n_diff_noises = 100
r_min = 0.1 # min lung radius
r_max = 0.4 # max lung radius
splts = 51 # splts*2-1 is the number or radial steps

print(f"num of samples = {n_diff_noises*(splts*2-2)}")

noise_div = 200

n_el = 16  # number of electrodes
h0 = 0.025
perm_obj = 10.0
dist_exc = 8
step_meas = 4
noise = True

r_up = np.linspace(r_min, r_max, splts-1, endpoint=False)
r_down = np.linspace(r_max, r_min, splts-1, endpoint=False)

r_period = np.concatenate((r_up,r_down))

s_path = f"data_thorax/{h0=}_{n_el=}_{r_min=}_{r_max=}_{dist_exc=}_{step_meas=}{noise=}/"

try:
    os.mkdir(s_path)
except BaseException:
    print(f"{s_path} already exist.")

# create meshes
mesh_obj = mesh.create(n_el, h0=h0, fd=thorax)
mesh_empty = mesh.create(n_el, h0=h0, fd=thorax)

s_idx = 0
for _ in range(n_diff_noises):
    # create noise vector
    r_noise = np.random.random(splts*2-2)/noise_div
    for lung_r in tqdm(r_period+r_noise):
        # generate data
        lung_anomaly_r = PyEITAnomaly_Circle(center=[0.5, 0], r=lung_r, perm=perm_obj)
        lung_anomaly_l = PyEITAnomaly_Circle(center=[-0.45, 0], r=lung_r, perm=perm_obj)

        mesh_obj = mesh.set_perm(
            mesh_obj, anomaly=[lung_anomaly_l, lung_anomaly_r], background=1.0
        )
        if noise:
            mesh_obj.perm = mesh_obj.perm + np.random.rand(len(mesh_obj.perm_array))
        protocol_obj = protocol.create(
            n_el, dist_exc=dist_exc, step_meas=step_meas, parser_meas="std"
        )
        fwd_v = EITForward(mesh_empty, protocol_obj)
        v_empty = fwd_v.solve_eit(perm=mesh_empty.perm)
        v_obj = fwd_v.solve_eit(perm=mesh_obj.perm)

        np.savez(
            s_path + "sample_{:06d}.npz".format(s_idx),
            anomaly=[lung_anomaly_l, lung_anomaly_r],
            perm_array=mesh_obj.perm,
            n_el=n_el,
            h0=h0,
            v_empty=v_empty,
            v_obj=v_obj,
            dist_exc=dist_exc,
            step_meas=step_meas,
        )
        s_idx += 1
