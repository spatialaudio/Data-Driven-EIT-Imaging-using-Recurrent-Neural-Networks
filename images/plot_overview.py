import matplotlib.pyplot as plt
import numpy as np
import pyeit.mesh as mesh
from pyeit.mesh.shape import thorax

l_path = "../data_thorax/h0=0.05_n_el=16_r_min=0.1_r_max=0.4_dist_exc=8_step_meas=4noise=Truendiv=2/"
L = ["a","b","c"]
perm_dict = {}
for u,l in zip([0,25,50],L):
    tmp = np.load(l_path + "sample_{0:06d}.npz".format(u), allow_pickle=True)
    perm_dict[l] = tmp["perm_array"]

mesh_obj = mesh.create(tmp["n_el"], h0=tmp["h0"], fd=thorax)
el_pos = np.arange(16)
pts = mesh_obj.node
tri = mesh_obj.element
x, y = pts[:, 0], pts[:, 1]

fig, axs = plt.subplots(1, 3, figsize=(8, 2))
for ax,l in zip(axs,L):
    ax.tripcolor(x, y, tri, perm_dict[l], shading="flat",  cmap="cividis")
    for i, e in enumerate(el_pos):
        ax.annotate(str(i + 1), xy=(x[e], y[e]), color="r")
        ax.set_aspect("equal")
        ax.set_xlabel("test")
        ax.set_title(f"{l})")
        ax.axis('off')
plt.tight_layout()
plt.savefig("thorax_p_overview.pdf")
plt.show()