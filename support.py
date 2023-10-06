import numpy as np
import matplotlib.pyplot as plt
from pyeit.mesh import PyEITMesh
import pyeit.mesh as mesh
from sciopy import plot_mesh


def rphi_to_xy(r, angle):
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y


def get_info(tmp: np.lib.npyio.NpzFile) -> None:
    for key in ["anomaly", "n_el", "h0"]:
        print(key, ":", tmp[key])


def show_mesh(tmp: np.lib.npyio.NpzFile, return_mesh: bool = False) -> PyEITMesh:
    mesh_obj = mesh.create(tmp["n_el"], h0=tmp["h0"])
    mesh_obj = mesh.set_perm(mesh_obj, anomaly=tmp["anomaly"].tolist(), background=1.0)
    plot_mesh(mesh_obj)
    if return_mesh:
        return mesh_obj


def get_permele_diff(
    tmp_1: np.lib.npyio.NpzFile, tmp_2: np.lib.npyio.NpzFile, perm_obj: float = 100.0
) -> int:
    """The surface difference between two meshes is counted."""
    ln_diff = np.abs(
        len(np.where(tmp_1["perm_array"] == perm_obj)[0])
        - len(np.where(tmp_2["perm_array"] == perm_obj)[0])
    )
    return ln_diff


def get_permidx_diff(tmp_1: np.lib.npyio.NpzFile, tmp_2: np.lib.npyio.NpzFile) -> int:
    """The number of indices, where the permittivity is different is counted."""
    count = 0
    for i in range(len(tmp_1["perm_array"])):
        if tmp_1["perm_array"][i] != tmp_2["perm_array"][i]:
            count += 1
    return count
