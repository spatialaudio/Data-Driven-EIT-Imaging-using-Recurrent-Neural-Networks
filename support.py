import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.greit as greit
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh import PyEITMesh
from sciopy import plot_mesh


def rphi_to_xy(r, angle):
    """
    Convert cylinder coordinates to cartesian coordinates.

    Parameters
    ----------
    r : floar
        radius
    angle : int
        angle

    Returns
    -------
    tuple
        x,y coordinates
    """
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y


def get_info(tmp: np.lib.npyio.NpzFile) -> None:
    """
    Get general information about a sample.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        loaded sample by np.load()
    """
    for key in ["anomaly", "n_el", "h0"]:
        print(key, ":", tmp[key])


def show_mesh(tmp: np.lib.npyio.NpzFile, return_mesh: bool = False) -> PyEITMesh:
    """
    Show the mesh of a single sample.
    Return the mesh if return_mesh==True.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        loaded sample by np.load()
    return_mesh : bool, optional
        return PyEITMesh object, by default False

    Returns
    -------
    PyEITMesh
        PyEITMesh object
    """
    mesh_obj = mesh.create(tmp["n_el"], h0=tmp["h0"])
    mesh_obj = mesh.set_perm(mesh_obj, anomaly=tmp["anomaly"].tolist(), background=1.0)
    plot_mesh(mesh_obj)
    if return_mesh:
        return mesh_obj


def get_permele_diff(
    tmp_1: np.lib.npyio.NpzFile,
    tmp_2: np.lib.npyio.NpzFile,
    perm_obj: float = 100.0,
) -> int:
    """
    The surface element difference between two FEM meshes is counted.

    Parameters
    ----------
    tmp_1 : np.lib.npyio.NpzFile
        loaded sample by np.load()
    tmp_2 : np.lib.npyio.NpzFile
        loaded sample by np.load()
    perm_obj : float, optional
        object permittivity, by default 100.0

    Returns
    -------
    int
        number of deviating mesh elements
    """
    ln_diff = np.abs(
        len(np.where(tmp_1["perm_array"] == perm_obj)[0])
        - len(np.where(tmp_2["perm_array"] == perm_obj)[0])
    )
    return ln_diff


def get_permidx_diff(tmp_1: np.lib.npyio.NpzFile, tmp_2: np.lib.npyio.NpzFile) -> int:
    """
    The number of indices, where the permittivity is different is counted.

    Parameters
    ----------
    tmp_1 : np.lib.npyio.NpzFile
        loaded sample by np.load()
    tmp_2 : np.lib.npyio.NpzFile
        loaded sample by np.load()

    Returns
    -------
    int
        number of deviating mesh elements indices
    """
    count = 0
    for i in range(len(tmp_1["perm_array"])):
        if tmp_1["perm_array"][i] != tmp_2["perm_array"][i]:
            count += 1
    return count


def GREIT_sample(
    tmp: np.lib.npyio.NpzFile,
    rec_only: bool = False,
) -> None:
    """
    Plot the numerical reconstruction of a single sample using the GREIT algorithm.

    Parameters
    ----------
    tmp : np.lib.npyio.NpzFile
        loaded sample by np.load()
    rec_only : bool, optional
        show only the GREIT reconstruction, by default False
    """
    dist_exc = int(tmp["dist_exc"])
    step_meas = int(tmp["step_meas"])
    n_el = int(tmp["n_el"])
    mesh_obj = mesh.create(n_el, h0=float(tmp["h0"]))
    mesh_new = mesh.set_perm(mesh_obj, anomaly=tmp["anomaly"].tolist(), background=1.0)
    protocol_obj = protocol.create(
        n_el, dist_exc=dist_exc, step_meas=step_meas, parser_meas="std"
    )

    pts = mesh_obj.node
    tri = mesh_obj.element

    v0 = tmp["v_empty"]
    v1 = tmp["v_obj"]
    delta_perm = np.real(mesh_new.perm - mesh_obj.perm)

    eit = greit.GREIT(mesh_obj, protocol_obj)
    eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    _, _, ds = eit.mask_value(ds, mask_value=np.NAN)

    if rec_only:
        plt.figure(figsize=(6, 6))
        im = plt.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)

        plt.colorbar(im)
        plt.show()
    else:
        fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))
        ax = axes[0]
        im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")
        ax.axis("equal")
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_title(r"$\Delta$ Conductivity")
        ax = axes[1]
        im = ax.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)
        ax.axis("equal")
        ax.set_title(r"GREIT Reconstruction")
        fig.colorbar(im, ax=axes.ravel().tolist())
        # fig.savefig('images/_greit.png', dpi=96)
        plt.show()
