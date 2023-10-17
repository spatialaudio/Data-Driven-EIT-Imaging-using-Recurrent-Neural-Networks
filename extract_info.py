import os

import numpy as np
from tqdm import tqdm
from sciopy import norm_data
from codes.support import (
    get_permele_diff,
    get_permidx_diff,
)

# to save the plots, set this to true
save_imgs = True
# Continue with n_el=32
n_el = 32
r_obj = 0.3
h0 = 0.025
dist_exc = 8
step_meas = 4

load_path = f"data/{h0=}_{n_el=}_{r_obj=}_{dist_exc=}_{step_meas=}/"

print(f"{load_path=}")

if os.path.isdir(load_path):
    print("This directory exists...continue")
else:
    print("You have to generate the data...")

if (
    os.path.isfile("saves/" + load_path.split("/")[1] + "_diff_cnt.npy")
    == False
):
    diff_cnt = list()
    diff_idx = list()

    for i in tqdm(range(len(os.listdir(load_path)) - 1)):
        tmp_1 = np.load(
            load_path + "sample_{:06d}.npz".format(i),
            allow_pickle=True,
        )
        tmp_2 = np.load(
            load_path + "sample_{:06d}.npz".format(i + 1),
            allow_pickle=True,
        )

        diff_cnt.append(get_permele_diff(tmp_1, tmp_2))
        diff_idx.append(get_permidx_diff(tmp_1, tmp_2))

    diff_cnt = np.array(diff_cnt)
    diff_idx = np.array(diff_idx)

    np.save("saves/" + load_path.split("/")[1] + "_diff_cnt.npy", diff_cnt)
    np.save("saves/" + load_path.split("/")[1] + "_diff_idx.npy", diff_idx)
else:
    diff_cnt = np.load(
        "saves/" + load_path.split("/")[1] + "_diff_cnt.npy",
        allow_pickle=True,
    )
    diff_idx = np.load(
        "saves/" + load_path.split("/")[1] + "_diff_idx.npy",
        allow_pickle=True,
    )

if os.path.isfile("saves/" + load_path.split("/")[1] + "_v_data.npy") == False:
    v_data = list()

    for i in tqdm(range(len(os.listdir(load_path)) - 1)):
        tmp_1 = np.load(load_path + "sample_{:06d}.npz".format(i), allow_pickle=True)
        v_data.append(norm_data(tmp_1["v_obj"]))
    v_data = np.array(v_data)

    ae_v = list()
    for i in range(v_data.shape[0] - 1):
        ae_v.append(np.sum(np.abs(v_data[i, :] - v_data[i + 1, :])))
    ae_v = np.array(ae_v)

    np.save("saves/" + load_path.split("/")[1] + "_v_data.npy", v_data)
    np.save("saves/" + load_path.split("/")[1] + "_ae_v.npy", ae_v)
else:
    v_data = np.load(
        "saves/" + load_path.split("/")[1] + "_v_data.npy", allow_pickle=True
    )
    ae_v = np.load("saves/" + load_path.split("/")[1] + "_ae_v.npy", allow_pickle=True)
    
print("Done...")