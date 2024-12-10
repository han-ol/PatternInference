import os
import pickle

import bayesflow as bf
import numpy as np
from tqdm import tqdm

import src.experts as exp
import src.simulations.online_simulator as sim_online

sim_rds = bf.make_simulator(sim_online.subsampled_resistance)

DATA_PATH = "data"
slice_files = [
    elem for elem in sorted(os.listdir(DATA_PATH)) if "GM-001-slice" in elem and "rds" not in elem and "train" in elem
]
slice_files

for slice_file_in in slice_files:
    with open(os.path.join(DATA_PATH, slice_file_in), "rb") as f:
        data_dict = pickle.load(f)

    slice_length = data_dict["parameters"].shape[0]

    slice_file_out = "{0}-rds{1}".format(*os.path.splitext(slice_file_in))

    print(slice_file_out, slice_length)
    batch_size = 500

    for i in (pbar := tqdm(range(11500, slice_length, batch_size))):
        idx_batch = slice(i, min(i + batch_size, slice_length))
        data_dict_batch = {key: value[idx_batch] for key, value in data_dict.items()}
        batch_length = data_dict_batch["parameters"].shape[0]

        root, ext = os.path.splitext(slice_file_out)
        slice_file_out_batch = f"{root}-{idx_batch.start:0>5d}-{idx_batch.stop:0>5d}{ext}"
        pbar.set_description(slice_file_out_batch)

        rds_dict = sim_rds.sample(batch_length, **data_dict_batch)

        with open(os.path.join(DATA_PATH, slice_file_out_batch), "wb") as f:
            pickle.dump(rds_dict, f)
