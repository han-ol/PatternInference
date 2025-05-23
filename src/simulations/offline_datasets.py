import os
import pickle
from functools import partial

import bayesflow as bf
import numpy as np

from .online_simulator import make_rdhistogram_expert_func


def get_individual_sim_data_std(data_dict):
    "Adds a pattern standardized to stretch between -1 and 1 as a key value pair to `data_dict`."
    individual_minimum = np.min(
        data_dict["final_states"],
        axis=(
            1,
            2,
        ),
        keepdims=True,
    )
    individual_maximum = np.max(
        data_dict["final_states"],
        axis=(
            1,
            2,
        ),
        keepdims=True,
    )
    individual_ptp = individual_maximum - individual_minimum
    final_states_std = ((data_dict["final_states"] - individual_minimum) / individual_ptp - 0.5) * 2

    return dict(final_states_std=final_states_std)


def get_adapter():
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .keep(["parameters", "final_states_std"])
        .constrain("parameters", lower=0)
        .rename("parameters", "inference_variables")
        .rename("final_states_std", "summary_variables")
        # .standardize(include="inference_variables", momentum=None, axis=0)
        .standardize(momentum=None)  # exclude=["patterns", "patterns_std"])
    )
    return adapter


def get_expert_rdh(slice_path, R_max, B, budget=None):
    sim_hist = bf.make_simulator(make_rdhistogram_expert_func(R_max, B))

    # get list of expert batches
    data_path, slice_filename = os.path.split(slice_path)
    batch_filenames = [
        elem for elem in sorted(os.listdir(data_path)) if os.path.splitext(slice_filename)[0] in elem and "rds" in elem
    ]

    # loop over batches (until >=budget), compute histgram and append to fresh dict(rdh=)
    rdh_list = []
    loaded_rdhs = 0
    for fname in batch_filenames:
        with open(os.path.join(data_path, fname), "rb") as f:
            rd_dict = pickle.load(f)

        batch_size = rd_dict["subsampled_resistance_distances"].shape[0]
        histograms = sim_hist.sample(batch_size, **rd_dict)
        rdh_list.append(histograms)

        loaded_rdhs += batch_size

        if budget and loaded_rdhs >= budget:
            break

    hist_dict = {}
    for d in rdh_list:
        for k, v in d.items():
            hist_dict[k] = np.concatenate([hist_dict[k], v], axis=0) if k in hist_dict.keys() else v

    hist_dict = {k: v[:budget, ...] for k, v in hist_dict.items()}
    return hist_dict


def ptp_mask(sim_dict, ptp_cutoff):
    pattern_key = "final_states"
    pattern_ptp = np.ptp(sim_dict[pattern_key], axis=(1, 2))
    mask = pattern_ptp > ptp_cutoff
    return {key: value[mask] for key, value in sim_dict.items()}


def make_data_dicts_from_pickled_data(
    data_path="/home/ho/code/PatternInference/data/",
    training_budget=None,
    R_max=None,
    B=None,
    ptp_cutoff=0.0,
):
    sims_path_train = os.path.join(data_path, "GM-001-slice-04000-20384-train.pkl")
    with open(sims_path_train, "rb") as f:
        train_dict = pickle.load(f)
        train_dict["parameters"] = train_dict["parameters"][:, :4]
        train_dict |= get_individual_sim_data_std(train_dict)
        if not (R_max == B == None):
            train_dict |= get_expert_rdh(sims_path_train, R_max=R_max, B=B)

        train_dict = ptp_mask(train_dict, ptp_cutoff)
        train_dict = {key: value[:training_budget, ...] for key, value in train_dict.items()}

    sims_path_val = os.path.join(data_path, "GM-001-slice-00000-02000-validation.pkl")
    with open(sims_path_val, "rb") as f:
        val_dict = pickle.load(f)
        val_dict["parameters"] = val_dict["parameters"][:, :4]
        val_dict |= get_individual_sim_data_std(val_dict)
        if not (R_max == B == None):
            val_dict |= get_expert_rdh(sims_path_val, R_max=R_max, B=B)

        val_dict = ptp_mask(val_dict, ptp_cutoff)

    for new_data_dict in (train_dict, val_dict):
        for key, value in new_data_dict.items():
            print(f"{key:20s} shape={str(value.shape):18s} min={np.min(value)}")
        print()

    test_dict = None
    return train_dict, val_dict, test_dict
