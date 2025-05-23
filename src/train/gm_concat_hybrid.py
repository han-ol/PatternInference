print("\n", 5 * "-", "TRAINING CONCAT HYBRID LEARNER", 5 * "-")

import bayesflow as bf
import keras
import numpy as np

from src.networks import CNN, PlotDiagnostics
from src.simulations import get_adapter, get_expert_simulator, get_pattern_simulator, make_data_dicts_from_pickled_data

simulator = get_pattern_simulator()

batch_size = 32
training_budget = 16384  # 2^14
R_max = 32
B = 6
ptp_cutoff = 0.3
# adapter = get_adapter()
adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .keep(["parameters", "final_states_std", "expert_rdhs"])
    .constrain("parameters", lower=0)
    .rename("parameters", "inference_variables")
    .rename("final_states_std", "summary_variables")
    .rename("expert_rdhs", "inference_conditions")
    # .standardize(include="inference_variables", momentum=None, axis=0)
    .standardize(momentum=None)  # exclude=["patterns", "patterns_std"])
)
train_dict, val_dict, _ = make_data_dicts_from_pickled_data(
    training_budget=training_budget, R_max=R_max, B=B, ptp_cutoff=ptp_cutoff
)
training_dataset = bf.datasets.OfflineDataset(data=train_dict, batch_size=batch_size, adapter=adapter)
validation_dataset = bf.datasets.OfflineDataset(data=val_dict, batch_size=batch_size, adapter=adapter)
print(training_dataset)

# define approximator and optimization algorithm
# conv_params = [
#     {"num_filters": 32, "kernel_size": 3, "pool_size": 2},
#     #    {"num_filters": 64, "kernel_size": 3, "pool_size": 2},
#     {"num_filters": 64, "kernel_size": 3},
# ]

num_filters_list = [32, 64]
kernel_size_list = [3, 3]
pool_size_list = [2, None]
assert len(num_filters_list) == len(kernel_size_list) == len(pool_size_list)
conv_params = [
    dict(num_filters=num_filters_list[i], kernel_size=kernel_size_list[i], pool_size=pool_size_list[i])
    for i in range(len(num_filters_list))
]


summary_dim = 6
summary_net_dropout = inference_net_dropout = 0.0
summary_net = CNN(
    summary_dim=summary_dim,
    conv_params=conv_params,
    num_fully_connected=15,
    conv_dropout_prob=summary_net_dropout,
    dense_dropout_prob=summary_net_dropout,
    kernel_regularizer="l1l2",
)
print(summary_net)

fm_residual = True
inference_net = bf.networks.FlowMatching(subnet_kwargs={"residual": True, "dropout": inference_net_dropout})

approximator = bf.approximators.ContinuousApproximator(
    inference_network=inference_net, summary_network=summary_net, adapter=adapter
)

initial_learning_rate = 5e-4
scheduled_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate, decay_steps=epochs * training_dataset.num_batches, alpha=1e-8
)
optimizer = keras.optimizers.AdamW(learning_rate=scheduled_lr, clipnorm=1.0)
metrics = [
    keras.metrics.KLDivergence(name="kld"),
]
approximator.compile(optimizer=optimizer, inference_metrics=metrics)  # , run_eagerly=True)
print("Approximator was compiled successfully.")

# define callbacks
variable_names = np.array(["$a$", "$b$", "$c$", r"$\delta$"])
callbacks = [
    keras.callbacks.TerminateOnNaN(),
    PlotDiagnostics(
        "diagnostics", val_dict=val_dict, num_diag_obs=200, num_diag_samples=500, variable_names=variable_names
    ),
]

epochs = 15
history = approximator.fit(
    epochs=epochs,
    dataset=training_dataset,
    validation_data=validation_dataset,
    callbacks=callbacks,
)
