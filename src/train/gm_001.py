import bayesflow as bf
import keras
import numpy as np

from src.networks import CNN, PlotDiagnostics
from src.simulations import get_adapter, get_pattern_simulator, make_data_dicts_from_pickled_data

condition_type = "concat_hybrid"

print("\n", 5 * "-", f"TRAINING {condition_type.upper()} LEARNER", 5 * "-")

assert condition_type in ["concat_hybrid", "pure_learner", "pure_expert"]
has_summary_net = condition_type in ["concat_hybrid", "pure_learner"]

simulator = get_pattern_simulator()

seed = 1234
keras.utils.set_random_seed(seed)

epochs = 15
batch_size = 32
training_budget = 16384  # 2^14
R_max = 32
B = 6
ptp_cutoff = 0.3
# adapter = get_adapter()
adapter = get_adapter(condition_type)
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
    dict(
        num_filters=num_filters_list[i],
        kernel_size=kernel_size_list[i],
        pool_size=pool_size_list[i],
    )
    for i in range(len(num_filters_list))
]

summary_dim = 6
summary_net_dropout = inference_net_dropout = 0.0
if has_summary_net:
    summary_net = CNN(
        summary_dim=summary_dim,
        conv_params=conv_params,
        num_fully_connected=15,
        conv_dropout_prob=summary_net_dropout,
        dense_dropout_prob=summary_net_dropout,
        kernel_regularizer="l1l2",
    )
else:
    summary_net = None

fm_residual = True
inference_network_name = "cf"
match inference_network_name:
    case "cf":
        inference_net = bf.networks.CouplingFlow(subnet_kwargs={"dropout": inference_net_dropout})
    case "fm":
        inference_net = bf.networks.FlowMatching(
            subnet_kwargs={"residual": fm_residual, "dropout": inference_net_dropout}
        )


approximator = bf.approximators.ContinuousApproximator(
    inference_network=inference_net, summary_network=summary_net, adapter=adapter
)

initial_learning_rate = 5e-4
scheduled_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=epochs * training_dataset.num_batches,
    alpha=1e-8,
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
        "diagnostics",
        val_dict=val_dict,
        num_diag_obs=200,
        num_diag_samples=500,
        variable_names=variable_names,
    ),
]

history = approximator.fit(
    epochs=epochs,
    dataset=training_dataset,
    validation_data=validation_dataset,
    callbacks=callbacks,
)
