import os

import bayesflow as bf
import keras


class PlotDiagnostics(keras.callbacks.Callback):
    def __init__(
        self,
        name,
        val_dict,
        num_diag_obs,
        num_diag_samples,
        variable_names,
        figure_dir="figures",
    ):
        self.name = name
        self.val_dict = val_dict
        self.num_diag_obs = num_diag_obs
        self.num_diag_samples = num_diag_samples
        self.variable_names = variable_names
        self.figure_dir = figure_dir
        os.makedirs(self.figure_dir, exist_ok=True)

    def on_train_end(self, logs=None):
        print("plot_diagnostics")

        conditions = {k: v[: self.num_diag_obs, ...] for k, v in self.val_dict.items() if k != "parameters"}

        pdraws = self.model.sample(conditions=conditions, num_samples=self.num_diag_samples)
        conditions |= pdraws

        f = bf.diagnostics.plots.recovery(
            pdraws["parameters"],
            self.val_dict["parameters"][: self.num_diag_obs, ...],
            variable_names=self.variable_names,
        )
        f = bf.diagnostics.plots.calibration_histogram(
            pdraws["parameters"],
            self.val_dict["parameters"][: self.num_diag_obs, ...],
            variable_names=self.variable_names,
        )
        f = bf.diagnostics.plots.calibration_ecdf(
            pdraws["parameters"],
            self.val_dict["parameters"][: self.num_diag_obs, ...],
            difference=True,
            variable_names=self.variable_names,
        )
        # f = bf.diagnostics.plots.z_score_contraction(
        #     pdraws["parameters"], self.val_dict["parameters"][:self.num_diag_obs, ...], variable_names=self.variable_names
        # )

        plot_path = f"{self.figure_dir}/{self.name}_calibration_ecdf.png"
        f.savefig(plot_path)
