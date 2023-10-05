import json
from typing import Iterable

import torch
import torch.nn as nn


class Surrogate(nn.Module):
    def __init__(self, layer_widths: Iterable[int]) -> None:
        super().__init__()
        self.network = nn.Sequential(
            *[
                self._create_layer(input_width, output_width)
                for input_width, output_width in zip(
                    layer_widths[:-1], layer_widths[1:]
                )
            ],
        )

    def _create_layer(self, input_width: int, output_width: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_width, output_width),
            nn.ReLU(),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.network(x)


if __name__ == "__main__":
    import os

    import h5py
    import numpy as np
    import tqdm

    # Load data
    print("Loading data...")
    q = []
    ecrh = []
    for stem in [
        "../../publications/paper/scripts/sobol_sampling/data",
        "../../examples/ecrh_q_optimisation/data/vector",
    ]:
        directories = os.listdir(stem)
        files = [
            f"{stem}/{directory}/results.h5"
            for directory in directories
            if "results.h5" in os.listdir(f"{stem}/{directory}")
        ]

        for filename in files:
            print(f"loading from {filename}")
            with h5py.File(filename, "r") as h5file:
                if "converged_q" in h5file.keys():
                    q.append(h5file["converged_q"][:])
                    ecrh.append(h5file["preconverged_ecrh"][:])
                elif "optimisation_step_1" in h5file.keys():
                    for i in range(len(h5file.keys())):
                        if i == 0:
                            q.append(h5file[f"initialisation/converged_q"][:])
                            ecrh.append(h5file[f"initialisation/preconverged_ecrh"][:])
                        else:
                            if "converged_q" in h5file[f"optimisation_step_{i}"].keys():
                                q.append(
                                    h5file[f"optimisation_step_{i}/converged_q"][:]
                                )
                                ecrh.append(
                                    h5file[f"optimisation_step_{i}/preconverged_ecrh"][
                                        :
                                    ]
                                )

    q = torch.tensor(np.concatenate(q), dtype=torch.float32)
    ecrh = torch.tensor(np.concatenate(ecrh), dtype=torch.float32)

    # Bin any data that has q = 0 everywhere
    bad_indices = q.sum(axis=1) == 0
    q = q[~bad_indices]
    ecrh = ecrh[~bad_indices]

    print(f"Loaded {q.shape[0]} input/output profile pairs.")

    # Initialise model
    widths = [150, 150, 150, 150]
    with open("nn_surrogate.json", "w") as f:
        json.dump({"layer_widths": widths}, f)

    surrogate = Surrogate(widths)

    # Train based on reconstruction loss
    print("Training...")
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    for i in tqdm.tqdm(range(int(2e5))):
        # Sample a batch
        batch_size = 16
        indices = np.random.choice(len(q), batch_size)

        optimizer.zero_grad()
        predicted_q = surrogate(ecrh[indices])
        loss = loss_fn(predicted_q, q[indices])
        if loss < 1e-3:
            break
        loss.backward()
        optimizer.step()

        if i % 1e3 == 0:
            print(loss)

    print("Done.")

    # Save model
    print("Saving model...")
    torch.save(surrogate.state_dict(), "nn_surrogate.pt")
    print("Done.")

    # Demonstrate reconstruction
    import plotly.graph_objects as go
    from plotly.colors import DEFAULT_PLOTLY_COLORS

    figure = go.Figure()
    for i in range(5):
        predicted_q = surrogate(ecrh[i])

        figure.add_traces(
            [
                go.Scatter(
                    y=q[i],
                    name="original",
                    line_color=DEFAULT_PLOTLY_COLORS[i],
                    line_dash="dash",
                    legendgroup=i,
                ),
                go.Scatter(
                    y=predicted_q.detach().numpy(),
                    name="reconstructed",
                    line_color=DEFAULT_PLOTLY_COLORS[i],
                    legendgroup=i,
                ),
            ]
        )
    figure.show(renderer="browser")
