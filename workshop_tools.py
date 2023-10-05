import asyncio
import json
import logging
import os
import shutil
import sys
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Union

import jetto_tools
import netCDF4 as nc
import numpy as np
import plotly
import plotly.graph_objects as go
import torch
from jetto_tools.config import RunConfig
from jetto_tools.results import JettoResults
from nn_surrogate import Surrogate
from plotly.subplots import make_subplots

from jetto_mobo.acquisition import generate_initial_candidates

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Load the NN surrogate model
with open("assets/nn_surrogate.json", "r") as f:
    widths = json.load(f)["layer_widths"]

nn_surrogate = Surrogate(widths)
nn_surrogate.load_state_dict(torch.load("assets/nn_surrogate.pt"))
nn_surrogate.eval()


async def run_toy(
    run_config: RunConfig,
    run_directory: Union[os.PathLike, str],
    jetto_image: Union[os.PathLike, str],
    timelimit: Optional[Union[int, float]] = None,
    container_id: Optional[str] = None,
    prepend_commands: Iterable[str] = [],
    rjettov_args: Iterable[str] = [
        "-x64",
        "-S",
        "-p0",
        "-n1",
    ],
) -> Optional[JettoResults]:
    """Toy version: run a surrogate model of JETTO, to test the ECRH-q optimisation problem

    .. warning::

        This is a toy function that is only used for illustrative examples! It does not actually run JETTO.

    .. note::

        As this is an *async* function, it must be run using ``asyncio``. For example::

            asyncio.run(
                simulation.run(
                    run_config=my_run_config,
                    run_directory=".",
                    jetto_image="jetto.sif",
                )
            )

    Parameters
    ----------
    run_config : RunConfig
        JETTO configuration to load for this run.
    run_directory : Union[os.PathLike, str]
        Path to the directory to store JETTO output files in. As this is a test function, the output files are not physically correct, but are transplanted from a template.
    jetto_image : Union[os.PathLike, str]
        Path to the JETTO .sif Singularity container image. This argument is unused, but must still be provided, to provide compatibility with the real ``simulation.run()`` function.
    timelimit : Optional[Union[int, float]], default=None
        Maximum number of seconds to wait for JETTO to complete. If ``None`` or < 0, run until complete. In this test version of the function, this timelimit is for the surrogate model, not JETTO.
    container_id : Optional[str], default=None
        ID to give the Singularity container for this run. If `None`, Singularity container will be given a new UUID. Unused.
    prepend_commands : Iterable[str], default=[]
        Commands to prepend before the `singularity` command. Can be used to launch using ``srun`` or similar. Unused.
    rjettov_args : Iterable[str], default=["-x64", "-S", "-p0", "-n1"]
        Additional arguments to pass to ``rjettov``. Unused.

    Returns
    -------
    Optional[JettoResults]
        The results of the JETTO run, or ``None`` if the run timed out or otherwise failed.
    """
    # Warn about unused arguments
    if container_id is not None:
        logger.warning(
            "container_id argument is unused in this test function, and will be ignored."
        )
    if prepend_commands:
        logger.warning(
            "prepend_commands argument is unused in this test function, and will be ignored."
        )
    if rjettov_args != ["-x64", "-S", "-p0", "-n1"]:
        logger.warning(
            "rjettov_args argument is unused in this test function, and will be ignored."
        )

    # Load the ECRH profile from the config's exfile
    exfile = jetto_tools.binary.read_binary_file(run_config.exfile)
    ecrh = torch.tensor(exfile["QECE"][0], dtype=torch.float32)
    # Predict the q profile
    q = nn_surrogate(ecrh).detach().cpu().numpy()
    ecrh = ecrh.detach().cpu().numpy()

    # Generate the profiles and timetraces CDF files
    results = JettoResults(path=run_directory)
    results.load_profiles()
    results.load_timetraces()
    # Update the profiles and timetraces
    with nc.Dataset(f"{run_directory}/profiles.CDF", "r+") as profiles:
        profiles["Q"][-1] = q
        profiles["QECE"][-1] = ecrh
    with nc.Dataset(f"{run_directory}/timetraces.CDF", "r+") as timetraces:
        timetraces["Q0"][-1] = q[0]
        timetraces["QMIN"][-1] = np.min(q)
        xrho = np.linspace(0, 1, len(q))
        timetraces["ROQM"][-1] = xrho[np.argmin(q)]

    # Return the results
    return JettoResults(path=run_directory)


async def run_many_toy(
    jetto_image: Union[os.PathLike, str],
    run_configs: Mapping[RunConfig, Union[os.PathLike, str]],
    timelimit: Optional[Union[int, float]] = None,
) -> Iterable[Optional[JettoResults]]:
    """Asynchronously run multiple JETTO runs, using ``jetto_mobo.simulation.run()``.

    .. note::

        As this is an *async* function, it must be run using ``asyncio``. For example::

            asyncio.run(
                simulation.run_many(
                    jetto_image=jetto_image, run_configs=configs, timelimit=jetto_timelimit
                )
            )


    Parameters
    ----------
    jetto_image : Union[os.PathLike, str]
        Path to the JETTO .sif Singularity container image.
    run_configs : Mapping[RunConfig, Union[os.PathLike, str]]
        A mapping from JETTO configurations to the directories to store their output files in, such as
        ``{config1: "/home/runs/1", config2: "/home/runs/run2"}``. The directories will be created if they don't exist.
    timelimit : Optional[Union[int, float]], default = None
        Maximum number of seconds to wait for JETTO to complete. If ``None``, run until complete.

    Returns
    -------
    Iterable[Optional[JettoResults]]
        The results of each JETTO run, or ``None`` if the run timed out or otherwise failed.
    """
    return await asyncio.gather(
        *[
            run_toy(
                run_config=run_config,
                run_directory=run_directory,
                timelimit=timelimit,
            )
            for run_config, run_directory in run_configs.items()
        ]
    )


def create_config_toy(
    template: Union[PathLike, str], directory: Union[PathLike, str]
) -> jetto_tools.config.RunConfig:
    template = Path(template)
    directory = Path(directory)

    # Copy the template as-is
    shutil.copytree(template, directory, dirs_exist_ok=True)

    # Create a config
    template_obj = jetto_tools.template.from_directory(template)
    config = jetto_tools.config.RunConfig(template_obj)
    config.exfile = directory / "jetto.ex"
    return config


def plot_profiles(profiles: nc.Dataset) -> None:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True)
    ecrh = profiles["QECE"][-1]
    q = profiles["Q"][-1]
    figure.add_traces(
        [
            go.Scatter(
                x=np.linspace(0, 1, len(ecrh)),
                y=ecrh,
                line_color=plotly.colors.DEFAULT_PLOTLY_COLORS[0],
                showlegend=False,
            ),
            go.Scatter(
                x=np.linspace(0, 1, len(q)),
                y=q,
                line_color=plotly.colors.DEFAULT_PLOTLY_COLORS[0],
                showlegend=False,
            ),
        ],
        rows=[1, 2],
        cols=[1, 1],
    )
    figure.update_xaxes(title_text="Normalised radius", range=[0, 1], row=2, col=1)
    figure.update_yaxes(title_text="Normalised QECE", row=1, col=1)
    figure.update_yaxes(title_text="q", row=2, col=1)
    figure.update_layout(template="simple_white")
    figure.show()


def plot_ecrh_profile(
    ecrh_profile_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bounds: np.ndarray,
    n: int = 4,
) -> None:
    """Visualise the ECRH profile function.

    Parameters
    ----------
    ecrh_profile_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        The ECRH profile function to visualise; must be decorated with ``@jetto_mobo.inputs.plasma_profile``
    bounds : np.ndarray
        A (2, M) array of the lower and upper bounds of the M parameters of the ECRH profile function.
    n : int, default=4
        The number of examples to plot.
    """
    figure = go.Figure()
    x = np.linspace(0, 1, int(1e3))
    parameters = (
        generate_initial_candidates(torch.tensor(bounds, dtype=torch.float32), n)
        .detach()
        .cpu()
        .numpy()
    )
    figure.add_traces(
        [
            go.Scatter(
                x=x,
                y=ecrh_profile_function(x, parameters[i, :]),
                name=f"Example {i + 1}",
            )
            for i in range(n)
        ],
    )
    figure.update_xaxes(title_text="Normalised radius", range=[0, 1])
    figure.update_yaxes(title_text="Normalised QECE", range=[0, 1])
    figure.update_layout(template="simple_white")
    figure.show()


def plot_results(
    ecrh: np.ndarray,
    q: np.ndarray,
    objective_values: np.ndarray,
    objective_labels: Iterable[str],
) -> None:
    """Plot the ECRH profile, q profile, and objective values for each solution."""
    n_solutions = ecrh.shape[0]

    figure = make_subplots(
        rows=n_solutions,
        cols=3,
        specs=[[{}, {}, {"type": "polar"}]] * n_solutions,
    )
    for i in range(n_solutions):
        figure.add_traces(
            [
                go.Scatterpolar(
                    r=np.concatenate(
                        [objective_values[i], [objective_values[i][0]]]
                    ),  # Repeat first point to create closed shape
                    theta=objective_labels + [objective_labels[0]],
                    name=i,
                    showlegend=True,
                    line_color=plotly.colors.DEFAULT_PLOTLY_COLORS,
                ),
                go.Scatter(
                    x=np.linspace(0, 1, len(ecrh[i])),
                    y=ecrh[i],
                    name=i,
                    showlegend=False,
                    mode="lines",
                    line_color=plotly.colors.DEFAULT_PLOTLY_COLORS,
                ),
                go.Scatter(
                    x=np.linspace(0, 1, len(q[i])),
                    y=q[i],
                    name=i,
                    showlegend=False,
                    mode="lines",
                    line_color=plotly.colors.DEFAULT_PLOTLY_COLORS,
                ),
                go.Scatter(
                    x=[np.linspace(0, 1, len(q))[np.argmin(q[i])]],
                    y=[np.min(q[i])],
                    name=f"{i} - minimum",
                    mode="markers",
                    line_color=plotly.colors.DEFAULT_PLOTLY_COLORS,
                    showlegend=False,
                ),
            ],
            rows=[1, 1, 2, 2],
            cols=[2, 1, 1, 1],
        )
        figure.update_yaxes(title_text="QECE", row=i, col=1)
        figure.update_xaxes(title_text="Normalised radius", range=[0, 1], row=i, col=1)
        figure.update_yaxes(title_text="q", row=i, col=1)
        figure.update_xaxes(title_text="Normalised radius", range=[0, 1], row=i, col=2)
    figure.update_layout(template="simple_white")
    figure.show()


if __name__ == "__main__":
    from jetto_mobo.simulation import create_config

    test_config = create_config(
        template="../../jetto/templates/spr45", directory="test"
    )

    results = asyncio.run(
        run_toy(
            run_config=test_config,
            run_directory="test",
            jetto_image="../../jetto/images/sim.v220922.sif",
        )
    )
    profiles = results.load_profiles()
    plot_profiles(profiles)
