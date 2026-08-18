from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator
from sequince import TrainableProposal
from sequince.proposals import FlowProposal
from sequince.steps.context import StepContext

from sequince_density import (
    Flow,
    FlowJax,
    GaussianMixtureProposal,
    ZukoFlow,
)
from sequince_density.flows import (
    Flow as SubmoduleFlow,
)
from sequince_density.flows import (
    FlowJax as SubmoduleFlowJax,
)
from sequince_density.flows import (
    ZukoFlow as SubmoduleZukoFlow,
)
from sequince_density.gmm import (
    GaussianMixtureProposal as SubmoduleGaussianMixtureProposal,
)


def test_top_level_exports_match_submodule_exports() -> None:
    assert Flow is SubmoduleFlow
    assert GaussianMixtureProposal is SubmoduleGaussianMixtureProposal
    assert ZukoFlow is SubmoduleZukoFlow
    assert FlowJax is SubmoduleFlowJax


def _proposal(kind: str):
    if kind == "gmm":
        pytest.importorskip("sklearn")
        return GaussianMixtureProposal(n_components=1)
    if kind == "zuko":
        pytest.importorskip("torch")
        pytest.importorskip("zuko")
        return FlowProposal(
            ZukoFlow(dims=2, hidden_features=[8, 8]),
            fit_kwargs={
                "n_epochs": 1,
                "batch_size": 16,
                "validation_fraction": 0.25,
            },
        )
    if kind == "flowjax":
        pytest.importorskip("jax")
        pytest.importorskip("flowjax")
        return FlowProposal(
            FlowJax(
                dims=2,
                flow_layers=1,
                nn_width=8,
                nn_depth=1,
            ),
            fit_kwargs={
                "max_epochs": 1,
                "batch_size": 16,
                "show_progress": False,
            },
        )
    raise AssertionError(f"Unknown proposal kind: {kind}")


@pytest.mark.parametrize("kind", ["gmm", "zuko", "flowjax"])
def test_trainable_proposal_contract(kind: str) -> None:
    particles = np.random.default_rng(42).normal(size=(64, 2))
    context = StepContext(t=0)
    proposal = _proposal(kind)

    fitted = proposal.fit(
        particles,
        context=context,
        rng=RandomGenerator(seed=11),
    )
    samples = fitted.sample(
        5,
        context=context,
        rng=RandomGenerator(seed=12),
        xp=np,
    )
    joint_samples, joint_log_prob = fitted.sample_and_log_prob(
        5,
        context=context,
        rng=RandomGenerator(seed=13),
        xp=np,
    )
    log_prob = fitted.log_prob(samples, context=context)
    repeated_a = fitted.sample(5, rng=RandomGenerator(seed=99), xp=np)
    repeated_b = fitted.sample(5, rng=RandomGenerator(seed=99), xp=np)

    assert isinstance(fitted, TrainableProposal)
    assert fitted is proposal
    assert isinstance(samples, np.ndarray)
    assert isinstance(joint_samples, np.ndarray)
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(joint_log_prob, np.ndarray)
    assert samples.shape == (5, 2)
    assert joint_samples.shape == (5, 2)
    assert log_prob.shape == (5,)
    assert joint_log_prob.shape == (5,)
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(joint_samples))
    assert np.all(np.isfinite(log_prob))
    assert np.all(np.isfinite(joint_log_prob))
    np.testing.assert_allclose(repeated_a, repeated_b)
