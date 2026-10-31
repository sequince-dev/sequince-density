from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from orng import RandomGenerator
from sequince.proposals import FlowProposal
from sequince.transforms import (
    AffineTransform,
    CompositeTransform,
    FlowTransform,
    ParticleTransform,
)

pytest.importorskip("torch")
pytest.importorskip("zuko")

import torch  # noqa: E402

from sequince_density.flows.zuko import ZukoFlow  # noqa: E402


@pytest.fixture
def fitted_flow() -> ZukoFlow:
    particles = np.random.default_rng(21).normal(size=(64, 2))
    particles = particles * [2.0, 3.0] + [4.0, -2.0]
    flow = ZukoFlow(
        dims=2,
        hidden_features=[8, 8],
        particle_transform=AffineTransform(),
        seed=22,
        device="cpu",
        dtype=torch.float32,
    )
    return flow.fit(particles, n_epochs=2, batch_size=16, rng=23)


def test_zuko_flow_transform_fits_and_round_trips() -> None:
    rng = np.random.default_rng(1234)
    particles = rng.normal(size=(64, 2))
    flow = ZukoFlow(
        dims=2,
        hidden_features=[8, 8],
        device="cpu",
    )

    transform = CompositeTransform(
        [
            AffineTransform(),
            FlowTransform(
                flow,
                fit_kwargs={
                    "n_epochs": 2,
                    "batch_size": 16,
                    "validation_fraction": 0.25,
                },
            ),
        ]
    )
    transform_rng = RandomGenerator(seed=19)
    transform = transform.fit(particles, rng=transform_rng)
    latent, log_det, auxiliary = transform.forward(
        particles, rng=transform_rng
    )
    restored, inverse_log_det = transform.inverse(
        latent, auxiliary=auxiliary, rng=transform_rng
    )

    assert isinstance(latent, np.ndarray)
    assert latent.shape == particles.shape
    assert log_det.shape == (particles.shape[0],)
    np.testing.assert_allclose(restored, particles, atol=1e-5)
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0, atol=1e-5)


@pytest.mark.parametrize("n_samples", [1, 8])
def test_sample_and_log_prob_matches_log_prob(
    fitted_flow: ZukoFlow, n_samples: int
) -> None:
    samples, log_prob = fitted_flow.sample_and_log_prob(n_samples, rng=24)

    assert samples.shape == (n_samples, 2)
    assert log_prob.shape == (n_samples,)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(log_prob).all()
    # Inverting the float32 flow introduces round-off in density evaluation.
    torch.testing.assert_close(
        log_prob, fitted_flow.log_prob(samples), rtol=1e-5, atol=1e-5
    )


def test_checkpoint_restores_fitted_density_and_seeded_samples(
    fitted_flow: ZukoFlow, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "flow.pkl"
    checkpoint.write_bytes(pickle.dumps(fitted_flow.state_dict()))
    restored = ZukoFlow(
        dims=2,
        hidden_features=[8, 8],
        seed=99,
        device="cpu",
        dtype=torch.float32,
    )
    restored.load_state_dict(pickle.loads(checkpoint.read_bytes()))

    points = torch.tensor([[4.0, -2.0], [6.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(
        restored.log_prob(points), fitted_flow.log_prob(points), rtol=0, atol=0
    )
    for seed in (25, 26):
        torch.testing.assert_close(
            restored.sample(8, rng=seed),
            fitted_flow.sample(8, rng=seed),
            rtol=0,
            atol=0,
        )
    assert fitted_flow.fit_history["training_loss"]
    assert fitted_flow.fit_history["validation_loss"]
    assert restored.fit_history == fitted_flow.fit_history


def test_rejects_periodic_particle_transform(
    periodic_transform: ParticleTransform,
) -> None:
    with pytest.raises(ValueError, match="PeriodicBoundaryTransform"):
        ZukoFlow(dims=2, particle_transform=periodic_transform)


def test_checkpoint_rejects_periodic_particle_transform(
    periodic_transform: ParticleTransform,
) -> None:
    source = ZukoFlow(dims=2, hidden_features=[8, 8], seed=22)
    flow = ZukoFlow(dims=2, hidden_features=[8, 8], seed=99)
    original_transform = flow.particle_transform
    points = torch.tensor([[0.5, -0.2], [4.0, 1.0]])
    original_log_prob = flow.log_prob(points).detach().clone()
    state = source.state_dict()
    state["particle_transform"] = periodic_transform

    with pytest.raises(ValueError, match="PeriodicBoundaryTransform"):
        flow.load_state_dict(state)

    assert flow.particle_transform is original_transform
    torch.testing.assert_close(
        flow.log_prob(points), original_log_prob, rtol=0, atol=0
    )


class TrainableGaussian(torch.nn.Module):
    """Unit-variance Gaussian with an observable, trainable location."""

    def __init__(self, dims: int, context: int = 0) -> None:
        super().__init__()
        # Separate parameters distinguish total-norm from per-parameter clipping.
        self.coordinates = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(())) for _ in range(dims)]
        )
        self.validation_locations: list[torch.Tensor] = []
        self.gradient_norms: list[float] = []

    @property
    def loc(self) -> torch.Tensor:
        return torch.stack(list(self.coordinates))

    def forward(self) -> torch.distributions.Distribution:
        if not self.training:
            self.validation_locations.append(self.loc.detach().clone())
            gradients = torch.stack([p.grad for p in self.coordinates])
            self.gradient_norms.append(float(gradients.norm()))
        return torch.distributions.Independent(
            torch.distributions.Normal(self.loc, torch.ones_like(self.loc)), 1
        )


def gaussian_flow() -> ZukoFlow:
    return ZukoFlow(
        dims=2, flow_class=TrainableGaussian, dtype=torch.float64, device="cpu"
    )


@pytest.mark.parametrize("clip_grad", [None, 0.25])
def test_gradient_clipping_bounds_total_norm(clip_grad: float | None) -> None:
    flow = gaussian_flow()
    flow.fit(
        torch.tensor([[5.0, -12.0]]).repeat(8, 1),
        n_epochs=4,
        lr=0.1,
        batch_size=8,
        validation_fraction=0,
        clip_grad=clip_grad,
        rng=42,
    )
    norms = flow.flow.gradient_norms
    assert len(norms) == 4
    if clip_grad is None:
        assert min(norms) > 10
    else:
        assert all(0 < norm <= clip_grad for norm in norms)
    assert not torch.equal(flow.flow.loc, torch.zeros_like(flow.flow.loc))


@pytest.mark.parametrize("lr_annealing", [False, True])
def test_annealing_matches_adam_updates(lr_annealing: bool) -> None:
    # With one batch per epoch, the Gaussian's gradient is loc - target.
    target = torch.tensor([10.0, -10.0], dtype=torch.float64)
    flow = gaussian_flow()
    flow.fit(
        target.repeat(8, 1),
        n_epochs=5,
        lr=0.1,
        batch_size=8,
        validation_fraction=0,
        lr_annealing=lr_annealing,
        rng=42,
    )
    loc = torch.zeros_like(target)
    first = torch.zeros_like(target)
    second = torch.zeros_like(target)
    for epoch, actual in enumerate(flow.flow.validation_locations):
        grad = loc - target
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad.square()
        rate = 0.1
        if lr_annealing:
            rate *= (1 + np.cos(np.pi * epoch / 5)) / 2
        loc -= (
            rate
            * (first / (1 - 0.9 ** (epoch + 1)))
            / ((second / (1 - 0.999 ** (epoch + 1))).sqrt() + 1e-8)
        )
        torch.testing.assert_close(actual, loc, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(flow.flow.loc, loc, rtol=0, atol=1e-12)


@pytest.mark.parametrize("patience", [None, 1, 3])
def test_patience_counts_ties(patience: int | None) -> None:
    flow = gaussian_flow()
    flow.fit(
        torch.ones(8, 2),
        n_epochs=8,
        lr=0,
        validation_fraction=0,
        patience=patience,
        rng=42,
    )
    expected = 8 if patience is None else 1 + patience
    assert len(flow.fit_history["training_loss"]) == expected
    assert len(flow.fit_history["validation_loss"]) == expected


@pytest.mark.parametrize("patience", [None, 1])
def test_patience_allows_continued_improvement(patience: int | None) -> None:
    flow = gaussian_flow()
    flow.fit(
        torch.full((8, 2), 10.0),
        n_epochs=8,
        lr=0.1,
        validation_fraction=0,
        patience=patience,
        rng=42,
    )
    losses = flow.fit_history["validation_loss"]
    assert len(losses) == 8
    assert all(a > b for a, b in zip(losses, losses[1:]))


@pytest.mark.parametrize("patience", [None, 3, 5])
def test_restores_best_validation_parameters(patience: int | None) -> None:
    flow = gaussian_flow()
    flow.fit(
        torch.full((8, 2), 0.25),
        n_epochs=20,
        lr=0.2,
        validation_fraction=0,
        patience=patience,
        rng=42,
    )
    losses = flow.fit_history["validation_loss"]
    locations = flow.flow.validation_locations
    best = int(np.argmin(losses))
    assert best < len(losses) - 1
    torch.testing.assert_close(flow.flow.loc, locations[best], rtol=0, atol=0)
    assert len(flow.fit_history["training_loss"]) == len(losses)
    if patience == 5:
        # Four unsuccessful epochs precede a new best at epoch six.
        # The counter must reset there, allowing five more epochs.
        assert best == 5
        assert len(losses) == 11
    if patience is not None:
        assert len(losses) == best + 1 + patience
    else:
        assert len(losses) == 20


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_progress_preserves_seeded_real_zuko_training(
    dtype: torch.dtype, capsys: pytest.CaptureFixture[str]
) -> None:
    particles = np.random.default_rng(21).normal(size=(32, 2))
    flows = []
    for show_progress in (False, True):
        flow = ZukoFlow(dims=2, hidden_features=[8, 8], dtype=dtype)
        state = torch.random.get_rng_state().clone()
        assert (
            flow.fit(
                particles,
                n_epochs=3,
                batch_size=8,
                clip_grad=0.5,
                lr_annealing=True,
                patience=2,
                show_progress=show_progress,
                rng=42,
            )
            is flow
        )
        assert torch.equal(torch.random.get_rng_state(), state)
        output = capsys.readouterr().err
        if show_progress:
            assert "Epochs" in output
            assert "train_loss=" in output and "val_loss=" in output
        else:
            assert output == ""
        assert all(
            p.dtype == dtype and p.device.type == "cpu"
            for p in flow.flow.parameters()
        )
        flows.append(flow)
    assert flows[0].fit_history == flows[1].fit_history
    for key, value in flows[0].flow.state_dict().items():
        torch.testing.assert_close(
            value, flows[1].flow.state_dict()[key], rtol=0, atol=0
        )


@pytest.mark.parametrize("adapter", [FlowTransform, FlowProposal])
def test_training_controls_through_sequince(adapter: Any) -> None:
    particles = np.random.default_rng(21).normal(size=(32, 2))
    options: dict[str, Any] = dict(
        n_epochs=3,
        batch_size=8,
        clip_grad=0.5,
        lr_annealing=True,
        patience=2,
        show_progress=False,
    )
    direct = ZukoFlow(dims=2, hidden_features=[8, 8], seed=22)
    direct.fit(particles, rng=42, **options)
    wrapped = adapter(
        ZukoFlow(dims=2, hidden_features=[8, 8], seed=22), fit_kwargs=options
    ).fit(particles, rng=42)
    assert wrapped.flow.fit_history == direct.fit_history
    for key, value in direct.flow.state_dict().items():
        torch.testing.assert_close(
            value, wrapped.flow.flow.state_dict()[key], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), -float("inf")]
)
def test_rejects_nonfinite_raw_data(bad_value: float) -> None:
    # The bounded logit clips infinities, so validation must precede it.
    from sequince.transforms import BoundedToUnboundedTransform

    flow = ZukoFlow(
        dims=3,
        hidden_features=[8, 8],
        particle_transform=BoundedToUnboundedTransform(
            {i: (0, 1) for i in range(3)}
        ),
    )
    particles = torch.full((8, 3), 0.5)
    particles[0, 0] = bad_value
    particles[-1, 2] = bad_value
    with pytest.raises(
        ValueError, match=r"Raw.*non-finite.*dimensions: \[0, 2\]"
    ):
        flow.fit(particles, n_epochs=1, rng=42)


class NonfiniteTransform(ParticleTransform):
    def forward(
        self, particles: torch.Tensor, context: Any = None, *, rng: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        values = particles.clone()
        values[0, 0] = float("nan")
        values[-1, 2] = float("inf")
        return values, torch.zeros(len(values)), None


def test_rejects_nonfinite_transformed_data() -> None:
    flow = ZukoFlow(dims=3, particle_transform=NonfiniteTransform())
    with pytest.raises(
        ValueError, match=r"Transformed.*non-finite.*dimensions: \[0, 2\]"
    ):
        flow.fit(torch.zeros(8, 3), n_epochs=1, rng=42)


@pytest.mark.parametrize(
    "samples", [[], [1, 2], np.zeros((0, 2)), np.zeros((8, 3))]
)
def test_rejects_invalid_training_shapes(samples: Any) -> None:
    with pytest.raises(ValueError, match="Raw training data.*shape"):
        gaussian_flow().fit(samples, n_epochs=1)


@pytest.mark.parametrize(
    "options",
    [
        {"clip_grad": 0},
        {"clip_grad": -1},
        {"clip_grad": float("nan")},
        {"clip_grad": float("inf")},
        {"patience": 0},
        {"patience": -1},
        {"patience": 1.5},
        {"patience": True},
    ],
)
def test_rejects_invalid_training_controls(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="clip_grad|patience"):
        gaussian_flow().fit(torch.zeros(8, 2), n_epochs=1, **options)
