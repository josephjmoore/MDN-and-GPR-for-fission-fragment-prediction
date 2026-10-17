import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def print_yield_curves(npz_path: str | Path, source_file: str) -> None:
    """Print all energy-dependent yield curves for one original ENDF file."""
    with np.load(npz_path, allow_pickle=False) as data:
        matches = np.flatnonzero(data["source_file"] == source_file)
        if matches.size == 0:
            available = ", ".join(sorted(set(data["source_file"].tolist())))
            raise ValueError(
                f"No curves found for {source_file!r}. Available files: {available}"
            )

        for index in matches:
            start, end = data["offsets"][index:index + 2]
            curve = np.column_stack((
                data["fragment_A"][start:end],
                data["yields"][start:end],
                data["yield_uncertainties"][start:end],
            ))
            print(
                f"\n{data['source_file'][index]} | "
                f"curve_id={data['curve_id'][index]} | "
                f"file_id={data['file_id'][index]}"
            )
            print(
                f"Target: Z={data['target_Z'][index]}, "
                f"A={data['target_A'][index]} | "
                f"incident energy={data['incident_energy_eV'][index]:g} eV "
                f"({data['incident_energy_MeV'][index]:g} MeV) | "
                f"excitation energy={data['excitation_energy_MeV'][index]:.6g} MeV"
            )
            print("Columns: fragment_A, yield, yield_uncertainty")
            print(curve)


def load_training_data(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return condition features, shared mass bins, and normalized yield curves.
    Each yield curve is normalized to sum to 1.0, and the total yield is 2.0.
    For each curve, the features are: target_Z, target_A, log10(incident_energy_MeV).
    Offsets mark the start and end of each curve in the connected fragment_A, yields, and yield_uncertainties arrays.

    """
    with np.load(npz_path, allow_pickle=False) as data:
        curve_count = len(data["target_Z"])
        offsets = data["offsets"]
        masses = np.unique(data["fragment_A"]).astype(np.float32)
        mass_to_index = {int(mass): index for index, mass in enumerate(masses)}
        features = np.empty((curve_count, 3), dtype=np.float32)
        targets = np.zeros((curve_count, len(masses)), dtype=np.float32)

        for index in range(curve_count):
            start, end = offsets[index:index + 2]
            energy = float(data["incident_energy_MeV"][index])
            if energy <= 0:
                raise ValueError(
                    f"Incident energy must be positive for curve {index}; "
                    "log-energy features cannot represent zero."
                )
            features[index] = (
                data["target_Z"][index],
                data["target_A"][index],
                np.log10(energy),
            )
            curve_masses = data["fragment_A"][start:end].astype(int)
            curve_yields = data["yields"][start:end]
            if np.any(curve_yields < 0) or curve_yields.sum() <= 0:
                raise ValueError(f"Curve {index} has invalid yields.")
            for mass, value in zip(curve_masses, curve_yields):
                targets[index, mass_to_index[mass]] = value
            targets[index] /= targets[index].sum()

    return features, masses, targets


class FissionYieldMDN(nn.Module):
    """
    Conditional 5-Gaussian-mixture model for fragment mass given target and energy.
    Each mixture component has a weight, mean, and standard deviation.
    The mean is constrained to the training mass range.
    """

    def __init__(
        self,
        mass_min: float,
        mass_max: float,
        input_size: int = 3,
        components: int = 5,
        hidden_size: int = 64,
    ):
        super().__init__()
        if mass_max <= mass_min:
            raise ValueError("mass_max must be greater than mass_min.")
        self.components = components
        self.mass_min = mass_min
        self.mass_max = mass_max
        # feedforward network: input (3 inputs)-> hidden (64 units)-> hidden (64 units)-> output (weights, means, stddevs) for each of the 5 gaussian curves
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, components * 3),
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.network(inputs).reshape(-1, self.components, 3)
        weights = torch.softmax(raw[:, :, 0], dim=1)
        means = self.mass_min + (
            self.mass_max - self.mass_min
        ) * torch.sigmoid(raw[:, :, 1])
        sigma_min = 0.5
        sigma_max = 30.0
        standard_deviations = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw[:, :, 2])
        return weights, means, standard_deviations


def mdn_loss(
    model: FissionYieldMDN,
    inputs: torch.Tensor,
    masses: torch.Tensor,
    target_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Yield-weighted negative log likelihood of each observed mass bin."""
    weights, means, standard_deviations = model(inputs)
    difference = masses[None, :, None] - means[:, None, :]
    log_normal = (
        -0.5 * (difference / standard_deviations[:, None, :]) ** 2
        - torch.log(standard_deviations[:, None, :])
        - 0.5 * np.log(2.0 * np.pi)
    )
    log_mixture = torch.logsumexp(
        torch.log(weights[:, None, :].clamp_min(1e-12)) + log_normal,
        dim=2,
    )
    return -(target_probabilities * log_mixture).sum(dim=1).mean()


def train_mdn(
    npz_path: str | Path,
    *,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[FissionYieldMDN, dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """
    Train on whole curves and retain the best validation-loss model.
    Also returns the validation curve indices so accuracy can be measured on unseen curves.
    """
    features, mass_values, targets = load_training_data(npz_path)
    if len(features) < 3:
        raise ValueError("At least three yield curves are required for training.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    validation_count = max(1, int(round(len(indices) * validation_fraction)))
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]

    feature_mean = features[training_indices].mean(axis=0)
    feature_scale = features[training_indices].std(axis=0)
    feature_scale[feature_scale == 0] = 1.0 # avoids division by 0
    normalized_features = (features - feature_mean) / feature_scale

    # Numpy -> Pytorch Tensor
    x = torch.tensor(normalized_features, dtype=torch.float32)
    mass_tensor = torch.tensor(mass_values, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)

    torch.manual_seed(seed)
    model = FissionYieldMDN(
        mass_min=float(mass_values.min()),
        mass_max=float(mass_values.max()),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    patience = 60

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = mdn_loss(
            model,
            x[training_indices],
            mass_tensor,
            target_tensor[training_indices],
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = mdn_loss(
                model,
                x[validation_indices],
                mass_tensor,
                target_tensor[validation_indices],
            ).item()
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    normalization = {
        "feature_mean": torch.tensor(feature_mean, dtype=torch.float32),
        "feature_scale": torch.tensor(feature_scale, dtype=torch.float32),
    }
    return model, normalization, mass_values, validation_indices


def predict_yield_curve(
    model: FissionYieldMDN,
    normalization: dict[str, torch.Tensor],
    mass_values: np.ndarray,
    target_Z: int,
    target_A: int,
    incident_energy_MeV: float,
) -> np.ndarray:
    """Predict yields on the training mass grid; predicted total yield is 2."""
    if incident_energy_MeV <= 0:
        raise ValueError("Incident neutron energy must be greater than zero.")
    raw_features = torch.tensor(
        [[target_Z, target_A, np.log10(incident_energy_MeV)]],
        dtype=torch.float32,
    )
    inputs = (
        raw_features - normalization["feature_mean"]
    ) / normalization["feature_scale"]
    mass_tensor = torch.tensor(mass_values, dtype=torch.float32)
    with torch.no_grad():
        weights, means, standard_deviations = model(inputs)
        difference = mass_tensor[:, None] - means[0][None, :]
        log_normal = (
            -0.5 * (difference / standard_deviations[0][None, :]) ** 2
            - torch.log(standard_deviations[0][None, :])
            - 0.5 * np.log(2.0 * np.pi)
        )
        log_density = torch.logsumexp(
            torch.log(weights[0].clamp_min(1e-12))[None, :] + log_normal,
            dim=1,
        )
        probabilities = torch.softmax(log_density, dim=0)
    return (2.0 * probabilities.numpy()).astype(np.float64)


def evaluate_predictions(
    model: FissionYieldMDN,
    normalization: dict[str, torch.Tensor],
    mass_values: np.ndarray,
    npz_path: str | Path,
    curve_indices: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """
    Compare predicted and evaluated yield curves (both scaled to total yield 2).

    "chi2_per_curve" holds chi^2/N for each curve in curve_indices (NaN where the
    curve has no non-zero uncertainties). The other metrics are averaged over the curves:
    - RMSE (yield %): root-mean-square difference over mass bins that have data.
    - Reduced chi-squared: chi^2/N using the ENDF yield uncertainties. A value
      near 1 means the prediction agrees with the data within its quoted
      uncertainties; much larger than 1 means the model misses the data.
      Bins with zero uncertainty are excluded.
    - KL divergence D(data || prediction): how much information is lost when
      the predicted distribution is used in place of the evaluated one (0 = identical).

    Sources:
    - Reduced chi-squared: P. R. Bevington and D. K. Robinson, "Data Reduction and
      Error Analysis for the Physical Sciences", 3rd ed., McGraw-Hill (2003), Ch. 11.
    - KL divergence: S. Kullback and R. A. Leibler, "On Information and Sufficiency",
      Ann. Math. Statist. 22(1), 79-86 (1951).
    - MDNs applied to fission mass yields and compared against evaluated data:
      A. E. Lovell, A. T. Mohan, T. M. Sprouse, M. R. Mumpower, "Quantifying
      uncertainties on fission fragment mass yields with mixture density networks",
      J. Phys. G: Nucl. Part. Phys. 47, 114001 (2020).
    """
    mass_to_index = {int(mass): index for index, mass in enumerate(mass_values)}
    rmse_values, chi2_values, kl_values = [], [], []

    with np.load(npz_path, allow_pickle=False) as data:
        for index in curve_indices:
            start, end = data["offsets"][index:index + 2]
            curve_masses = data["fragment_A"][start:end].astype(int)
            curve_yields = data["yields"][start:end].astype(np.float64)
            curve_errors = data["yield_uncertainties"][start:end].astype(np.float64)

            # Rescale data and its uncertainties to the same total yield of 2 as the prediction.
            scale = 2.0 / curve_yields.sum()
            observed = curve_yields * scale
            errors = curve_errors * scale

            prediction = predict_yield_curve(
                model,
                normalization,
                mass_values,
                int(data["target_Z"][index]),
                int(data["target_A"][index]),
                float(data["incident_energy_MeV"][index]),
            )
            predicted = prediction[[mass_to_index[mass] for mass in curve_masses]]

            residuals = predicted - observed
            rmse_values.append(100.0 * np.sqrt(np.mean(residuals ** 2)))

            has_error = errors > 0
            chi2_values.append(
                np.mean((residuals[has_error] / errors[has_error]) ** 2)
                if np.any(has_error)
                else np.nan
            )

            # KL divergence on the full mass grid, using probabilities (sum to 1).
            p = np.zeros(len(mass_values))
            p[[mass_to_index[mass] for mass in curve_masses]] = observed / 2.0
            q = prediction / 2.0
            nonzero = p > 0
            kl_values.append(np.sum(p[nonzero] * np.log(p[nonzero] / q[nonzero])))

    chi2_per_curve = np.array(chi2_values, dtype=np.float64)
    has_chi2 = ~np.isnan(chi2_per_curve)
    return {
        "rmse_percent": float(np.mean(rmse_values)),
        "reduced_chi2": (
            float(np.mean(chi2_per_curve[has_chi2])) if np.any(has_chi2) else float("nan")
        ),
        "chi2_per_curve": chi2_per_curve,
        "kl_divergence": float(np.mean(kl_values)),
    }


def find_curve_index(
    npz_path: str | Path,
    target_Z: int,
    target_A: int,
    incident_energy_MeV: float,
) -> int | None:
    """Return the index of the first evaluated curve matching the inputs, or None."""
    with np.load(npz_path, allow_pickle=False) as data:
        matches = np.flatnonzero(
            (data["target_Z"] == target_Z)
            & (data["target_A"] == target_A)
            & np.isclose(data["incident_energy_MeV"], incident_energy_MeV)
        )
    return int(matches[0]) if matches.size else None


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    npz_path = base_dir / "yield_curves.npz"
    model, normalization, mass_values, validation_indices = train_mdn(npz_path)

    metrics = evaluate_predictions(
        model, normalization, mass_values, npz_path, validation_indices
    )
    print(
        f"Validation ({len(validation_indices)} curves): "
        f"RMSE = {metrics['rmse_percent']:.4f} yield-%, "
        f"reduced chi^2 = {metrics['reduced_chi2']:.3f}, "
        f"KL divergence = {metrics['kl_divergence']:.4f}"
    )
    target_Z, target_A, incident_energy_MeV = 92, 235, 14.0
    prediction = predict_yield_curve(
        model,
        normalization,
        mass_values,
        target_Z=target_Z,
        target_A=target_A,
        incident_energy_MeV=incident_energy_MeV,
    )
    print(f"Trained MDN. Predicted total yield: {prediction.sum():.6f}")

    prediction_label = "MDN prediction"
    curve_index = find_curve_index(npz_path, target_Z, target_A, incident_energy_MeV)
    if curve_index is None:
        print("No evaluated curve matches these inputs; chi^2/N not computed.")
    else:
        curve_chi2 = evaluate_predictions(
            model, normalization, mass_values, npz_path, np.array([curve_index])
        )["chi2_per_curve"][0]
        split = "validation" if curve_index in validation_indices else "training"
        print(f"chi^2/N for curve {curve_index} ({split} set) = {curve_chi2:.3f}")
        prediction_label += f" ($\\chi^2/N$ = {curve_chi2:.3f}, {split} curve)"

        with np.load(npz_path, allow_pickle=False) as data:
            start, end = data["offsets"][curve_index:curve_index + 2]
            curve_yields = data["yields"][start:end]
            scale = 100.0 / curve_yields.sum()
            plt.errorbar(
                data["fragment_A"][start:end],
                curve_yields * scale,
                yerr=data["yield_uncertainties"][start:end] * scale,
                fmt=".",
                color="black",
                label="ENDF evaluation",
            )

    plt.plot(mass_values, 100.0 * prediction / prediction.sum(), label=prediction_label)
    plt.xlabel("Fragment mass number")
    plt.ylabel("Yield (%)")
    plt.title(
        f"Z={target_Z}, A={target_A} mass-yield distribution at {incident_energy_MeV} MeV"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
