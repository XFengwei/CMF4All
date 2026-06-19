"""Core mass function construction and power-law fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .survey import Survey, clean_masses


@dataclass(frozen=True)
class PowerLawFit:
    """Single-slope power-law fit for a high-mass tail.

    ``gamma`` follows the ``dN/dlogM ∝ M^(-gamma)`` convention, so
    ``gamma = alpha - 1`` when ``dN/dM ∝ M^(-alpha)``.
    """

    alpha: float
    alpha_err: float
    gamma: float
    gamma_err: float
    mmin: float
    n_fit: int
    ks_distance: float
    method: str = "mle"
    alpha_samples: np.ndarray | None = field(default=None, repr=False)
    gamma_samples: np.ndarray | None = field(default=None, repr=False)
    alpha_ci: tuple[float, float] | None = None
    gamma_ci: tuple[float, float] | None = None


def masses_from(data: Survey | Iterable[float]) -> np.ndarray:
    """Return cleaned masses from either a Survey or an array-like object."""

    if isinstance(data, Survey):
        return data.masses()
    return clean_masses(data)


def compile_survey_masses(
    surveys: Iterable[Survey],
    *,
    mass_min: float | None = None,
    completeness: str | None = None,
) -> np.ndarray:
    """Return one pooled mass array built from multiple surveys.

    Parameters
    ----------
    surveys
        Iterable of :class:`Survey` objects to combine.
    mass_min
        Shared lower-mass cut applied to every survey after any completeness
        filtering.
    completeness
        Optional completeness handling before pooling:

        - ``None`` or ``"none"``: no survey-specific completeness cut
        - ``"per-survey"``: each survey is truncated at its own
          ``completeness_msun``
        - ``"shared_max"``: all surveys are truncated at the highest
          completeness limit among them
        - ``"fit"``: each survey is truncated at ``fit_mass_min_msun`` and
          falls back to ``completeness_msun`` if needed
    """

    surveys_list = _as_survey_list(surveys)
    completeness_mode = "none" if completeness is None else str(completeness).casefold()
    if completeness_mode not in {"none", "per-survey", "shared_max", "fit"}:
        raise ValueError(
            "completeness must be None, 'none', 'per-survey', 'shared_max', or 'fit'."
        )

    shared_threshold = None
    if completeness_mode == "shared_max":
        completeness_values = [survey.completeness_msun for survey in surveys_list]
        if any(value is None for value in completeness_values):
            missing = [survey.key for survey in surveys_list if survey.completeness_msun is None]
            raise ValueError(
                "shared_max completeness requires completeness_msun for every survey; "
                f"missing for {missing}."
            )
        shared_threshold = float(max(completeness_values))

    compiled: list[np.ndarray] = []
    for survey in surveys_list:
        values = survey.masses()
        threshold = _compiled_mass_threshold(
            survey,
            completeness_mode=completeness_mode,
            shared_threshold=shared_threshold,
        )
        if threshold is not None:
            values = values[values >= threshold]
        if mass_min is not None:
            values = values[values >= float(mass_min)]
        if values.size:
            compiled.append(values)

    if not compiled:
        raise ValueError("No masses remain after compiling the selected surveys.")
    return np.concatenate(compiled)


def compiled_cmf(
    surveys: Iterable[Survey],
    *,
    kind: str = "differential",
    mass_min: float | None = None,
    completeness: str | None = None,
    bins_log10: np.ndarray | None = None,
    dlogm: float = 0.2,
    normalize: str | bool | None = None,
    tail_mass: float | None = None,
) -> pd.DataFrame:
    """Build a CMF from the pooled masses of multiple surveys."""

    pooled_masses = compile_survey_masses(
        surveys,
        mass_min=mass_min,
        completeness=completeness,
    )
    kind_key = kind.casefold()
    if kind_key == "differential":
        return differential_cmf(
            pooled_masses,
            bins_log10=bins_log10,
            dlogm=dlogm,
            normalize=normalize if isinstance(normalize, str) or normalize is None else None,
            tail_mass=tail_mass,
        )
    if kind_key in {"ccdf", "complementary", "cumulative"}:
        ccdf_normalize = True if normalize is None else normalize
        return complementary_cmf(
            pooled_masses,
            normalize=ccdf_normalize,
            tail_mass=tail_mass,
        )
    raise ValueError("kind must be 'differential' or 'ccdf'.")


def make_log_bins(
    masses: Survey | Iterable[float],
    dlogm: float = 0.2,
    logm_min: float | None = None,
    logm_max: float | None = None,
) -> np.ndarray:
    """Create log10 mass-bin edges."""

    values = masses_from(masses)
    if values.size == 0:
        raise ValueError("No positive finite masses were provided.")
    if logm_min is None:
        logm_min = np.floor(np.log10(values.min()) / dlogm) * dlogm
    if logm_max is None:
        logm_max = np.ceil(np.log10(values.max()) / dlogm) * dlogm
    return np.arange(logm_min, logm_max + dlogm, dlogm)


def differential_cmf(
    masses: Survey | Iterable[float],
    bins_log10: np.ndarray | None = None,
    dlogm: float = 0.2,
    normalize: str | None = None,
    tail_mass: float | None = None,
) -> pd.DataFrame:
    """Compute binned differential CMF as dN/dlogM."""

    values = masses_from(masses)
    if bins_log10 is None:
        bins_log10 = make_log_bins(values, dlogm=dlogm)
    logm = np.log10(values)
    counts, edges = np.histogram(logm, bins=bins_log10)
    widths = np.diff(edges)
    centers_log = 0.5 * (edges[:-1] + edges[1:])
    centers_mass = 10.0**centers_log
    dn_dlogm = counts / widths
    err = np.sqrt(counts) / widths
    threshold = _resolve_tail_normalization_threshold(masses, normalize, tail_mass)
    tail_mask = np.ones_like(centers_mass, dtype=bool) if threshold is None else centers_mass >= threshold
    dn_dlogm, err = _normalize_binned(dn_dlogm, err, widths, normalize, tail_mask=tail_mask)
    result = pd.DataFrame(
        {
            "logm_center": centers_log,
            "mass_msun": centers_mass,
            "bin_low_msun": 10.0 ** edges[:-1],
            "bin_high_msun": 10.0 ** edges[1:],
            "count": counts,
            "dn_dlogm": dn_dlogm,
            "dn_dlogm_err": err,
        }
    )
    if threshold is not None:
        result = result.loc[tail_mask].reset_index(drop=True)
    return result


def complementary_cmf(
    masses: Survey | Iterable[float],
    normalize: bool | str = True,
    tail_mass: float | None = None,
) -> pd.DataFrame:
    """Compute the complementary cumulative CMF, N(>=M) or P(>=M)."""

    values = masses_from(masses)
    if values.size == 0:
        raise ValueError("No positive finite masses were provided.")
    unique = np.unique(values)
    counts = np.array([np.count_nonzero(values >= mass) for mass in unique], dtype=float)
    threshold = _resolve_tail_normalization_threshold(masses, normalize, tail_mass)
    if threshold is not None:
        mask = unique >= threshold
        unique = unique[mask]
        counts = counts[mask]
        if counts.size == 0:
            raise ValueError(f"No masses are available above tail_mass={threshold:g}.")
        denominator = counts[0]
        probability = counts / denominator
    else:
        probability = counts / values.size if normalize else counts.copy()
    return pd.DataFrame(
        {
            "mass_msun": unique,
            "n_greater_equal": counts.astype(int),
            "p_greater_equal": probability,
            "percent_greater_equal": 100.0 * probability,
        }
    )


def fit_power_law_mle(
    masses: Survey | Iterable[float],
    mmin: float,
) -> PowerLawFit:
    """Fit p(M) proportional to M^-alpha above ``mmin`` using unbinned MLE."""

    values = masses_from(masses)
    tail = values[values >= mmin]
    if tail.size < 2:
        raise ValueError(f"Only {tail.size} cores above mmin={mmin:g}; need at least 2.")
    denominator = np.sum(np.log(tail / mmin))
    if denominator <= 0:
        raise ValueError("Cannot fit a power law when all tail masses equal mmin.")
    alpha = 1.0 + tail.size / denominator
    alpha_err = (alpha - 1.0) / np.sqrt(tail.size)
    gamma = alpha - 1.0
    return PowerLawFit(
        alpha=float(alpha),
        alpha_err=float(alpha_err),
        gamma=float(gamma),
        gamma_err=float(alpha_err),
        mmin=float(mmin),
        n_fit=int(tail.size),
        ks_distance=ks_distance(tail, alpha=float(alpha), mmin=float(mmin)),
    )


def select_mmin_ks(
    masses: Survey | Iterable[float],
    min_tail: int = 20,
    candidates: Iterable[float] | None = None,
) -> PowerLawFit:
    """Select ``mmin`` by minimizing the KS distance of the fitted tail."""

    values = masses_from(masses)
    source = values if candidates is None else np.asarray(list(candidates), dtype=float)
    candidate_values = np.unique(source)
    best: PowerLawFit | None = None
    for candidate in candidate_values:
        if not np.isfinite(candidate) or candidate <= 0:
            continue
        if np.count_nonzero(values >= candidate) < min_tail:
            continue
        fit = fit_power_law_mle(values, mmin=float(candidate))
        if best is None or fit.ks_distance < best.ks_distance:
            best = fit
    if best is None:
        raise ValueError(f"No mmin candidate has at least {min_tail} tail masses.")
    return best


def fit_power_law(
    masses: Survey | Iterable[float],
    mmin: float | str | None = None,
    min_tail: int = 20,
    uncertainty: str | None = None,
    n_steps: int = 6000,
    burn_in: int = 1000,
    proposal_width: float = 0.04,
    random_seed: int | None = 12345,
) -> PowerLawFit:
    """Fit a power-law tail, optionally selecting ``mmin`` and MCMC errors.

    Parameters
    ----------
    uncertainty
        If ``None`` or ``"mle"``, use the analytic MLE uncertainty. If
        ``"mcmc"``, run a one-parameter Metropolis sampler for ``alpha``
        after choosing the fitting lower limit.
    """

    if isinstance(mmin, str):
        if mmin.casefold() != "ks":
            raise ValueError("String mmin must be 'ks'.")
        base_fit = select_mmin_ks(masses, min_tail=min_tail)
    else:
        if mmin is None:
            if isinstance(masses, Survey):
                mmin = masses.fit_mass_min_msun or masses.completeness_msun
            if mmin is None:
                base_fit = select_mmin_ks(masses, min_tail=min_tail)
            else:
                base_fit = fit_power_law_mle(masses, mmin=float(mmin))
        else:
            base_fit = fit_power_law_mle(masses, mmin=float(mmin))

    if uncertainty is None or uncertainty == "mle":
        return base_fit
    if uncertainty != "mcmc":
        raise ValueError("uncertainty must be None, 'mle', or 'mcmc'.")
    return fit_power_law_mcmc(
        masses,
        mmin=base_fit.mmin,
        n_steps=n_steps,
        burn_in=burn_in,
        proposal_width=proposal_width,
        random_seed=random_seed,
    )


def fit_power_law_mcmc(
    masses: Survey | Iterable[float],
    mmin: float,
    n_steps: int = 6000,
    burn_in: int = 1000,
    proposal_width: float = 0.04,
    random_seed: int | None = 12345,
) -> PowerLawFit:
    """Estimate power-law slope uncertainty with Metropolis MCMC.

    The sampler uses the exact continuous power-law likelihood for the tail
    ``M >= mmin`` and a flat prior for ``alpha > 1``. The returned ``alpha``
    and ``gamma = alpha - 1`` are posterior medians, while ``*_err`` is half
    the 16--84 percentile width.
    """

    values = masses_from(masses)
    tail = values[values >= mmin]
    if tail.size < 2:
        raise ValueError(f"Only {tail.size} cores above mmin={mmin:g}; need at least 2.")
    if burn_in >= n_steps:
        raise ValueError("burn_in must be smaller than n_steps.")

    mle_fit = fit_power_law_mle(tail, mmin=mmin)
    alpha_samples = sample_power_law_alpha_mcmc(
        tail,
        mmin=mmin,
        n_steps=n_steps,
        burn_in=burn_in,
        proposal_width=proposal_width,
        random_seed=random_seed,
        initial_alpha=mle_fit.alpha,
    )
    alpha16, alpha50, alpha84 = np.percentile(alpha_samples, [16, 50, 84])
    gamma_samples = alpha_samples - 1.0
    gamma16, gamma50, gamma84 = np.percentile(gamma_samples, [16, 50, 84])
    alpha_err = 0.5 * (alpha84 - alpha16)
    gamma_err = 0.5 * (gamma84 - gamma16)
    return PowerLawFit(
        alpha=float(alpha50),
        alpha_err=float(alpha_err),
        gamma=float(gamma50),
        gamma_err=float(gamma_err),
        mmin=float(mmin),
        n_fit=int(tail.size),
        ks_distance=ks_distance(tail, alpha=float(alpha50), mmin=float(mmin)),
        method="mcmc",
        alpha_samples=alpha_samples,
        gamma_samples=gamma_samples,
        alpha_ci=(float(alpha16), float(alpha84)),
        gamma_ci=(float(gamma16), float(gamma84)),
    )


def sample_power_law_alpha_mcmc(
    masses: Survey | Iterable[float],
    mmin: float,
    n_steps: int = 6000,
    burn_in: int = 1000,
    proposal_width: float = 0.04,
    random_seed: int | None = 12345,
    initial_alpha: float | None = None,
) -> np.ndarray:
    """Return posterior samples for the power-law ``alpha`` slope."""

    values = masses_from(masses)
    tail = values[values >= mmin]
    if tail.size < 2:
        raise ValueError(f"Only {tail.size} cores above mmin={mmin:g}; need at least 2.")
    if burn_in >= n_steps:
        raise ValueError("burn_in must be smaller than n_steps.")
    if proposal_width <= 0:
        raise ValueError("proposal_width must be positive.")

    rng = np.random.default_rng(random_seed)
    alpha = float(initial_alpha) if initial_alpha is not None else fit_power_law_mle(tail, mmin).alpha
    current_logp = _log_likelihood_alpha(alpha, tail, mmin)
    samples = np.empty(n_steps, dtype=float)
    for index in range(n_steps):
        proposal = alpha + rng.normal(0.0, proposal_width)
        proposal_logp = _log_likelihood_alpha(proposal, tail, mmin)
        if np.log(rng.uniform()) < proposal_logp - current_logp:
            alpha = proposal
            current_logp = proposal_logp
        samples[index] = alpha
    return samples[burn_in:]


def _log_likelihood_alpha(alpha: float, tail: np.ndarray, mmin: float) -> float:
    if alpha <= 1.0:
        return -np.inf
    n = tail.size
    return float(
        n * np.log(alpha - 1.0)
        + n * (alpha - 1.0) * np.log(mmin)
        - alpha * np.sum(np.log(tail))
    )


def ks_distance(tail_masses: Iterable[float], alpha: float, mmin: float) -> float:
    """One-sample KS distance between empirical tail CDF and power-law CDF."""

    tail = clean_masses(tail_masses)
    tail = tail[tail >= mmin]
    if tail.size == 0:
        raise ValueError("No tail masses were provided.")
    empirical = np.arange(1, tail.size + 1, dtype=float) / tail.size
    model = 1.0 - (tail / mmin) ** (1.0 - alpha)
    return float(np.max(np.abs(empirical - model)))


def powerlaw_differential_shape(
    mass_grid: Iterable[float],
    gamma: float,
    norm_mass: float,
    norm_value: float,
) -> np.ndarray:
    """Evaluate dN/dlogM proportional to M^(-gamma) at ``mass_grid``."""

    mass_grid = np.asarray(mass_grid, dtype=float)
    return norm_value * (mass_grid / norm_mass) ** (-gamma)


def powerlaw_ccdf_shape(
    mass_grid: Iterable[float],
    gamma: float,
    mmin: float,
    p_at_mmin: float,
) -> np.ndarray:
    """Evaluate N(>M) or P(>M) proportional to M^(-gamma)."""

    mass_grid = np.asarray(mass_grid, dtype=float)
    return p_at_mmin * (mass_grid / mmin) ** (-gamma)


def _normalize_binned(
    y: np.ndarray,
    err: np.ndarray,
    widths: np.ndarray,
    normalize: str | None,
    tail_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if normalize is None or normalize == "none":
        return y, err
    if normalize == "area":
        scale = np.sum(y * widths)
    elif normalize == "peak":
        scale = np.nanmax(y)
    elif _is_tail_normalization(normalize):
        mask = np.ones_like(y, dtype=bool) if tail_mask is None else tail_mask
        if not np.any(mask):
            raise ValueError("No bins are available in the selected tail-normalized regime.")
        scale = np.sum(y[mask] * widths[mask])
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")
    if np.isfinite(scale) and scale > 0:
        return y / scale, err / scale
    return y, err


def _resolve_tail_normalization_threshold(
    data: Survey | Iterable[float],
    normalize: bool | str | None,
    tail_mass: float | None,
) -> float | None:
    if not _is_tail_normalization(normalize):
        return None
    mode = str(normalize).casefold()
    if tail_mass is not None:
        return float(tail_mass)
    if isinstance(data, Survey):
        if mode == "above_completeness":
            if data.completeness_msun is None:
                raise ValueError(f"{data.key} does not define completeness_msun.")
            return data.completeness_msun
        if mode == "above_mmin":
            if data.fit_mass_min_msun is not None:
                return data.fit_mass_min_msun
            if data.completeness_msun is not None:
                return data.completeness_msun
            raise ValueError(f"{data.key} does not define fit_mass_min_msun or completeness_msun.")
    raise ValueError(
        "Tail normalization requires tail_mass, or a Survey with completeness/fit metadata."
    )


def _is_tail_normalization(normalize: bool | str | None) -> bool:
    if not isinstance(normalize, str):
        return False
    return normalize.casefold() in {"tail", "above_completeness", "above_mmin"}


def _as_survey_list(surveys: Iterable[Survey]) -> list[Survey]:
    result = list(surveys)
    if not result:
        raise ValueError("At least one survey is required.")
    return result


def _compiled_mass_threshold(
    survey: Survey,
    *,
    completeness_mode: str,
    shared_threshold: float | None,
) -> float | None:
    if completeness_mode == "none":
        return None
    if completeness_mode == "per-survey":
        if survey.completeness_msun is None:
            raise ValueError(f"{survey.key} does not define completeness_msun.")
        return survey.completeness_msun
    if completeness_mode == "shared_max":
        return shared_threshold
    if completeness_mode == "fit":
        if survey.fit_mass_min_msun is not None:
            return survey.fit_mass_min_msun
        if survey.completeness_msun is not None:
            return survey.completeness_msun
        raise ValueError(f"{survey.key} does not define fit_mass_min_msun or completeness_msun.")
    raise ValueError(f"Unknown completeness mode: {completeness_mode}")
