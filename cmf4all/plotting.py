"""Plotting helpers for surveys and core mass functions."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .mass_function import (
    PowerLawFit,
    compile_survey_masses,
    complementary_cmf,
    differential_cmf,
    fit_power_law,
    masses_from,
    powerlaw_ccdf_shape,
    powerlaw_differential_shape,
)
from .survey import Survey


DEFAULT_DATA_COLOR = "#222222"
DEFAULT_MARKER_FACECOLOR = "white"
DEFAULT_FIT_COLOR = "#0072B2"
DEFAULT_MMIN_COLOR = "0.45"
DEFAULT_SHADE68_ALPHA = 0.28
DEFAULT_SHADE95_ALPHA = 0.14
DEFAULT_MULTI_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)


def plot_mass_radius(
    survey: Survey,
    ax=None,
    data_color: str = DEFAULT_DATA_COLOR,
    marker_facecolor: str = DEFAULT_MARKER_FACECOLOR,
    **kwargs,
):
    """Plot the core mass-radius relation for one survey."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    table = survey.mass_radius_table()
    defaults = {
        "s": 22,
        "alpha": 0.85,
        "label": survey.short_name,
        "edgecolors": data_color,
        "facecolors": marker_facecolor,
        "linewidths": 1.0,
    }
    defaults.update(kwargs)
    ax.scatter(table["radius_au"], table["mass_msun"], **defaults)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Core radius / size [au]")
    ax.set_ylabel(r"Core mass [$M_\odot$]")
    return ax


def plot_differential_cmf(
    survey_or_masses: Survey | Iterable[float],
    ax=None,
    dlogm: float = 0.2,
    normalize: str | None = None,
    fit: bool = True,
    mmin: float | str | None = None,
    min_tail: int = 20,
    uncertainty: str | None = None,
    n_steps: int = 6000,
    burn_in: int = 1000,
    proposal_width: float = 0.04,
    random_seed: int | None = 12345,
    tail_mass: float | None = None,
    label: str | None = None,
    data_color: str = DEFAULT_DATA_COLOR,
    marker_facecolor: str = DEFAULT_MARKER_FACECOLOR,
    fit_color: str = DEFAULT_FIT_COLOR,
    mmin_color: str = DEFAULT_MMIN_COLOR,
    shade_posterior: bool = True,
    shade68_alpha: float = DEFAULT_SHADE68_ALPHA,
    shade95_alpha: float = DEFAULT_SHADE95_ALPHA,
    **kwargs,
):
    """Plot a binned differential CMF and optionally overlay a power-law fit."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    fit_result = None
    if fit:
        fit_result = fit_power_law(
            survey_or_masses,
            mmin=mmin,
            min_tail=min_tail,
            uncertainty=uncertainty,
            n_steps=n_steps,
            burn_in=burn_in,
            proposal_width=proposal_width,
            random_seed=random_seed,
        )
    resolved_tail_mass = _resolve_plot_tail_mass(
        survey_or_masses,
        normalize=normalize,
        tail_mass=tail_mass,
        mmin=mmin,
        fit_result=fit_result,
    )
    cmf = differential_cmf(
        survey_or_masses,
        dlogm=dlogm,
        normalize=normalize,
        tail_mass=resolved_tail_mass,
    )
    plot_label = label or _label_for(survey_or_masses)
    ok = cmf["count"].to_numpy() > 0
    defaults = {
        "fmt": "o",
        "ms": 4.5,
        "lw": 1.1,
        "capsize": 2,
        "label": plot_label,
        "mec": data_color,
        "ecolor": data_color,
        "mfc": marker_facecolor,
        "color": data_color,
    }
    defaults.update(kwargs)
    ax.errorbar(
        cmf.loc[ok, "mass_msun"],
        cmf.loc[ok, "dn_dlogm"],
        yerr=cmf.loc[ok, "dn_dlogm_err"],
        **defaults,
    )
    if fit and fit_result is not None:
        fit_bins = ok & (cmf["mass_msun"].to_numpy() >= fit_result.mmin)
        if np.any(fit_bins):
            x0 = cmf.loc[fit_bins, "mass_msun"].iloc[0]
            y0 = cmf.loc[fit_bins, "dn_dlogm"].iloc[0]
            grid = np.logspace(np.log10(fit_result.mmin), np.log10(cmf["mass_msun"].max()), 200)
            if shade_posterior and fit_result.gamma_samples is not None:
                _shade_differential_posterior(
                    ax,
                    grid,
                    fit_result.gamma_samples,
                    norm_mass=x0,
                    norm_value=y0,
                    color=fit_color,
                    shade68_alpha=shade68_alpha,
                    shade95_alpha=shade95_alpha,
                )
            ax.plot(
                grid,
                powerlaw_differential_shape(grid, fit_result.gamma, x0, y0),
                lw=1.5,
                color=fit_color,
                label=rf"{plot_label} fit: $\Gamma={fit_result.gamma:.2f}$",
            )
            ax.axvline(fit_result.mmin, ls="--", lw=1.0, color=mmin_color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Core mass [$M_\odot$]")
    ax.set_ylabel(_differential_ylabel(normalize))
    return ax, fit_result


def plot_ccdf(
    survey_or_masses: Survey | Iterable[float],
    ax=None,
    fit: bool = True,
    normalize: bool | str = True,
    mmin: float | str | None = None,
    min_tail: int = 20,
    uncertainty: str | None = None,
    n_steps: int = 6000,
    burn_in: int = 1000,
    proposal_width: float = 0.04,
    random_seed: int | None = 12345,
    tail_mass: float | None = None,
    label: str | None = None,
    data_color: str = DEFAULT_DATA_COLOR,
    fit_color: str = DEFAULT_FIT_COLOR,
    mmin_color: str = DEFAULT_MMIN_COLOR,
    shade_posterior: bool = True,
    shade68_alpha: float = DEFAULT_SHADE68_ALPHA,
    shade95_alpha: float = DEFAULT_SHADE95_ALPHA,
    **kwargs,
):
    """Plot the complementary cumulative CMF and optionally overlay a fit."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    fit_result = None
    if fit:
        fit_result = fit_power_law(
            survey_or_masses,
            mmin=mmin,
            min_tail=min_tail,
            uncertainty=uncertainty,
            n_steps=n_steps,
            burn_in=burn_in,
            proposal_width=proposal_width,
            random_seed=random_seed,
        )
    resolved_tail_mass = _resolve_plot_tail_mass(
        survey_or_masses,
        normalize=normalize,
        tail_mass=tail_mass,
        mmin=mmin,
        fit_result=fit_result,
    )
    ccdf = complementary_cmf(
        survey_or_masses,
        normalize=normalize,
        tail_mass=resolved_tail_mass,
    )
    plot_label = label or _label_for(survey_or_masses)
    defaults = {"where": "post", "lw": 1.4, "label": plot_label, "color": data_color}
    defaults.update(kwargs)
    ax.step(ccdf["mass_msun"], ccdf["percent_greater_equal"], **defaults)
    if fit and fit_result is not None:
        masses = masses_from(survey_or_masses)
        p_at_mmin = float(np.mean(masses >= fit_result.mmin))
        grid = np.logspace(np.log10(fit_result.mmin), np.log10(ccdf["mass_msun"].max()), 200)
        if shade_posterior and fit_result.gamma_samples is not None:
            _shade_ccdf_posterior(
                ax,
                grid,
                fit_result.gamma_samples,
                mmin=fit_result.mmin,
                p_at_mmin=p_at_mmin,
                color=fit_color,
                shade68_alpha=shade68_alpha,
                shade95_alpha=shade95_alpha,
            )
        ax.plot(
            grid,
            100.0 * powerlaw_ccdf_shape(grid, fit_result.gamma, fit_result.mmin, p_at_mmin),
            lw=1.5,
            color=fit_color,
            label=rf"{plot_label} fit: $\Gamma={fit_result.gamma:.2f}$",
        )
        ax.axvline(fit_result.mmin, ls="--", lw=1.0, color=mmin_color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Core mass [$M_\odot$]")
    ax.set_ylabel(_ccdf_ylabel(normalize))
    return ax, fit_result


def plot_compiled_differential_cmf(
    surveys: Iterable[Survey],
    ax=None,
    *,
    completeness: str | None = None,
    mass_min: float | None = None,
    label: str = "Compiled sample",
    **kwargs,
):
    """Plot a differential CMF built from pooled masses of multiple surveys."""

    pooled_masses = compile_survey_masses(
        surveys,
        mass_min=mass_min,
        completeness=completeness,
    )
    return plot_differential_cmf(
        pooled_masses,
        ax=ax,
        label=label,
        **kwargs,
    )


def plot_compiled_ccdf(
    surveys: Iterable[Survey],
    ax=None,
    *,
    completeness: str | None = None,
    mass_min: float | None = None,
    label: str = "Compiled sample",
    **kwargs,
):
    """Plot a complementary cumulative CMF built from pooled survey masses."""

    pooled_masses = compile_survey_masses(
        surveys,
        mass_min=mass_min,
        completeness=completeness,
    )
    return plot_ccdf(
        pooled_masses,
        ax=ax,
        label=label,
        **kwargs,
    )


def plot_slope_posterior(
    fit_or_data: PowerLawFit | Survey | Iterable[float],
    ax=None,
    *,
    slope: str = "gamma",
    bins: int = 30,
    density: bool = True,
    color: str = DEFAULT_FIT_COLOR,
    error_color: str = "gray",
    show_salpeter: bool = True,
    salpeter_color: str = "C1",
    salpeter_linestyle: str = "--",
    label: str | None = None,
    mmin: float | str | None = "ks",
    min_tail: int = 20,
    n_steps: int = 6000,
    burn_in: int = 1000,
    proposal_width: float = 0.04,
    random_seed: int | None = 12345,
):
    """Plot the MCMC posterior distribution of the fitted slope.

    If a survey or raw masses are provided, the power-law fit is first run with
    ``uncertainty="mcmc"`` using the supplied fitting controls.
    """

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    fit_result = _fit_from_input(
        fit_or_data,
        mmin=mmin,
        min_tail=min_tail,
        n_steps=n_steps,
        burn_in=burn_in,
        proposal_width=proposal_width,
        random_seed=random_seed,
    )
    samples, x_median, ci, x_label = _posterior_for_slope(fit_result, slope=slope)
    plot_label = label or f"{slope} posterior"
    ax.hist(samples, bins=bins, density=density, color=color, alpha=0.22, histtype="stepfilled",)
    ax.hist(samples, bins=bins, density=density, color=color, alpha=1.0, histtype="step", lw=1.5, label=plot_label)
    ax.axvline(x_median, color=color, lw=1.6)
    if ci is not None:
        ax.axvspan(ci[0], ci[1], color=error_color, alpha=0.2, label=r'$1\sigma$ uncertainty')
        ax.axvline(ci[0], color=error_color, ls=":", lw=1.5)
        ax.axvline(ci[1], color=error_color, ls=":", lw=1.5)
        # In-panel value
        label_text = (
            rf"${x_median:.2f}"
            rf"^{{+{ci[1]-x_median:.2f}}}"
            rf"_{{-{x_median-ci[0]:.2f}}}$"
        )
        
        ax.text(
            0.95,
            0.92,
            label_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.85
            )
        )
    if show_salpeter:
        salpeter_value = 1.35 if slope == "gamma" else 2.35
        ax.axvline(salpeter_value, color=salpeter_color, ls=salpeter_linestyle, lw=1.2, label="Salpeter")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Posterior density" if density else "Count")
    return ax, fit_result


def plot_multi_slope_posterior(
    fits_or_surveys: Iterable[PowerLawFit | Survey] | dict[str, dict],
    ax=None,
    *,
    slope: str = "gamma",
    bins: int = 30,
    density: bool = True,
    labels: dict[str, str] | None = None,
    styles: dict[str, dict] | None = None,
    show_salpeter: bool = True,
    salpeter_color: str = "C1",
    salpeter_linestyle: str = "--",
    show_ci: bool = True,
    ci_alpha: float = 0.12,
    hist_alpha: float = 0.16,
    line_width: float = 1.5,
    mmin: float | str | dict[str, float | str | None] | None = "ks",
    min_tail: int = 20,
    n_steps: int = 6000,
    burn_in: int = 1000,
    proposal_width: float = 0.04,
    random_seed: int | None = 12345,
):
    """Overlay MCMC slope posteriors for multiple surveys or existing fits."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    items = _normalize_multi_posterior_items(fits_or_surveys)
    if not items:
        raise ValueError("At least one fit or survey is required.")
    results: dict[str, dict] = {}
    x_label = None
    for index, item_info in enumerate(items):
        item = item_info["item"]
        fit_result = _fit_from_input(
            item,
            mmin=_value_for_fit_or_survey(item, mmin),
            min_tail=min_tail,
            n_steps=n_steps,
            burn_in=burn_in,
            proposal_width=proposal_width,
            random_seed=None if random_seed is None else random_seed + index,
        )
        style = _multi_style_for_posterior(
            item,
            index,
            labels=labels,
            styles=styles,
            fallback_label=item_info["label"],
            key=item_info["key"],
        )
        samples, median, ci, x_label = _posterior_for_slope(fit_result, slope=slope)
        ax.hist(
            samples,
            bins=bins,
            density=density,
            color=style["fit_color"],
            alpha=hist_alpha,
            histtype="stepfilled",
        )
        ax.hist(
            samples,
            bins=bins,
            density=density,
            color=style["fit_color"],
            histtype="step",
            lw=line_width,
            label=style["label"],
        )
        ax.axvline(median, color=style["fit_color"], lw=line_width)
        if show_ci and ci is not None:
            ax.axvspan(ci[0], ci[1], color=style["fit_color"], alpha=ci_alpha)
        results[item_info["key"]] = {
            "fit": fit_result,
            "samples": samples,
            "label": style["label"],
        }
    if show_salpeter:
        salpeter_value = 1.35 if slope == "gamma" else 2.35
        ax.axvline(
            salpeter_value,
            color=salpeter_color,
            ls=salpeter_linestyle,
            lw=1.2,
            label="Salpeter",
        )
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.set_ylabel("Posterior density" if density else "Count")
    return ax, results


def plot_multi_mass_radius(
    surveys: Iterable[Survey],
    ax=None,
    *,
    labels: dict[str, str] | None = None,
    styles: dict[str, dict] | None = None,
):
    """Plot the mass-radius relation for multiple surveys on one axis."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    results: dict[str, dict] = {}
    for index, survey in enumerate(_as_surveys(surveys)):
        style = _multi_style_for(survey, index, styles=styles, labels=labels)
        plot_mass_radius(
            survey,
            ax=ax,
            data_color=style["data_color"],
            marker_facecolor=style["marker_facecolor"],
            label=style["label"],
            **style["kwargs"],
        )
        results[survey.key] = {"table": survey.mass_radius_table()}
    return ax, results


def plot_multi_cmf(
    surveys: Iterable[Survey],
    ax=None,
    *,
    kind: str = "differential",
    dlogm: float = 0.2,
    normalize: str | bool | None = None,
    fit: bool = True,
    mmin: float | str | dict[str, float | str | None] | None = None,
    min_tail: int = 20,
    uncertainty: str | None = None,
    n_steps: int = 6000,
    burn_in: int = 1000,
    proposal_width: float = 0.04,
    random_seed: int | None = 12345,
    tail_mass: float | dict[str, float | str | None] | None = None,
    labels: dict[str, str] | None = None,
    styles: dict[str, dict] | None = None,
):
    """Plot multiple surveys in one CMF comparison axis.

    Parameters
    ----------
    kind
        Either ``"differential"`` or ``"ccdf"``.
    mmin
        May be a shared value/rule or a dictionary keyed by survey key or short
        name for per-survey fitting lower limits.
    """

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    results: dict[str, dict] = {}
    surveys_list = _as_surveys(surveys)
    for index, survey in enumerate(surveys_list):
        style = _multi_style_for(survey, index, styles=styles, labels=labels)
        survey_mmin = _value_for_survey(survey, mmin)
        survey_tail_mass = _value_for_survey(survey, tail_mass)
        if kind == "differential":
            _, fit_result = plot_differential_cmf(
                survey,
                ax=ax,
                dlogm=dlogm,
                normalize=normalize,
                fit=fit,
                mmin=survey_mmin,
                min_tail=min_tail,
                uncertainty=uncertainty,
                n_steps=n_steps,
                burn_in=burn_in,
                proposal_width=proposal_width,
                random_seed=random_seed,
                tail_mass=survey_tail_mass,
                label=style["label"],
                data_color=style["data_color"],
                marker_facecolor=style["marker_facecolor"],
                fit_color=style["fit_color"],
                mmin_color=style["mmin_color"],
                **style["kwargs"],
            )
            product = differential_cmf(
                survey,
                dlogm=dlogm,
                normalize=normalize,
                tail_mass=_resolve_plot_tail_mass(
                    survey,
                    normalize=normalize,
                    tail_mass=survey_tail_mass,
                    mmin=survey_mmin,
                    fit_result=fit_result,
                ),
            )
        elif kind in {"ccdf", "complementary", "cumulative"}:
            _, fit_result = plot_ccdf(
                survey,
                ax=ax,
                fit=fit,
                normalize=normalize if normalize is not None else True,
                mmin=survey_mmin,
                min_tail=min_tail,
                uncertainty=uncertainty,
                n_steps=n_steps,
                burn_in=burn_in,
                proposal_width=proposal_width,
                random_seed=random_seed,
                tail_mass=survey_tail_mass,
                label=style["label"],
                data_color=style["data_color"],
                fit_color=style["fit_color"],
                mmin_color=style["mmin_color"],
                **style["kwargs"],
            )
            product = complementary_cmf(
                survey,
                normalize=normalize if normalize is not None else True,
                tail_mass=_resolve_plot_tail_mass(
                    survey,
                    normalize=normalize if normalize is not None else True,
                    tail_mass=survey_tail_mass,
                    mmin=survey_mmin,
                    fit_result=fit_result,
                ),
            )
        else:
            raise ValueError("kind must be 'differential' or 'ccdf'.")
        results[survey.key] = {
            "survey": survey,
            "fit": fit_result,
            "cmf": product,
            "label": style["label"],
        }
    return ax, results


def plot_multi_differential_cmf(
    surveys: Iterable[Survey],
    ax=None,
    **kwargs,
):
    """Convenience wrapper for multi-survey differential CMF plots."""

    return plot_multi_cmf(surveys, ax=ax, kind="differential", **kwargs)


def plot_multi_ccdf(
    surveys: Iterable[Survey],
    ax=None,
    **kwargs,
):
    """Convenience wrapper for multi-survey complementary cumulative plots."""

    return plot_multi_cmf(surveys, ax=ax, kind="ccdf", **kwargs)


def _label_for(value) -> str:
    return value.short_name if isinstance(value, Survey) else "sample"


def _differential_ylabel(normalize: str | None) -> str:
    if _is_tail_normalization_mode(normalize):
        return r"Tail-normalized $dN/d\log M$"
    if normalize == "area":
        return r"Area-normalized $dN/d\log M$"
    if normalize == "peak":
        return r"Peak-normalized $dN/d\log M$"
    return r"$dN/d\log M$"


def _ccdf_ylabel(normalize: bool | str) -> str:
    if _is_tail_normalization_mode(normalize):
        return r"$P(>M_{\rm core}\mid M_{\rm core}\geq M_{\rm cut})$"
    return r"$P(>M_{\rm core})$ [%]"


def _as_surveys(surveys: Iterable[Survey]) -> list[Survey]:
    result = list(surveys)
    if not result:
        raise ValueError("At least one survey is required.")
    return result


def _multi_style_for(
    survey: Survey,
    index: int,
    *,
    styles: dict[str, dict] | None,
    labels: dict[str, str] | None,
) -> dict:
    override = _style_override_for_survey(survey, styles)
    data_color = override.get("data_color", DEFAULT_MULTI_COLORS[index % len(DEFAULT_MULTI_COLORS)])
    marker_facecolor = override.get("marker_facecolor", DEFAULT_MARKER_FACECOLOR)
    fit_color = override.get("fit_color", data_color)
    mmin_color = override.get("mmin_color", data_color)
    label = override.get("label", _label_override_for_survey(survey, labels))
    kwargs = dict(override.get("kwargs", {}))
    for key in ("data_color", "marker_facecolor", "fit_color", "mmin_color", "label"):
        kwargs.pop(key, None)
    return {
        "data_color": data_color,
        "marker_facecolor": marker_facecolor,
        "fit_color": fit_color,
        "mmin_color": mmin_color,
        "label": label,
        "kwargs": kwargs,
    }


def _multi_style_for_posterior(
    fit_or_survey: PowerLawFit | Survey,
    index: int,
    *,
    labels: dict[str, str] | None,
    styles: dict[str, dict] | None,
    fallback_label: str | None = None,
    key: str | None = None,
) -> dict:
    if isinstance(fit_or_survey, Survey):
        return _multi_style_for(fit_or_survey, index, labels=labels, styles=styles)
    fit_color = DEFAULT_MULTI_COLORS[index % len(DEFAULT_MULTI_COLORS)]
    label = fallback_label or f"fit {index + 1}"
    if labels:
        for label_key in filter(None, (key, f"fit_{index}")):
            if label_key in labels:
                label = labels[label_key]
                break
    override = {}
    if styles:
        for style_key in filter(None, (key, f"fit_{index}")):
            if style_key in styles:
                override = dict(styles[style_key])
                break
    if override:
        fit_color = override.get("fit_color", override.get("data_color", fit_color))
        label = override.get("label", label)
    return {
        "fit_color": fit_color,
        "label": label,
    }


def _style_override_for_survey(survey: Survey, styles: dict[str, dict] | None) -> dict:
    if not styles:
        return {}
    for key in (survey.key, survey.short_name, survey.name):
        if key in styles:
            return dict(styles[key])
    return {}


def _label_override_for_survey(survey: Survey, labels: dict[str, str] | None) -> str:
    if labels:
        for key in (survey.key, survey.short_name, survey.name):
            if key in labels:
                return labels[key]
    return survey.short_name


def _value_for_survey(
    survey: Survey,
    value: float | str | dict[str, float | str | None] | None,
):
    if not isinstance(value, dict):
        return value
    for key in (survey.key, survey.short_name, survey.name):
        if key in value:
            return value[key]
    return None


def _resolve_plot_tail_mass(
    survey_or_masses: Survey | Iterable[float],
    *,
    normalize: bool | str | None,
    tail_mass: float | str | None,
    mmin: float | str | None,
    fit_result: PowerLawFit | None,
) -> float | None:
    if not _is_tail_normalization_mode(normalize):
        return None if tail_mass is None or isinstance(tail_mass, str) else float(tail_mass)
    if tail_mass is not None:
        if isinstance(tail_mass, str):
            if tail_mass.casefold() == "mmin":
                if fit_result is not None:
                    return fit_result.mmin
                if isinstance(mmin, (int, float)):
                    return float(mmin)
                raise ValueError("tail_mass='mmin' requires a resolved fitting threshold.")
            raise ValueError("tail_mass string must be 'mmin'.")
        return float(tail_mass)
    if isinstance(normalize, str):
        mode = normalize.casefold()
        if mode in {"tail", "above_mmin"}:
            if fit_result is not None:
                return fit_result.mmin
            if isinstance(mmin, (int, float)):
                return float(mmin)
        if isinstance(survey_or_masses, Survey):
            if mode == "above_completeness" and survey_or_masses.completeness_msun is not None:
                return survey_or_masses.completeness_msun
            if mode == "above_mmin":
                if survey_or_masses.fit_mass_min_msun is not None:
                    return survey_or_masses.fit_mass_min_msun
                if survey_or_masses.completeness_msun is not None:
                    return survey_or_masses.completeness_msun
    raise ValueError("Could not resolve a tail normalization threshold for this plot.")


def _is_tail_normalization_mode(normalize: bool | str | None) -> bool:
    return isinstance(normalize, str) and normalize.casefold() in {
        "tail",
        "above_completeness",
        "above_mmin",
    }


def _value_for_fit_or_survey(
    item: PowerLawFit | Survey,
    value: float | str | dict[str, float | str | None] | None,
):
    if isinstance(item, PowerLawFit):
        return value if not isinstance(value, dict) else None
    return _value_for_survey(item, value)


def _normalize_multi_posterior_items(
    fits_or_surveys: Iterable[PowerLawFit | Survey] | dict[str, dict],
) -> list[dict]:
    if isinstance(fits_or_surveys, dict):
        items: list[dict] = []
        for key, value in fits_or_surveys.items():
            if isinstance(value, dict):
                item = value.get("fit") or value.get("survey")
                label = value.get("label")
            else:
                item = value
                label = None
            if item is None:
                raise ValueError(f"results[{key!r}] must contain 'fit' or 'survey'.")
            items.append({"key": key, "item": item, "label": label})
        return items
    items = []
    for index, item in enumerate(fits_or_surveys):
        key = item.key if isinstance(item, Survey) else f"fit_{index}"
        label = item.short_name if isinstance(item, Survey) else None
        items.append({"key": key, "item": item, "label": label})
    return items


def _fit_from_input(
    fit_or_data: PowerLawFit | Survey | Iterable[float],
    *,
    mmin: float | str | None,
    min_tail: int,
    n_steps: int,
    burn_in: int,
    proposal_width: float,
    random_seed: int | None,
) -> PowerLawFit:
    if isinstance(fit_or_data, PowerLawFit):
        return fit_or_data
    return fit_power_law(
        fit_or_data,
        mmin=mmin,
        min_tail=min_tail,
        uncertainty="mcmc",
        n_steps=n_steps,
        burn_in=burn_in,
        proposal_width=proposal_width,
        random_seed=random_seed,
    )


def _posterior_for_slope(
    fit_result: PowerLawFit,
    *,
    slope: str,
) -> tuple[np.ndarray, float, tuple[float, float] | None, str]:
    slope_key = slope.casefold()
    if slope_key == "gamma":
        samples = fit_result.gamma_samples
        median = fit_result.gamma
        ci = fit_result.gamma_ci
        x_label = r"$\Gamma$ posterior"
    elif slope_key == "alpha":
        samples = fit_result.alpha_samples
        median = fit_result.alpha
        ci = fit_result.alpha_ci
        x_label = r"$\alpha$ posterior"
    else:
        raise ValueError("slope must be 'gamma' or 'alpha'.")
    if samples is None:
        raise ValueError("Posterior samples are required; run the fit with uncertainty='mcmc'.")
    return samples, median, ci, x_label


def _shade_differential_posterior(
    ax,
    grid: np.ndarray,
    gamma_samples: np.ndarray,
    *,
    norm_mass: float,
    norm_value: float,
    color: str,
    shade68_alpha: float,
    shade95_alpha: float,
):
    curves = np.array(
        [
            powerlaw_differential_shape(grid, gamma, norm_mass, norm_value)
            for gamma in gamma_samples
        ]
    )
    _shade_posterior_band(
        ax,
        grid,
        curves,
        color=color,
        shade68_alpha=shade68_alpha,
        shade95_alpha=shade95_alpha,
    )


def _shade_ccdf_posterior(
    ax,
    grid: np.ndarray,
    gamma_samples: np.ndarray,
    *,
    mmin: float,
    p_at_mmin: float,
    color: str,
    shade68_alpha: float,
    shade95_alpha: float,
):
    curves = np.array(
        [
            100.0 * powerlaw_ccdf_shape(grid, gamma, mmin, p_at_mmin)
            for gamma in gamma_samples
        ]
    )
    _shade_posterior_band(
        ax,
        grid,
        curves,
        color=color,
        shade68_alpha=shade68_alpha,
        shade95_alpha=shade95_alpha,
    )


def _shade_posterior_band(
    ax,
    grid: np.ndarray,
    curves: np.ndarray,
    *,
    color: str,
    shade68_alpha: float,
    shade95_alpha: float,
):
    q025, q16, q84, q975 = np.percentile(curves, [2.5, 16, 84, 97.5], axis=0)
    ax.fill_between(grid, q025, q975, color=color, alpha=shade95_alpha, lw=0)
    ax.fill_between(grid, q16, q84, color=color, alpha=shade68_alpha, lw=0)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install cmf4all[plot] to use plotting helpers.") from exc
    return plt
