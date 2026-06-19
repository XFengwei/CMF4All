"""Example diagnostics for one CMF4All survey."""

from pathlib import Path

import matplotlib.pyplot as plt

from cmf4all import SurveyRegistry, fit_power_law
from cmf4all.plotting import (
    plot_ccdf,
    plot_differential_cmf,
    plot_mass_radius,
    plot_slope_posterior,
)


OUTDIR = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR.mkdir(exist_ok=True)

registry = SurveyRegistry.from_default()
survey = registry.get("ALMA-IMF")

print(survey.summary())
print(f"Missing registered data: {[item.key for item in registry.missing_data()]}")

fit = fit_power_law(
    survey,
    mmin="ks",
    min_tail=30,
    uncertainty="mcmc",
    n_steps=6000,
    burn_in=1000,
    random_seed=12345,
)
print(
    f"KS-selected MCMC fit: mmin={fit.mmin:.3g} Msun, "
    f"alpha={fit.alpha:.2f}+/-{fit.alpha_err:.2f}, "
    f"Gamma(alpha-1)={fit.gamma:.2f}+/-{fit.gamma_err:.2f}, "
    f"N={fit.n_fit}, KS={fit.ks_distance:.3f}"
)

fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
plot_mass_radius(survey, ax=axes[0])
plot_differential_cmf(
    survey,
    ax=axes[1],
    normalize="area",
    mmin="ks",
    min_tail=30,
    uncertainty="mcmc",
    n_steps=6000,
    burn_in=1000,
    data_color="#222222",
    marker_facecolor="white",
    fit_color="#0072B2",
)
plot_ccdf(
    survey,
    ax=axes[2],
    mmin="ks",
    min_tail=30,
    uncertainty="mcmc",
    n_steps=6000,
    burn_in=1000,
    data_color="#222222",
    fit_color="#0072B2",
)
plot_slope_posterior(
    fit,
    ax=axes[3],
    slope="gamma",
    color="#0072B2",
    show_salpeter=True,
)
for ax in axes:
    ax.legend(frameon=False, fontsize=8)
fig.savefig(OUTDIR / "almaimf_diagnostics.png", dpi=200)
print(f"Saved {OUTDIR / 'almaimf_diagnostics.png'}")
