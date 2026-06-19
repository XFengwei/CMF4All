# CMF4All

Python tools for collecting, loading, and comparing core mass function survey
catalogues.

The current first API is survey-oriented:

```python
from cmf4all import SurveyRegistry

registry = SurveyRegistry.from_default()
survey = registry.get("ALMAIMF_all")

print(survey.name)
print(survey.completeness_msun)
table = survey.load_table()
masses = survey.masses()
```

The package currently reads the existing metadata file at
`cmf_zoo/data/metadata/surveys.yaml` and the five standardized catalogue tables
under `cmf_zoo/data/catalogues/`.


## Custom Data Directories

The bundled examples live under `cmf_zoo/`, but published users should point
`cmf4all` at their own metadata and catalogue directory:

```python
from cmf4all import SurveyRegistry

registry = SurveyRegistry.from_directory(
    "/path/to/my_cmf_project",
    metadata_file="data/metadata/surveys.yaml",
)
```

In this mode, relative `data_file` entries in the YAML file are resolved against
the `project_root` passed to `from_directory`. For a standalone YAML file, use:

```python
registry = SurveyRegistry.from_yaml(
    "/path/to/surveys.yaml",
    project_root="/path/to/data_root",
)
```

If `project_root` is omitted and the metadata file has the conventional layout
`PROJECT/data/metadata/surveys.yaml`, the project root is inferred as `PROJECT`.


## Survey Diagnostics

The framework uses a hybrid design: `Survey` handles metadata and catalogue
access, while CMF calculations and plotting live in external modules.

Within `cmf4all`, the logarithmic CMF slope uses the convention
`dN/dlogM ∝ M^{-Gamma}`, so `Gamma = alpha - 1` when `dN/dM ∝ M^{-alpha}`.

```python
from cmf4all import SurveyRegistry, differential_cmf, complementary_cmf, fit_power_law
from cmf4all.relations import mass_radius_relation
from cmf4all.plotting import (
    plot_ccdf,
    plot_differential_cmf,
    plot_mass_radius,
    plot_slope_posterior,
)

survey = SurveyRegistry.from_default().get("ALMA-IMF")

mr = mass_radius_relation(survey)
dcmf = differential_cmf(survey, dlogm=0.2, normalize="area")
ccdf = complementary_cmf(survey)
fit = fit_power_law(survey, mmin="ks", min_tail=30)

plot_mass_radius(survey)
plot_differential_cmf(survey, normalize="area", mmin="ks", min_tail=30)
plot_ccdf(survey, mmin="ks", min_tail=30)
plot_slope_posterior(survey, slope="gamma", mmin="ks", min_tail=30)
```

Surveys can be registered before their catalogue tables are collected. Use
`registry.available()` for surveys with local data files and
`registry.missing_data()` for metadata entries whose tables are not present yet.


Plot style can be customized per call. The default differential-CMF style uses
hollow data markers with dark outlines/error bars and a blue fit line:

```python
plot_differential_cmf(
    survey,
    data_color="#222222",
    marker_facecolor="white",
    fit_color="#0072B2",
)
```

The plotting helpers can also request MCMC-based uncertainties for the fitted
power-law tail:

```python
ax, fit = plot_differential_cmf(
    survey,
    mmin="ks",
    min_tail=30,
    uncertainty="mcmc",
    n_steps=6000,
    burn_in=1000,
)
```

For comparison plots that should emphasize only the complete high-mass regime,
tail-normalized modes are available:

```python
dcmf = differential_cmf(survey, normalize="above_completeness")
ccdf = complementary_cmf(survey, normalize="above_completeness")

ax, results = plot_multi_differential_cmf(
    surveys,
    normalize="above_mmin",
    mmin="ks",
    uncertainty="mcmc",
)
```

When MCMC samples are present, `plot_differential_cmf(...)` and `plot_ccdf(...)`
shade the 68% and 95% posterior envelopes around the fitted relation.

To inspect the fitted slope distribution directly, including whether the
posterior is close to Salpeter, use:

```python
ax, fit = plot_slope_posterior(
    survey,
    slope="gamma",
    mmin="ks",
    min_tail=30,
    n_steps=6000,
    burn_in=1000,
    show_salpeter=True,
)
```

## Multi-survey Comparison

The comparison layer works directly with iterables of `Survey` objects:

```python
surveys = [
    registry.get("ALMA-IMF"),
    registry.get("ALMAGAL"),
    registry.get("ASHES"),
]
```

If you want to *pool* several surveys into one compiled CMF sample instead of
overplotting them separately, use:

```python
from cmf4all import compile_survey_masses, compiled_cmf, fit_power_law

pooled_masses = compile_survey_masses(
    surveys,
    completeness="shared_max",
)

dcmf = compiled_cmf(
    surveys,
    kind="differential",
    dlogm=0.2,
    normalize="area",
    completeness="shared_max",
)

fit = fit_power_law(pooled_masses, mmin=2.0, uncertainty="mcmc")
```

For direct plotting of the pooled sample, wrappers are also available:

```python
from cmf4all import plot_compiled_ccdf, plot_compiled_differential_cmf

ax, fit = plot_compiled_differential_cmf(
    surveys,
    completeness="shared_max",
    normalize="area",
    mmin="ks",
    uncertainty="mcmc",
)

ax2, fit2 = plot_compiled_ccdf(
    surveys,
    completeness="shared_max",
    mmin="ks",
    uncertainty="mcmc",
)
```

The `completeness` option controls how survey-specific selection limits are
handled before pooling:

- `"none"`: no completeness cut
- `"per-survey"`: each survey uses its own `completeness_msun`
- `"shared_max"`: all surveys use the highest completeness limit in the set
- `"fit"`: each survey uses `fit_mass_min_msun`, falling back to
  `completeness_msun`

For overplotting CMFs:

```python
from cmf4all.plotting import plot_multi_ccdf, plot_multi_differential_cmf, plot_multi_mass_radius

ax, results = plot_multi_differential_cmf(
    surveys,
    normalize="area",
    mmin="ks",
    uncertainty="mcmc",
)
```

Per-survey fitting thresholds and labels can be passed as dictionaries keyed by
survey key or short name:

```python
ax, results = plot_multi_ccdf(
    surveys,
    mmin={"LANCET_all": 0.97, "ALMA-IMF": "ks"},
    labels={"ALMAIMF_all": "ALMA-IMF sample"},
)
```

To compare the posterior slope distributions directly:

```python
from cmf4all.plotting import plot_multi_slope_posterior

ax, results = plot_multi_differential_cmf(
    surveys,
    mmin="ks",
    uncertainty="mcmc",
)

ax2, posterior_results = plot_multi_slope_posterior(
    results,
    slope="gamma",
    show_salpeter=True,
)
```


## MCMC Uncertainties

The default power-law fit uses the analytic MLE uncertainty. To estimate the
slope uncertainty with a one-parameter Metropolis MCMC sampler, use:

```python
fit = fit_power_law(
    survey,
    mmin="ks",
    min_tail=30,
    uncertainty="mcmc",
    n_steps=6000,
    burn_in=1000,
    random_seed=12345,
)

print(fit.alpha, fit.alpha_err, fit.alpha_ci)
print(fit.gamma, fit.gamma_err, fit.gamma_ci)
alpha_samples = fit.alpha_samples
```

The sampler uses the continuous power-law likelihood for masses above `mmin`
and a flat prior for `alpha > 1`. The returned `alpha` and `gamma` are posterior
medians; `alpha_err` and `gamma_err` are half the 16--84 percentile widths.
