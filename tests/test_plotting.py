import unittest

from cmf4all import SurveyRegistry
from cmf4all.plotting import (
    DEFAULT_DATA_COLOR,
    DEFAULT_FIT_COLOR,
    DEFAULT_MARKER_FACECOLOR,
    plot_ccdf,
    plot_compiled_ccdf,
    plot_compiled_differential_cmf,
    plot_differential_cmf,
    plot_mass_radius,
    plot_multi_ccdf,
    plot_multi_differential_cmf,
    plot_multi_mass_radius,
    plot_multi_slope_posterior,
    plot_slope_posterior,
)


class PlottingStyleTests(unittest.TestCase):
    def test_default_plotting_colors(self):
        self.assertEqual(DEFAULT_DATA_COLOR, "#222222")
        self.assertEqual(DEFAULT_MARKER_FACECOLOR, "white")
        self.assertEqual(DEFAULT_FIT_COLOR, "#0072B2")


class PlottingTests(unittest.TestCase):
    def setUp(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            self.skipTest("matplotlib is not installed")
        self.survey = SurveyRegistry.from_default().get("LANCET")
        self.surveys = [
            self.survey,
            SurveyRegistry.from_default().get("ALMA-IMF"),
        ]

    def test_plot_mass_radius(self):
        ax = plot_mass_radius(self.survey)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")

    def test_plot_differential_cmf(self):
        ax, fit = plot_differential_cmf(
            self.survey,
            fit=True,
            mmin="ks",
            min_tail=20,
            uncertainty="mcmc",
            n_steps=300,
            burn_in=50,
            random_seed=7,
        )
        self.assertEqual(ax.get_xscale(), "log")
        self.assertIsNotNone(fit)
        self.assertEqual(fit.method, "mcmc")

    def test_plot_ccdf(self):
        ax, fit = plot_ccdf(
            self.survey,
            fit=True,
            mmin="ks",
            min_tail=20,
            uncertainty="mcmc",
            n_steps=300,
            burn_in=50,
            random_seed=7,
        )
        self.assertEqual(ax.get_xscale(), "log")
        self.assertIsNotNone(fit)
        self.assertEqual(fit.method, "mcmc")

    def test_plot_compiled_differential_cmf(self):
        ax, fit = plot_compiled_differential_cmf(
            self.surveys,
            fit=True,
            completeness="shared_max",
            mmin="ks",
            min_tail=20,
            uncertainty="mcmc",
            n_steps=300,
            burn_in=50,
            random_seed=7,
        )
        self.assertEqual(ax.get_xscale(), "log")
        self.assertIsNotNone(fit)
        self.assertEqual(fit.method, "mcmc")

    def test_plot_compiled_ccdf(self):
        ax, fit = plot_compiled_ccdf(
            self.surveys,
            fit=True,
            completeness="per-survey",
            mmin="ks",
            min_tail=20,
            uncertainty="mcmc",
            n_steps=300,
            burn_in=50,
            random_seed=7,
        )
        self.assertEqual(ax.get_xscale(), "log")
        self.assertIsNotNone(fit)
        self.assertEqual(fit.method, "mcmc")

    def test_plot_slope_posterior(self):
        ax, fit = plot_slope_posterior(
            self.survey,
            slope="gamma",
            mmin="ks",
            min_tail=20,
            n_steps=300,
            burn_in=50,
            random_seed=7,
        )
        self.assertIsNotNone(fit)
        self.assertEqual(fit.method, "mcmc")
        self.assertGreaterEqual(len(ax.patches), 1)

    def test_plot_multi_mass_radius(self):
        ax, results = plot_multi_mass_radius(self.surveys)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(set(results), {"LANCET_all", "ALMAIMF_all"})

    def test_plot_multi_differential_cmf(self):
        ax, results = plot_multi_differential_cmf(
            self.surveys,
            fit=True,
            normalize="above_mmin",
            mmin={"LANCET_all": 0.97, "ALMAIMF_all": "ks"},
            min_tail=20,
            uncertainty="mle",
            labels={"ALMAIMF_all": "ALMA-IMF sample"},
        )
        self.assertEqual(ax.get_xscale(), "log")
        self.assertIn("ALMAIMF_all", results)
        self.assertIsNotNone(results["ALMAIMF_all"]["fit"])

    def test_plot_multi_ccdf_with_per_survey_mmin(self):
        ax, results = plot_multi_ccdf(
            self.surveys,
            fit=True,
            normalize="above_mmin",
            mmin={"LANCET_all": 0.97, "ALMAIMF_all": "ks"},
            min_tail=20,
            uncertainty="mle",
        )
        self.assertEqual(ax.get_xscale(), "log")
        self.assertIsNotNone(results["LANCET_all"]["fit"])

    def test_plot_multi_slope_posterior(self):
        _, cmf_results = plot_multi_differential_cmf(
            self.surveys,
            fit=True,
            mmin={"LANCET_all": 0.97, "ALMAIMF_all": "ks"},
            min_tail=20,
            uncertainty="mcmc",
            n_steps=300,
            burn_in=50,
            random_seed=7,
            labels={"ALMAIMF_all": "ALMA-IMF sample"},
        )
        ax, results = plot_multi_slope_posterior(cmf_results, slope="gamma")
        self.assertIn("LANCET_all", results)
        self.assertIn("ALMAIMF_all", results)
        self.assertEqual(results["LANCET_all"]["fit"].method, "mcmc")
        self.assertGreaterEqual(len(ax.patches), 2)
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertIn("LANCET", legend_labels)
        self.assertIn("ALMA-IMF sample", legend_labels)


if __name__ == "__main__":
    unittest.main()
