import unittest

import numpy as np

from cmf4all import SurveyRegistry
from cmf4all.mass_function import (
    compile_survey_masses,
    compiled_cmf,
    complementary_cmf,
    differential_cmf,
    fit_power_law,
    fit_power_law_mcmc,
    fit_power_law_mle,
    sample_power_law_alpha_mcmc,
    select_mmin_ks,
)
from cmf4all.relations import mass_radius_relation


class MassFunctionTests(unittest.TestCase):
    def setUp(self):
        self.survey = SurveyRegistry.from_default().get("ALMA-IMF")
        self.surveys = [
            self.survey,
            SurveyRegistry.from_default().get("ASHES"),
        ]

    def test_survey_has_mass_radius_table(self):
        table = self.survey.mass_radius_table()
        self.assertIn("mass_msun", table.columns)
        self.assertIn("radius_au", table.columns)
        self.assertGreater(len(table), 0)
        self.assertGreater(table["mass_msun"].min(), 0)
        self.assertGreater(table["radius_au"].min(), 0)

    def test_mass_radius_relation_returns_standard_columns(self):
        relation = mass_radius_relation(self.survey)
        self.assertEqual(list(relation.columns), ["mass_msun", "radius_au"])
        self.assertGreater(len(relation), 0)

    def test_differential_cmf_from_survey(self):
        cmf = differential_cmf(self.survey, dlogm=0.25, normalize="area")
        self.assertIn("dn_dlogm", cmf.columns)
        self.assertGreater(cmf["count"].sum(), 0)
        widths = np.log10(cmf["bin_high_msun"]) - np.log10(cmf["bin_low_msun"])
        area = np.sum(cmf["dn_dlogm"] * widths)
        self.assertAlmostEqual(area, 1.0)

    def test_complementary_cmf_from_survey(self):
        ccdf = complementary_cmf(self.survey)
        self.assertEqual(ccdf["n_greater_equal"].iloc[0], self.survey.masses().size)
        self.assertAlmostEqual(ccdf["p_greater_equal"].iloc[0], 1.0)

    def test_differential_cmf_above_completeness(self):
        cmf = differential_cmf(self.survey, dlogm=0.25, normalize="above_completeness")
        self.assertGreaterEqual(cmf["mass_msun"].min(), self.survey.completeness_msun)
        widths = np.log10(cmf["bin_high_msun"]) - np.log10(cmf["bin_low_msun"])
        area = np.sum(cmf["dn_dlogm"] * widths)
        self.assertAlmostEqual(area, 1.0)

    def test_complementary_cmf_above_completeness(self):
        ccdf = complementary_cmf(self.survey, normalize="above_completeness")
        self.assertGreaterEqual(ccdf["mass_msun"].min(), self.survey.completeness_msun)
        self.assertAlmostEqual(ccdf["p_greater_equal"].iloc[0], 1.0)

    def test_differential_cmf_tail_normalization_from_array(self):
        cmf = differential_cmf([1, 2, 3, 5, 8, 13, 21], dlogm=0.3, normalize="tail", tail_mass=5)
        self.assertGreaterEqual(cmf["mass_msun"].min(), 5.0)

    def test_compile_survey_masses_combines_catalogues(self):
        compiled = compile_survey_masses(self.surveys)
        expected_size = sum(survey.masses().size for survey in self.surveys)
        self.assertEqual(compiled.size, expected_size)

    def test_compile_survey_masses_with_shared_completeness(self):
        compiled = compile_survey_masses(self.surveys, completeness="shared_max")
        shared_limit = max(survey.completeness_msun for survey in self.surveys)
        self.assertGreaterEqual(compiled.min(), shared_limit)

    def test_compiled_differential_cmf(self):
        cmf = compiled_cmf(
            self.surveys,
            kind="differential",
            dlogm=0.25,
            normalize="area",
            completeness="shared_max",
        )
        self.assertIn("dn_dlogm", cmf.columns)
        widths = np.log10(cmf["bin_high_msun"]) - np.log10(cmf["bin_low_msun"])
        area = np.sum(cmf["dn_dlogm"] * widths)
        self.assertAlmostEqual(area, 1.0)

    def test_compiled_ccdf(self):
        ccdf = compiled_cmf(
            self.surveys,
            kind="ccdf",
            normalize=True,
            completeness="per-survey",
        )
        self.assertAlmostEqual(ccdf["p_greater_equal"].iloc[0], 1.0)

    def test_mle_fit_uses_completeness_by_default(self):
        fit = fit_power_law(self.survey)
        self.assertEqual(fit.mmin, self.survey.completeness_msun)
        self.assertGreater(fit.alpha, 1.0)
        self.assertGreater(fit.gamma, 0.0)
        self.assertGreater(fit.n_fit, 0)

    def test_ks_mmin_selection(self):
        fit = select_mmin_ks(self.survey, min_tail=30)
        self.assertGreater(fit.alpha, 1.0)
        self.assertGreaterEqual(fit.n_fit, 30)
        self.assertGreaterEqual(fit.ks_distance, 0.0)

    def test_mle_fit_from_array(self):
        fit = fit_power_law_mle([1, 2, 3, 5, 8, 13, 21], mmin=2)
        self.assertGreater(fit.alpha, 1.0)
        self.assertAlmostEqual(fit.gamma, fit.alpha - 1.0)
        self.assertEqual(fit.n_fit, 6)

    def test_mcmc_alpha_samples_are_returned(self):
        samples = sample_power_law_alpha_mcmc(
            self.survey,
            mmin=self.survey.completeness_msun,
            n_steps=300,
            burn_in=50,
            random_seed=7,
        )
        self.assertEqual(samples.size, 250)
        self.assertTrue(np.all(samples > 1.0))

    def test_mcmc_fit_returns_posterior_summary(self):
        fit = fit_power_law_mcmc(
            self.survey,
            mmin=self.survey.completeness_msun,
            n_steps=300,
            burn_in=50,
            random_seed=7,
        )
        self.assertEqual(fit.method, "mcmc")
        self.assertIsNotNone(fit.alpha_samples)
        self.assertIsNotNone(fit.gamma_samples)
        self.assertIsNotNone(fit.alpha_ci)
        self.assertGreater(fit.alpha, 1.0)
        self.assertGreater(fit.gamma, 0.0)
        self.assertTrue(np.allclose(fit.gamma_samples, fit.alpha_samples - 1.0))
        self.assertGreater(fit.alpha_err, 0.0)

    def test_fit_power_law_can_use_mcmc_uncertainty(self):
        fit = fit_power_law(
            self.survey,
            mmin="ks",
            min_tail=30,
            uncertainty="mcmc",
            n_steps=300,
            burn_in=50,
            random_seed=11,
        )
        self.assertEqual(fit.method, "mcmc")
        self.assertGreaterEqual(fit.n_fit, 30)
        self.assertEqual(fit.alpha_samples.size, 250)

    def test_mcmc_fit_repr_stays_compact(self):
        fit = fit_power_law(
            self.survey,
            mmin=self.survey.completeness_msun,
            uncertainty="mcmc",
            n_steps=300,
            burn_in=50,
            random_seed=5,
        )
        self.assertIn("method='mcmc'", repr(fit))
        self.assertNotIn("alpha_samples=array", repr(fit))


if __name__ == "__main__":
    unittest.main()
