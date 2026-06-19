from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cmf4all import SurveyRegistry


class SurveyRegistryTests(unittest.TestCase):
    def test_default_registry_loads_registered_surveys(self):
        registry = SurveyRegistry.from_default()
        self.assertGreaterEqual(len(registry), 5)
        self.assertIn("ALMAIMF_all", registry.keys)

    def test_survey_can_be_loaded_by_alias(self):
        registry = SurveyRegistry.from_default()
        survey = registry.get("ALMA-IMF")
        self.assertEqual(survey.key, "ALMAIMF_all")
        self.assertEqual(survey.completeness_msun, 1.64)

    def test_survey_masses_are_positive(self):
        survey = SurveyRegistry.from_default().get("LANCET")
        masses = survey.masses()
        self.assertGreater(masses.size, 0)
        self.assertGreater(masses.min(), 0)

    def test_registry_validates_available_data(self):
        SurveyRegistry.from_default().validate(require_data=False)

    def test_registry_tracks_missing_data(self):
        registry = SurveyRegistry.from_default()
        missing_keys = {survey.key for survey in registry.missing_data()}
        available_keys = {survey.key for survey in registry.available()}
        self.assertTrue(missing_keys.isdisjoint(available_keys))
        self.assertEqual(missing_keys | available_keys, set(registry.keys))

    def test_registry_loads_custom_project_directory(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "catalogues"
            metadata_dir = root / "metadata"
            data_dir.mkdir()
            metadata_dir.mkdir()
            (data_dir / "toy.csv").write_text("core_id,mass_msun\na,1.0\nb,2.0\n")
            (metadata_dir / "surveys.yaml").write_text(
                "Toy_all:\n"
                "  label: Toy survey\n"
                "  short_label: Toy\n"
                "  data_file: catalogues/toy.csv\n"
                "  data_type: catalogue\n"
                "  mass_column: mass_msun\n"
                "  completeness_msun: 1.0\n"
            )

            registry = SurveyRegistry.from_directory(root, "metadata/surveys.yaml")
            survey = registry.get("Toy")

            self.assertEqual(registry.project_root, root)
            self.assertEqual(survey.data_file, data_dir / "toy.csv")
            self.assertEqual(survey.masses().tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
