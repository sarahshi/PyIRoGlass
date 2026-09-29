import os
import tempfile
import unittest
import importlib.util
import numpy as np
import pandas as pd
import PyIRoGlass as pig

HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None


class test_templates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spectra_dir = os.path.join(self.tmp.name, "spectra")
        os.makedirs(self.spectra_dir)
        for name in ["SAMPLE_b.CSV", "SAMPLE_a.CSV"]:
            with open(os.path.join(self.spectra_dir, name), "w") as f:
                f.write("1000,0.1\n1001,0.2\n")
        self.samples = ["SAMPLE_a", "SAMPLE_b"]
        self.chemthick_cols = ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO",
                               "MgO", "CaO", "Na2O", "K2O", "P2O5",
                               "Thickness", "Sigma_Thickness"]

    def out(self, name):
        return os.path.join(self.tmp.name, name)

    def test_transmission_template_from_directory(self):
        path = self.out("ChemThick.csv")
        template = pig.create_transmission_template(self.spectra_dir, path)
        self.assertListEqual(list(template.index), self.samples,
                             msg="Samples should be sorted file names "
                             "without the extension")
        self.assertListEqual(list(template.columns), self.chemthick_cols)
        self.assertTrue(template.isna().all().all())
        written = pd.read_csv(path)
        self.assertListEqual(list(written.columns),
                             ["Sample"] + self.chemthick_cols,
                             msg="Written template should match the "
                             "ChemThick column format")

    def test_transmission_template_loads_as_chemthick(self):
        path = self.out("ChemThick.csv")
        pig.create_transmission_template(self.spectra_dir, path)
        loader = pig.SampleDataLoader(chemistry_thickness_path=path)
        chemistry, thickness = loader.load_chemistry_thickness()
        self.assertListEqual(list(chemistry.index), self.samples)
        self.assertListEqual(list(thickness.columns),
                             ["Thickness", "Sigma_Thickness"])

    def test_transmission_template_from_dict(self):
        dfs_dict = {"X1": pd.DataFrame(), "X2": pd.DataFrame()}
        template = pig.create_transmission_template(
            dfs_dict, self.out("from_dict.csv"))
        self.assertListEqual(list(template.index), ["X1", "X2"])

    def test_reflectance_template(self):
        path = self.out("RefractiveIndex.csv")
        template = pig.create_reflectance_template(self.spectra_dir, path,
                                                   default_n=1.546)
        self.assertListEqual(list(template.index), self.samples)
        self.assertListEqual(list(template.columns), ["n"])
        self.assertTrue(np.allclose(template["n"], 1.546))
        written = pd.read_csv(path, index_col="Sample")
        self.assertTrue(np.allclose(written["n"], 1.546))

    def test_reflectance_template_default_nan(self):
        template = pig.create_reflectance_template(
            self.spectra_dir, self.out("RefractiveIndex.csv"))
        self.assertTrue(template["n"].isna().all())

    def test_default_export_path(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp.name)
        pig.create_transmission_template(self.spectra_dir)
        pig.create_reflectance_template(self.spectra_dir)
        self.assertTrue(os.path.exists("ChemThick_Template.csv"))
        self.assertTrue(os.path.exists("RefractiveIndex_Template.csv"))

    @unittest.skipUnless(HAS_OPENPYXL, "openpyxl not installed")
    def test_xlsx_export(self):
        path = self.out("ChemThick.xlsx")
        pig.create_transmission_template(self.spectra_dir, path)
        written = pd.read_excel(path, index_col="Sample")
        self.assertListEqual(list(written.index), self.samples)

    def test_empty_directory_raises(self):
        empty = self.out("empty")
        os.makedirs(empty)
        with self.assertRaises(ValueError):
            pig.create_transmission_template(empty, self.out("x.csv"))
        with self.assertRaises(ValueError):
            pig.create_reflectance_template(empty, self.out("y.csv"))


if __name__ == "__main__":
    unittest.main()
