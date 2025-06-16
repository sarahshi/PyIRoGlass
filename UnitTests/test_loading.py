import os
import unittest
import PyIRoGlass as pig


class test_loading_npz(unittest.TestCase):
    def test_load_pc(self):
        vector_loader = pig.VectorLoader()
        PCmatrix = vector_loader.baseline_PC

        # Assuming that the PCA matrix should not be empty after reading a
        # valid .npz file
        self.assertIsNotNone(PCmatrix, "Loading the PCA matrix failed.")
        self.assertNotEqual(PCmatrix.size, 0, "PCA matrix is empty.")

    def test_load_wavenumber(self):
        vector_loader = pig.VectorLoader()
        wavenumber = vector_loader.wavenumber

        # Assuming that the wavenumber array should not be empty after reading
        # a valid .npz file
        self.assertIsNotNone(wavenumber, "Loading the wavenumber array failed.")
        self.assertNotEqual(wavenumber.size, 0, "Wavenumber array is empty.")


class test_loading_csv(unittest.TestCase):
    def test_load_samplecsvs(self):
        dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../Inputs/TransmissionSpectra/Fuego/",
        )

        loader = pig.SampleDataLoader(spectrum_path=dir_path)
        dfs_dict = loader.load_spectrum_directory()
        self.assertEqual(len(dfs_dict), 94)  # Adjust based on your test data

    def test_load_chemthick(self):
        file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../Inputs/ChemThick_Template.csv",
        )
        loader = pig.SampleDataLoader(chemistry_thickness_path=file_path)
        chemistry, thickness = loader.load_chemistry_thickness()

        # Assuming that the PCA matrix should not be empty after reading a
        # valid .npz file. Adjust based on your test data
        self.assertEqual(chemistry.shape, (9, 11))
        # Adjust based on your test data
        self.assertEqual(thickness.shape, (9, 2))

    def test_load_dir_chemthick(self):
        dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../Inputs/TransmissionSpectra/Fuego/",
        )

        file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../Inputs/ChemThick_Template.csv",
        )
        loader = pig.SampleDataLoader(
            spectrum_path=dir_path, chemistry_thickness_path=file_path
        )
        (dfs_dict, chemistry, thickness) = loader.load_all_data()

        self.assertEqual(len(dfs_dict), 94)  # Adjust based on your test data
        self.assertEqual(chemistry.shape, (9, 11))
        self.assertEqual(thickness.shape, (9, 2))

    def test_spectrum_path_none(self):
        with self.assertRaises(ValueError):
            loader = pig.SampleDataLoader(spectrum_path=None)
            loader.load_spectrum_directory()

    def test_chemistry_thickness_path_none(self):
        with self.assertRaises(ValueError):
            loader = pig.SampleDataLoader(chemistry_thickness_path=None)
            loader.load_chemistry_thickness()

    def test_spectrum_chemistry_thickness_path_none(self):
        with self.assertRaises(ValueError):
            loader = pig.SampleDataLoader(
                spectrum_path=None, chemistry_thickness_path=None
            )
            loader.load_all_data()


if __name__ == "__main__":
    unittest.main()
