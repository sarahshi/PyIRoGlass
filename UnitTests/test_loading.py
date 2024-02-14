import unittest
import os
import glob
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
        self.assertIsNotNone(wavenumber,
                             "Loading the wavenumber array failed.")
        self.assertNotEqual(wavenumber.size, 0, "Wavenumber array is empty.")


class test_loading_csv(unittest.TestCase):

    def test_load_samplecsvs(self):

        file_path = os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)),
            '../Inputs/TransmissionSpectra/Fuego/')
        # FILES = sorted(glob.glob(file_path + "*"))

        loader = pig.SampleDataLoader(spectrum_path=file_path)
        files, dfs_dict = loader.load_spectrum_directory()
        # files, dfs_dict = pig.Load_SampleCSV(FILES, 5500, 1000)
        self.assertEqual(len(files), 97)  # Adjust based on your test data

    def test_load_chemthick(self):

        file_path = os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)),
            '../Inputs/ChemThick_Template.csv')
        loader = pig.SampleDataLoader(chemistry_thickness_path=file_path)
        chemistry, thickness = loader.load_chemistry_thickness(file_path)

        # Assuming that the PCA matrix should not be empty after reading a
        # valid .npz file. Adjust based on your test data
        self.assertEqual(chemistry.shape, (9, 11))
        # Adjust based on your test data
        self.assertEqual(thickness.shape, (9, 2))


if __name__ == '__main__':
    unittest.main()
