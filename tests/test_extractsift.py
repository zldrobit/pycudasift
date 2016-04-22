'''Test extraction of SIFT keypoints'''
import numpy as np
import unittest
import cudasift

class TestExtractKeypoints(unittest.TestCase):
    def test_01_01_nothing(self):
        # "How? Nothing will come of nothing.", Lear 1:1
        #
        data = cudasift.PySiftData(100)
        img = np.zeros((100, 100), np.uint8)
        cudasift.ExtractKeypoints(img, data)
        self.assertEqual(len(data), 0)
        
    def test_01_02_speak_again(self):
        data = cudasift.PySiftData(100)
        img = np.zeros((100, 100), np.uint8)
        img[10:-10, 10] = 128
        img[10, 10:-10] = 128
        img[10:-10, -10] = 128
        img[-10, 10:-10] = 128
        cudasift.ExtractKeypoints(img, data)
        df, keypoints = data.to_data_frame()
        pass
        
if __name__== "__main__":
    unittest.main.main()
        