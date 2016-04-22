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
        #
        # Check the four corners of a square
        #
        data = cudasift.PySiftData(100)
        img = np.zeros((100, 100), np.uint8)
        img[10:-9, 10] = 128
        img[10, 10:-9] = 128
        img[10:-9, -10] = 128
        img[-10, 10:-9] = 128
        cudasift.ExtractKeypoints(img, data)
        self.assertEqual(len(data), 4)
        df, keypoints = data.to_data_frame()
        idx = np.lexsort((df.xpos, df.ypos))
        #
        # Check that the four corners are just inside the square
        #
        for i in (0, 1):
            self.assertTrue(df.ypos[idx[i]] > 10 and df.ypos[idx[i]] < 15)
        for i in (2, 3):
            self.assertTrue(df.ypos[idx[i]] > 85 and df.ypos[idx[i]] < 90)
        for i in (0, 2):
            self.assertTrue(df.xpos[idx[i]] > 10 and df.xpos[idx[i]] < 15)
        for i in (1, 3):
            self.assertTrue(df.xpos[idx[i]] > 85 and df.xpos[idx[i]] < 90)
        
if __name__== "__main__":
    unittest.main()
    
