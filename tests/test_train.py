import unittest


import sys
sys.path.append('/workspace/B_CNN')
from b_cnn.tasks.train import train

class TestTrain(unittest.TestCase):
    def test_train_is_runable(self):
        train()
        # self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
