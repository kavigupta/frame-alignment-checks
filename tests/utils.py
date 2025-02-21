import os
import sys
import unittest

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

is_testing = True


def skip_on_mac(func):
    def wrapper(*args, **kwargs):
        if sys.platform == "darwin":
            return None
        return func(*args, **kwargs)

    return wrapper


class ImageTestBase(unittest.TestCase):
    def setUp(self):
        self.count = 0

    def create_image(self):
        plt.savefig(".temp.png")
        with Image.open(".temp.png") as i:
            result = np.array(i)
        os.remove(".temp.png")
        return result

    def check_image(self, min_delta=0.015):
        path = f"tests/images/{self.id()}_{self.count}.png"
        self.count += 1
        img_as_array = self.create_image()
        if is_testing:
            saved_img_as_array = np.array(Image.open(path))
            # tolerate mismatched elements up to 1.5% of the total
            if (img_as_array != saved_img_as_array).mean() > min_delta:
                # this will fail, it's just for the error message
                np.testing.assert_array_equal(img_as_array, saved_img_as_array)
        else:
            Image.fromarray(img_as_array).save(path)
        plt.close()

    def test_is_testing(self):
        self.assertTrue(is_testing)
