from setuptools import setup
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('geo_gpu',parent_package=None,top_path=None)
config.packages = ["geo_gpu"]
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))