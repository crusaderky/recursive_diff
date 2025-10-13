"""Test import the library and print essential information"""

import platform
import sys

import recursive_diff

print("Python interpreter:", sys.executable)
print("Python version    :", sys.version)
print("Platform          :", platform.platform())
print("Library path      :", recursive_diff.__file__)
print("Library version   :", recursive_diff.__version__)
