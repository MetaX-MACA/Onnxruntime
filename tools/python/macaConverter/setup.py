import os
from setuptools import find_packages, setup
import platform
from maca_converter.common import MXC_CONFIG

def readme(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    return content

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
README_PATH = os.path.join(SCRIPT_DIR, "README.md")
if platform.machine()=='aarch64':
    REQUIREMENTS_PATH = os.path.join(SCRIPT_DIR, "requirements-aarch64.txt")
else:
    REQUIREMENTS_PATH = os.path.join(SCRIPT_DIR, "requirements-x86.txt")
print(REQUIREMENTS_PATH)

setup(author='metax',
      name='maca_converter',
      version=MXC_CONFIG.VERSION,
      description='macaConverter is an offline converter  tools',
      install_requires=open(REQUIREMENTS_PATH).readlines(),
      long_description=readme(README_PATH),
      python_requires='>=3.6',
      packages=find_packages(),
    )