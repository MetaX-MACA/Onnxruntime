import os
from setuptools import find_packages, setup
from maca_quantizer.version import MXQ_CONFIG

def readme(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    return content

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
README_PATH = os.path.join(SCRIPT_DIR, "README.md")
REQUIREMENTS_PATH = os.path.join(SCRIPT_DIR, "requirements.txt")


setup(author='maca_quantizer',
      name='maca_quantizer',
      version=MXQ_CONFIG.VERSION,
      description='macaQuantizer is an offline quantization tools',
      install_requires=open(REQUIREMENTS_PATH).readlines(),
      long_description=readme(README_PATH),
      python_requires='>=3.6',
      packages=find_packages(),
      include_package_data=True

    )