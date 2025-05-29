import os
from setuptools import find_packages, setup
from maca_precision.common import MXP_CONFIG

def readme(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    return content

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
README_PATH = os.path.join(SCRIPT_DIR, "README.md")
REQUIREMENTS_PATH = os.path.join(SCRIPT_DIR, "requirements.txt")


setup(author='metax',
      name='maca_precision',
      version=MXP_CONFIG.VERSION,
      description='macaPrecision is an offline precision debug tools',
      install_requires=open(REQUIREMENTS_PATH).readlines(),
      long_description=readme(README_PATH),
      python_requires='>=3.6',
      packages=find_packages(),
    )