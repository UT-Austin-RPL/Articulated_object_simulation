# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD!

from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml.
setup_args = generate_distutils_setup(
    packages=["s2u", "s2u.utils",], package_dir={"": "src"},
)

setup(**setup_args)
