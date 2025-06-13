from setuptools import find_packages, setup


def get_install_requirements():
    return []


def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    setup(
        name='grid_opt',
        maintainer="Yulun Tian",
        maintainer_email="yuluntian.research@gmail.com",
        version='0.1',
        license='BSD-2-Clause',
        description='Feature optimization over 2D/3D grids.',
        long_description=long_description,
        packages=find_packages(include=['grid_opt']),
        install_requires=get_install_requirements(),
    )


if __name__ == "__main__":
    setup_package()