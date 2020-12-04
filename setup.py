#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
        "numpy",
        "pandas",
        "tensorflow",
        "panel",
        "param",
        "pillow",
        "matplotlib",
        "mlflow>=1.12",
        "tqdm",
        "sklearn",
        "scipy"
        ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Joe Gezo",
    author_email='joegezo@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Active learning for quickly building image patch classifiers",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='patchwork',
    name='patchwork',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jg10545/patchwork',
    version='0.1.0',
    zip_safe=False,
)
