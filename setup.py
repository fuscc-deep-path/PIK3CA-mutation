#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Jingcheng Yang",
    author_email='yjcyxky@163.com',
    maintainer="Choppy Team",
    maintainer_email="choppyteam@gmail.com",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="AI models for pathology images.",
    entry_points={
        'console_scripts': [
            'PIK3CA_mutation=PIK3CA_mutation.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='PIK3CA_mutation',
    name='PIK3CA-mutation',
    packages=find_packages(include=['PIK3CA_mutation']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/deep-omics/PIK3CA_mutation',
    version='0.1.0',
    zip_safe=False,
)
