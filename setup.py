'''
@author: Cafcoding
'''
# !/usr/bin/env python

from setuptools import setup

setup(name='cafcoding',
      version='1.1.1',
      description='Cafcoding libraries',
      author='Cafcoding',
      author_email='',
      url='',
      packages=['cafcoding','cafcoding.tools', 'cafcoding.stages'],
      install_requires=['geopy',
                        'colorlog',
                        'git+https://github.com/tkrajina/srtm.py.git',
                        's3fs',
                        'numpy',
                        'pandas'],
      )
