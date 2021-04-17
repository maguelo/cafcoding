'''
@author: Cafcoding
'''
# !/usr/bin/env python

from setuptools import setup

setup(name='cafcoding',
      version='1.1.2',
      description='Cafcoding libraries',
      author='Cafcoding',
      author_email='',
      url='',
      packages=['cafcoding','cafcoding.tools', 'cafcoding.stages'],
      install_requires=['geopy',
                        'colorlog',
                        'srtm.py',
                        's3fs',
                        'numpy',
                        'pandas',
                        'seaborn'],
      dependency_links=['http://github.com/tkrajina/srtm.py/tarball/master#egg=srtm.py-0.3.7']
      )
