# -*- coding: utf-8 -*-

"""Top-level package for patchwork."""

__author__ = """Joe Gezo"""
__email__ = 'joegezo@gmail.com'
__version__ = '0.1.0'


from patchwork._main import PatchWork
from patchwork._experiment import run_experiment, show_experiments
from patchwork._prep import prep_label_dataframe
import patchwork.feature
import patchwork.viz