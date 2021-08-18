# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:08:12 2021

@author: TinoGehlert
"""

import os
import pandas as pd


def load_examples():
    dctExamples = {}
    dctExamples["ElectricLoad"] = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                           'examples/entsoe.csv'))

    return dctExamples
