# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:22:24 2022

@author: richa
"""

"""
Extract openSMILE features from an audio/audio segment.
"""

import opensmile
import os, glob
import numpy as np
import pandas as pd
import re

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def get_SMILE(audio_file:str):
    
    smile_fea = smile.process_file(audio_file)
    
    return smile_fea