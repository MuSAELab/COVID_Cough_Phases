# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:29:00 2021

@author: richa
"""
import sys, os, glob
import textgrid
import re

def get_annotation(abs_path=r'C:\Users\richa\OneDrive\desktop\Praat_results\cough_annotation',
                   split='train'):

    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    path_ant = abs_path
    tg=[]
    for filename in sorted(glob.glob(os.path.join(path_ant,'%s_*.TextGrid')%(split)),key=numericalSort):
        print(filename)
        tg.append(textgrid.TextGrid.fromFile(filename))
        
    return tg


