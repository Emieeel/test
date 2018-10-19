# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:10:59 2018

@author: emiel
"""

import pickle
import numpy as np
import pandas as pd

with open("predictionsemielscaledlin.txt", "rb") as fp:
    predictions = pickle.load(fp)
#for n in range(len(predictions)):
#    predictions[n] = int(round(predictions[n]))

test = pd.read_csv('test.csv')
print(test.columns)
submission = test.drop(columns =['year', 'month', 'day', 'sched_dep_time', 'sched_arr_time', 'carrier', 'origin', 'dest', 'distance'])

#submission.drop(columns =['year', 'month', 'day', 'sched_dep_time', 'sched_arr_time', 'carrier', 'origin', 'dest', 'distance'])
submission['is_delayed'] = pd.Series(predictions, index=test.index)


submission.to_csv('submissiontestemielscaledlin.csv')

#column_order = ['is_delayed', 'id', 'month', 'day', \
#                'sched_dep_time', 'sched_arr_time', 'carrier', 'origin', 'dest', 'distance']
#test[column_order].to_csv('submission.csv')


print(test)
print(test.dtypes)