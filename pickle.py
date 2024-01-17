# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:12:22 2023

@author: Yasmin
"""

import pickle 
import matplotlib.pyplot as plt 
import numpy as np

file = open('C:/Users/Yamin/Downloads/pickle/loses_pickle_lstm1window','rb')


data = pickle.load(file)

#0 losses
#1 losses test 

print((data[1]))
#descomentar una u otra dependiendo si es de train o test la 
#loss que queremos visualizar 

plt.plot(data[0], label='Loss train')
# # 
plt.plot(data[1], label='Loss test')

plt.xlabel('èpoques')
plt.ylabel('loss')
plt.title('Gràfica Loss train i test Model 1 LSTM- Window')


plt.legend()
plt.show()
