import numpy as np
from matplotlib import pyplot as plt

from csv_parse import CSVParser
from load_data import convert_raw_to_csv

import pandas as pd

from preprocessor import preprocess_time_to_freq

csvp = CSVParser("data-parsed/Data_20251031_105214_Samuel.csv")
y = csvp.get_channel(12, "EEG 7")
dt = 1/1000
fourier, freqs, df = preprocess_time_to_freq(y, dt)
plt.plot(freqs, fourier, 'b')
plt.show()