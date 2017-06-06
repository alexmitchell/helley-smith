#!/usr/bin/env python

import numpy as np

class Constants:

    verbose = True

    # Data location
    reload_data = False
    data_path="./data/"
    pickle_names = {'hs':'helly_smith', 'lt':'light_table'}

    hs_max_size = 32 #mm
    lt_size_classes = 0.5, 0.71, 1, 1.4, 2, 2.8, 4, 5.6, 8, 11.2, 16, 22, 32, 45
    window_duration = 60 * 5 # 5min windows in seconds
    window_tolerance = 0.15 # max percent of window that can be skipped when doing rolling calculations
    experiment_start_time = 2870
    hs_sample_times = (np.array([2890,2950,3010,3070]) - 2870) * 60 # Feeling too lazy to parse this from the HS data labels

    gravity = 9.81 #m/s^2
    density_w = 1000 #kg/m^3
    density_s = 2650 #kg/m^3
    slope = 0.015 #m/m
    width = 1 #m
    crit_shields_num = 0.045


