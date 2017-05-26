#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import geoutilities.ut_data_loading as utdl
#import ut_data_loading as utdl

file = "DesignWidthRev1.txt"
#file = "Station_Width_Data.txt"
datapath = "./data/widths/"
filepath = datapath + file

design = np.loadtxt(filepath)/1000 # Convert to meters
#design = design.transpose()
stations = design[0]
widths = design[1]

max_width = np.max(widths)
min_width = np.min(widths)
max_station = np.max(stations)
min_station = np.min(stations)

# Calculate average width
areas = (widths[1:] + widths[:-1]) / 2 * (stations[1:] - stations[:-1])
av_width = areas.sum() / (max_station - min_station)
print("Average width is {}m".format(av_width))


# Plot channel boundaries
for sign in 1, -1:
    # need to divide widths by 2 so that they will be centered
    edge = sign * widths / 2
    plt.plot(stations, edge)
    plt.plot([min_station, max_station], [sign * av_width / 2]*2)

plt.ylim((-0.5,0.5))
plt.show()
