import matplotlib.pyplot as plt
import numpy as np

# Example data
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([5, 3, 8, 2, 7])

x = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

############## geometrical adaptation #####################################################
y1 = np.array([0.775,0.805,0.842,0.841,0.815,0.753,0.702,0.657,0.566,0.433,0.362]) #sketch
y2 = np.array([0.829,0.835,0.842,0.805,0.727,0.639,0.532,0.426,0.370,0.291,0.212]) #art
y3 = np.array([0.781,0.793,0.8053,0.803,0.753,0.721,0.672,0.518,0.457,0.383,0.353]) #cartoon
y4 = np.array([0.933,0.952,0.947,0.948,0.911,0.8655,0.7023,0.611,0.444,0.301,0.206]) #photo
############## Statistical adaptation #####################################################
# y1 = np.array([0.775,0.805,0.842,0.841,0.815,0.753,0.702,0.657,0.566,0.433,0.362]) #sketch
# y2 = np.array([0.829,0.835,0.842,0.805,0.727,0.639,0.532,0.426,0.370,0.291,0.212]) #art
# y3 = np.array([0.781,0.793,0.8053,0.803,0.753,0.721,0.672,0.518,0.457,0.383,0.353]) #cartoon
# y4 = np.array([0.933,0.952,0.947,0.948,0.911,0.8655,0.7023,0.611,0.444,0.301,0.206]) #photo

# Plotting with 'o' markers and lines connecting the points
plt.plot(x, y1, marker='>', linestyle='-', label='Sketch')
plt.plot(x, y2, marker='>', linestyle='-', label='Art')
plt.plot(x, y3, marker='o', linestyle='-', label='Cartoon')
plt.plot(x, y4, marker='p', linestyle='-', label='Photo')

# Adding labels
plt.xlabel('Weight of Geometric loss in total loss ')
plt.ylabel('Average Accuracy')
plt.legend()
# Display the plot
plt.show()
