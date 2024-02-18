import matplotlib.pyplot as plt
import numpy as np

# Example data
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([5, 3, 8, 2, 7])

x = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

x_weight = np.array([0,0.1,0.15,0.20,0.25,0.3,0.4,0.5,0.6])
y_total = np.array([82.95,83.41,84.18,85.05,85.35,85.18,83.22,82.76,80.04])

# ############## geometrical adaptation #####################################################
# y1 = np.array([0.775,0.805,0.842,0.841,0.815,0.753,0.702,0.657,0.566,0.433,0.362]) #sketch
# y2 = np.array([0.829,0.835,0.842,0.805,0.727,0.639,0.532,0.426,0.370,0.291,0.212]) #art
# y3 = np.array([0.781,0.793,0.8053,0.803,0.753,0.721,0.672,0.518,0.457,0.383,0.353]) #cartoon
# y4 = np.array([0.933,0.952,0.947,0.948,0.911,0.8655,0.7023,0.611,0.444,0.301,0.206]) #photo
# ############## Statistical adaptation #####################################################
# # y1 = np.array([0.775,0.805,0.842,0.841,0.815,0.753,0.702,0.657,0.566,0.433,0.362]) #sketch
# # y2 = np.array([0.829,0.835,0.842,0.805,0.727,0.639,0.532,0.426,0.370,0.291,0.212]) #art
# # y3 = np.array([0.781,0.793,0.8053,0.803,0.753,0.721,0.672,0.518,0.457,0.383,0.353]) #cartoon
# # y4 = np.array([0.933,0.952,0.947,0.948,0.911,0.8655,0.7023,0.611,0.444,0.301,0.206]) #photo

# Plotting with 'o' markers and lines connecting the points
# Add a horizontal line
# plt.axhline(y=83.57, color='pink', linestyle='-', linewidth=40)

y = [83.57 for i in x_weight]

plt.axhline(y=83.57, color='pink', linestyle='-', linewidth=30,alpha = 0.9)
# print (y)
# Plot the data

custom_marker = [(0.5, 0), (-0.5, 0)]
# vars = [[82.75,83.20],[83.01,83.82],[83.95,84.41],[84.86,85.24],[85.12,85.58],[84.90,85.46],[82.90,83.54],[82.46,83.06],[79.70,80.38]]
vars = [0.51,0.61,0.36,0.25,0.33,0.52,0.39,0.29,0.42]
for i in range(len(x_weight)):
    plt.plot([x_weight[i],x_weight[i]], [ y_total[i]-vars[i], y_total[i] + vars[i]], color='navy', linewidth=2)
    # Mark the start and end points with circles
    # plt.scatter([x_weight[i], x_weight[i]], [vars[i][0], vars[i][1]], color='navy', marker=custom_marker)
    plt.scatter([x_weight[i], x_weight[i]], [y_total[i]-vars[i], y_total[i] + vars[i]], color='navy', marker=custom_marker)




# Set axis limits to include all points
plt.xlim(min(x_weight) -0.1 , max(x_weight) + 0.1)
plt.ylim(min(y_total) - 1, max(y_total) + 1)

plt.plot(x_weight, y_total, marker='o', linestyle='-', label='total',color = 'navy')
plt.plot(x_weight, y, label='Baseline',linestyle = '--',color='red')
plt.xticks(x_weight)  # Set x-axis ticks to 0, 2, 4, 6, 8, 10
plt.yticks(y_total)
plt.xticks(rotation=45)
plt.tight_layout()

# plt.yticks(rotation=45)
# Enable autoscaling
# plt.gca().autoscale(enable=True, axis='both', tight=True)
# plt.plot(x, y1, marker='>', linestyle='-', label='Sketch')
# plt.plot(x, y2, marker='>', linestyle='-', label='Art')
# plt.plot(x, y3, marker='o', linestyle='-', label='Cartoon')
# plt.plot(x, y4, marker='p', linestyle='-', label='Photo')

# Adding labels
plt.xlabel('Weight of Geometric loss in total loss ')
plt.ylabel('Average Accuracy')
plt.legend()
# Display the plot
plt.show()
