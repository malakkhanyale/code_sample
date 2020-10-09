import ps1_functions
import numpy as np
import numpy.linalg as la
import codecs, json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#SWISS ROLL EXPERIMENTS

print("SWISS ROLL EXPERIMENTS\n\n\n")

swiss_roll_points = ps1_functions.load_json_files("/home/malakkhan/CPSC453/CPSC453 PS1/Data/swiss_roll_points.json")
swiss_roll_labels = ps1_functions.load_json_files("/home/malakkhan/CPSC453/CPSC453 PS1/Data/swiss_roll_labels.json")


#VISUALIZING SWISS ROLL DATA

print("data:\n")
print(swiss_roll_points)

#mean centering data
mean_arr = np.mean(swiss_roll_points, axis = 0)

n, m = np.shape(swiss_roll_points)
for observation in range(n):
	swiss_roll_points[observation] = swiss_roll_points[observation] - mean_arr

print("mean centered data:\n")
print(swiss_roll_points)

print("labels:\n")
print(swiss_roll_labels)


#COMPUTING DIFFUSION MAP
D = ps1_functions.compute_distances(swiss_roll_points)
W = ps1_functions.compute_affinity_matrix(D, "gaussian", sigma=6.0)
diff_vec, diff_eig = ps1_functions.diff_map_info(W)
diff_map = ps1_functions.get_diff_map(diff_vec, diff_eig, 1)


fig = plt.figure()
ax1 = Axes3D(fig)

# # Data for three-dimensional scattered points
xdata = swiss_roll_points[:,0]
ydata = swiss_roll_points[:,1]
zdata = swiss_roll_points[:,2]

plt.title("Swiss Roll Labelled Per Values of Largest Left Eigenvector")
ax1.scatter(xdata, ydata, zdata, c=diff_map[:,-1])
plt.show()