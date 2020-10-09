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

#VISUALIZING SWISS ROLL DATA WITH DIFFUSION MAP ADAPTIVE GAUSSIAN k = 5, t = 1
D = ps1_functions.compute_distances(swiss_roll_points)
W = ps1_functions.compute_affinity_matrix(D, "adaptive", k=5)
diff_vec, diff_eig = ps1_functions.diff_map_info(W)
diff_map = ps1_functions.get_diff_map(diff_vec, diff_eig, 1)

print(diff_map)

xdata = np.asarray(range(np.size(diff_eig))) + 2
plt.title('Markov Matrix Eigenvalues-  k = 5, t = 1')
plt.plot(xdata, np.flip(diff_eig))
# plt.show()
plt.savefig('Markov Matrix Eigenvalues-  k = 5, t = 1.png')


n, m = np.shape(diff_map)

print(n, m)

#DC2 and DC3
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -2]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("Swiss Roll DC2 and DC3- k = 5, t = 1")
plt.xlabel("DC2")
plt.ylabel("DC3")
plt.scatter(plane[:, 0], plane[:,1], c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll DC2 and DC3- k = 5, t = 1.png')


#DC3 and DC4
DM1 = diff_map[:, -2]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("Swiss Roll DC3 and DC4- k = 5, t = 1")
plt.xlabel("DC3")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1], c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll DC3 and DC4- k = 5, t = 1.png')


#DM2 and DM4
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("Swiss Roll DC2 and DC4- k = 5, t = 1")
plt.xlabel("DC2")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1], c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll DC2 and DC4- k = 5, t = 1.png')




#VISUALIZING SWISS ROLL DATA WITH DIFFUSION MAPS ADAPTIVE GAUSSIAN k = 10, t = 1
D = ps1_functions.compute_distances(swiss_roll_points)
W = ps1_functions.compute_affinity_matrix(D, "adaptive", k=10)
diff_vec, diff_eig = ps1_functions.diff_map_info(W)
diff_map = ps1_functions.get_diff_map(diff_vec, diff_eig, 1)

xdata = np.asarray(range(np.size(diff_eig))) + 2
plt.title('Markov Matrix Eigenvalues-  k = 10, t = 1')
plt.plot(xdata, np.flip(diff_eig))
# plt.show()
plt.savefig('Markov Matrix Eigenvalues-  k = 10, t = 1.png')


print(diff_map)


n, m = np.shape(diff_map)

print(n, m)

#DC2 and DC3
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -2]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("Swiss Roll DC2 and DC3- k = 10, t = 1")
plt.xlabel("DC2")
plt.ylabel("DC3")
plt.scatter(plane[:, 0], plane[:,1], c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll DC2 and DC3- k = 10, t = 1.png')


#DC3 and DC4
DM1 = diff_map[:, -2]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("Swiss Roll DC3 and DC4- k = 10, t = 1")
plt.xlabel("DC3")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1], c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll DC3 and DC4- k = 10, t = 1.png')


#DM2 and DM4
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("Swiss Roll DC2 and DC4- k = 10, t = 1")
plt.xlabel("DC2")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1], c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll DC2 and DC4- k = 10, t = 1.png')

