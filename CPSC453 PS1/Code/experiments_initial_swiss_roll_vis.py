import ps1_functions
import numpy as np
import numpy.linalg as la
import codecs, json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.lines import Lines2D


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


fig = plt.figure()
ax1 = Axes3D(fig)

# # Data for three-dimensional scattered points
xdata = swiss_roll_points[:,0]
ydata = swiss_roll_points[:,1]
zdata = swiss_roll_points[:,2]

plt.title("Swiss Roll Initial Points/Labels")
ax1.scatter(xdata, ydata, zdata, c=swiss_roll_labels)
plt.show()






#VISUALIZING SWISS ROLL DATA WITH PCA
u, s, v_t = la.svd(swiss_roll_points)

#VISUALIZING THE SINGULAR VALUES TO ASSESS INTRINSIC DIMENSIONALITY
xdata = np.asarray(range(m)) + 1
plt.title('Swiss Roll Singular Values')
plt.plot(xdata, s)
# plt.show()
plt.savefig('Swiss Roll Singular Values.png')

# #VISUALIZING EXPLAINED VARIANCE
ydata = np.true_divide(np.power(s, 2), n - 1)
plt.title('Swiss Roll Explained Variance')
plt.plot(xdata, ydata)
# plt.show()
plt.savefig('Swiss Roll Explained Variance.png')

#PC1 and PC2
diag_mat = np.zeros((n, 2))
diag_mat[0][0] = s[0]
diag_mat[1][1] = s[1]

proj_1_2 = np.dot(u, diag_mat)

fig = plt.figure()

xdata = proj_1_2[:,0]
ydata = proj_1_2[:,1]
plt.title("Swiss Roll PC1 and PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.scatter(xdata, ydata, c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll PC1 and PC2.png')

#PC2 and PC3
diag_mat = np.zeros((n, 2))
diag_mat[1][0] = s[1]
diag_mat[2][1] = s[2]

proj_1_2 = np.dot(u, diag_mat)

fig = plt.figure()

xdata = proj_1_2[:,0]
ydata = proj_1_2[:,1]
plt.title("Swiss Roll PC2 and PC3")
plt.xlabel("PC2")
plt.ylabel("PC3")

plt.scatter(xdata, ydata, c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll PC2 and PC3.png')


#PC1 and PC3
diag_mat = np.zeros((n, 2))
diag_mat[0][0] = s[0]
diag_mat[2][1] = s[2]

proj_1_2 = np.dot(u, diag_mat)

fig = plt.figure()

xdata = proj_1_2[:,0]
ydata = proj_1_2[:,1]
plt.title("Swiss Roll PC1 and PC3")
plt.xlabel("PC1")
plt.ylabel("PC3")

plt.scatter(xdata, ydata, c=swiss_roll_labels)
# plt.show()
plt.savefig('Swiss Roll PC1 and PC3.png')




