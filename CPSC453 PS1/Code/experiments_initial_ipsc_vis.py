import ps1_functions
import numpy as np
import numpy.linalg as la
import codecs, json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.lines import Lines2D


#iPSC EXPERIMENTS

print("iPSC EXPERIMENTS\n\n\n")

ipsc_points = ps1_functions.load_json_files("/home/malakkhan/CPSC453/CPSC453 PS1/Data/ipsc_data.json")
channel_names = ps1_functions.load_json_files("/home/malakkhan/CPSC453/CPSC453 PS1/Data/ipsc_channel_names.json")


#VISUALIZING iPSC DATA

print("data:\n")
print(ipsc_points)

#mean centering data
mean_arr = np.mean(ipsc_points, axis = 0)

n, m = np.shape(ipsc_points)
for observation in range(n):
	ipsc_points[observation] = ipsc_points[observation] - mean_arr

print("mean centered data:\n")
print(ipsc_points)

print("channel names:\n")
print(channel_names)


# fig = plt.figure()
# ax1 = Axes3D(fig)

# # # Data for three-dimensional scattered points
# xdata = ipsc_points[:,0]
# ydata = ipsc_points[:,1]
# zdata = ipsc_points[:,2]

# plt.title("iPSC Initial Points/Labels")
# ax1.scatter(xdata, ydata, zdata)
# plt.show()






#VISUALIZING iPSC DATA WITH PCA
u, s, v_t = la.svd(ipsc_points)

#VISUALIZING THE SINGULAR VALUES TO ASSESS INTRINSIC DIMENSIONALITY
xdata = np.asarray(range(m)) + 1
plt.title('iPSC Singular Values')
plt.plot(xdata, s)
# plt.show()
plt.savefig('iPSC Singular Values.png')

# #VISUALIZING EXPLAINED VARIANCE
ydata = np.true_divide(np.power(s, 2), n - 1)
plt.title('iPSC Explained Variance')
plt.plot(xdata, ydata)
# plt.show()
plt.savefig('iPSC Explained Variance.png')

#PC1 and PC2
diag_mat = np.zeros((n, 2))
diag_mat[0][0] = s[0]
diag_mat[1][1] = s[1]

proj_1_2 = np.dot(u, diag_mat)

fig = plt.figure()

xdata = proj_1_2[:,0]
ydata = proj_1_2[:,1]
plt.title("iPSC PC1 and PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.scatter(xdata, ydata)
# plt.show()
plt.savefig('iPSC PC1 and PC2.png')

#PC2 and PC3
diag_mat = np.zeros((n, 2))
diag_mat[1][0] = s[1]
diag_mat[2][1] = s[2]

proj_1_2 = np.dot(u, diag_mat)

fig = plt.figure()

xdata = proj_1_2[:,0]
ydata = proj_1_2[:,1]
plt.title("iPSC PC2 and PC3")
plt.xlabel("PC2")
plt.ylabel("PC3")

plt.scatter(xdata, ydata)
# plt.show()
plt.savefig('iPSC PC2 and PC3.png')


#PC1 and PC3
diag_mat = np.zeros((n, 2))
diag_mat[0][0] = s[0]
diag_mat[2][1] = s[2]

proj_1_2 = np.dot(u, diag_mat)

fig = plt.figure()

xdata = proj_1_2[:,0]
ydata = proj_1_2[:,1]
plt.title("iPSC PC1 and PC3")
plt.xlabel("PC1")
plt.ylabel("PC3")

plt.scatter(xdata, ydata)
# plt.show()
plt.savefig('iPSC PC1 and PC3.png')

#PC4 and PC5
diag_mat = np.zeros((n, 2))
diag_mat[0][0] = s[3]
diag_mat[2][1] = s[4]

proj_1_2 = np.dot(u, diag_mat)

fig = plt.figure()

xdata = proj_1_2[:,0]
ydata = proj_1_2[:,1]
plt.title("iPSC PC4 and PC5")
plt.xlabel("PC4")
plt.ylabel("PC5")

plt.scatter(xdata, ydata)
# plt.show()
plt.savefig('iPSC PC4 and PC5.png')

#PC1 and PC5
diag_mat = np.zeros((n, 2))
diag_mat[0][0] = s[0]
diag_mat[2][1] = s[4]

proj_1_2 = np.dot(u, diag_mat)

fig = plt.figure()

xdata = proj_1_2[:,0]
ydata = proj_1_2[:,1]
plt.title("iPSC PC1 and PC5")
plt.xlabel("PC1")
plt.ylabel("PC5")

plt.scatter(xdata, ydata)
# plt.show()
plt.savefig('iPSC PC1 and PC5.png')