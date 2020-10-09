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


#VISUALIZING iPSC DATA WITH DIFFUSION MAP ADAPTIVE GAUSSIAN k = 2, t = 10
D = ps1_functions.compute_distances(ipsc_points)
W = ps1_functions.compute_affinity_matrix(D, "adaptive", k=2)
diff_vec, diff_eig = ps1_functions.diff_map_info(W)
diff_map = ps1_functions.get_diff_map(diff_vec, diff_eig, 10)

print(diff_map)

xdata = np.asarray(range(np.size(diff_eig))) + 2
plt.title('iPSC Markov Matrix Eigenvalues-  k = 2, t = 10')
plt.plot(xdata, np.flip(diff_eig))
# plt.show()
plt.savefig('iPSC Markov Matrix Eigenvalues-  k = 2, t = 10.png')


n, m = np.shape(diff_map)

print(n, m)


#CORRELATIONS
n, m = np.shape(ipsc_points)

first_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -1])
	first_comp[channel_names[channel]] = coeff

second_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -2])
	second_comp[channel_names[channel]] = coeff

third_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -3])
	third_comp[channel_names[channel]] = coeff

print(first_comp)
print(second_comp)
print(third_comp)


#DC2 and DC3
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -2]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC2 and DC3- k = 2, t = 10")
plt.xlabel("DC2")
plt.ylabel("DC3")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC2 and DC3- k = 2, t = 10.png')


#DC3 and DC4
DM1 = diff_map[:, -2]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC3 and DC4- k = 2, t = 10")
plt.xlabel("DC3")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC3 and DC4- k = 2, t = 10.png')


#DM2 and DM4
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC2 and DC4- k = 2, t = 10")
plt.xlabel("DC2")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC2 and DC4- k = 2, t = 10.png')







#VISUALIZING iPSC DATA WITH DIFFUSION MAP ADAPTIVE GAUSSIAN k = 2, t = 20
D = ps1_functions.compute_distances(ipsc_points)
W = ps1_functions.compute_affinity_matrix(D, "adaptive", k=2)
diff_vec, diff_eig = ps1_functions.diff_map_info(W)
diff_map = ps1_functions.get_diff_map(diff_vec, diff_eig, 20)

print(diff_map)

xdata = np.asarray(range(np.size(diff_eig))) + 2
plt.title('iPSC Markov Matrix Eigenvalues-  k = 2, t = 20')
plt.plot(xdata, np.flip(diff_eig))
# plt.show()
plt.savefig('iPSC Markov Matrix Eigenvalues-  k = 2, t = 20.png')


n, m = np.shape(diff_map)

print(n, m)


#CORRELATIONS
n, m = np.shape(ipsc_points)

first_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -1])
	first_comp[channel_names[channel]] = coeff

second_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -2])
	second_comp[channel_names[channel]] = coeff

third_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -3])
	third_comp[channel_names[channel]] = coeff

print(first_comp)
print(second_comp)
print(third_comp)


#DC2 and DC3
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -2]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC2 and DC3- k = 2, t = 20")
plt.xlabel("DC2")
plt.ylabel("DC3")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC2 and DC3- k = 2, t = 20.png')


#DC3 and DC4
DM1 = diff_map[:, -2]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC3 and DC4- k = 2, t = 20")
plt.xlabel("DC3")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC3 and DC4- k = 2, t = 20.png')


#DM2 and DM4
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC2 and DC4- k = 2, t = 20")
plt.xlabel("DC2")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC2 and DC4- k = 2, t = 20.png')













#VISUALIZING iPSC DATA WITH DIFFUSION MAP ADAPTIVE GAUSSIAN k = 5, t = 1
D = ps1_functions.compute_distances(ipsc_points)
W = ps1_functions.compute_affinity_matrix(D, "adaptive", k=5)
diff_vec, diff_eig = ps1_functions.diff_map_info(W)
diff_map = ps1_functions.get_diff_map(diff_vec, diff_eig, 1)

print(diff_map)

xdata = np.asarray(range(np.size(diff_eig))) + 2
plt.title('iPSC Markov Matrix Eigenvalues-  k = 5, t = 1')
plt.plot(xdata, np.flip(diff_eig))
# plt.show()
plt.savefig('iPSC Markov Matrix Eigenvalues-  k = 5, t = 1.png')


n, m = np.shape(diff_map)

print(n, m)


#CORRELATIONS
n, m = np.shape(ipsc_points)

first_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -1])
	first_comp[channel_names[channel]] = coeff

second_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -2])
	second_comp[channel_names[channel]] = coeff

third_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -3])
	third_comp[channel_names[channel]] = coeff

print(first_comp)
print(second_comp)
print(third_comp)


#DC2 and DC3
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -2]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC2 and DC3- k = 5, t = 1")
plt.xlabel("DC2")
plt.ylabel("DC3")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC2 and DC3- k = 5, t = 1.png')


#DC3 and DC4
DM1 = diff_map[:, -2]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC3 and DC4- k = 5, t = 1")
plt.xlabel("DC3")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC3 and DC4- k = 5, t = 1.png')


#DM2 and DM4
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC2 and DC4- k = 5, t = 1")
plt.xlabel("DC2")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC2 and DC4- k = 5, t = 1.png')








#VISUALIZING iPSC DATA WITH DIFFUSION MAP ADAPTIVE GAUSSIAN k = 10, t = 1
D = ps1_functions.compute_distances(ipsc_points)
W = ps1_functions.compute_affinity_matrix(D, "adaptive", k=10)
diff_vec, diff_eig = ps1_functions.diff_map_info(W)
diff_map = ps1_functions.get_diff_map(diff_vec, diff_eig, 1)

print(diff_map)

xdata = np.asarray(range(np.size(diff_eig))) + 2
plt.title('iPSC Markov Matrix Eigenvalues-  k = 10, t = 1')
plt.plot(xdata, np.flip(diff_eig))
# plt.show()
plt.savefig('iPSC Markov Matrix Eigenvalues-  k = 10, t = 1.png')


n, m = np.shape(diff_map)

print(n, m)


#CORRELATIONS
n, m = np.shape(ipsc_points)

first_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -1])
	first_comp[channel_names[channel]] = coeff

second_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -2])
	second_comp[channel_names[channel]] = coeff

third_comp = {}
for channel in range(m):
	coeff = np.corrcoef(ipsc_points[:, channel], diff_map[:, -3])
	third_comp[channel_names[channel]] = coeff

print(first_comp)
print(second_comp)
print(third_comp)


#DC2 and DC3
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -2]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC2 and DC3- k = 10, t = 1")
plt.xlabel("DC2")
plt.ylabel("DC3")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC2 and DC3- k = 10, t = 1.png')


#DC3 and DC4
DM1 = diff_map[:, -2]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC3 and DC4- k = 10, t = 1")
plt.xlabel("DC3")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC3 and DC4- k = 10, t = 1.png')


#DM2 and DM4
DM1 = diff_map[:, -1]
DM2 = diff_map[:, -3]

plane = np.asarray([DM1, DM2]).T
n, m = np.shape(plane)

fig = plt.figure()
plt.title("iPSC DC2 and DC4- k = 10, t = 1")
plt.xlabel("DC2")
plt.ylabel("DC4")
plt.scatter(plane[:, 0], plane[:,1])
# plt.show()
plt.savefig('iPSC DC2 and DC4- k = 10, t = 1.png')