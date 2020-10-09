# ps1_functions.py
# Skeleton file by Chris Harshaw, Yale University, Fall 2017
# Adapted by Jay Stanley, Yale University, Fall 2018
# Adapted by Scott Gigante, Yale University, Fall 2019
# CPSC 553 -- Problem Set 1
#
# This script contains uncompleted functions for implementing diffusion maps.
#
# NOTE: please keep the variable names that I have put here, as it makes grading easier.

# import required libraries
import numpy as np
import numpy.linalg as la
import codecs, json

##############################
# Predefined functions
##############################

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        json_data    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


##############################
# Skeleton code (fill these in)
##############################


def compute_distances(X):
    '''
    Constructs a distance matrix from data set, assumes Euclidean distance

    Inputs:
        X       a numpy array of size n x p holding the data set (n observations, p features)

    Outputs:
        D       a numpy array of size n x n containing the euclidean distances between points

    '''
    n, p = X.shape

    D = np.zeros((n, n))

    for i in range (n):
    	for j in range (i + 1, n):
    		D[i, j] = la.norm(X[i] - X[j])
    		D[j, i] = D[i, j]


    return D


def compute_affinity_matrix(D, kernel_type, sigma=None, k=None):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    '''

    n, n2 = D.shape

    W = np.zeros((n, n))


 	#calculating kaffinity matrix using non-adaptive gaussian kernel
    if kernel_type == "gaussian":

    	if sigma <= 0:
    		
    		sys.exit("Sigma must be a positive number.")

    	for i in range (n):
    		for j in range (i + 1, n):
    			W[i, j] = -1 * ((D[i, j] ** 2) / (sigma ** 2))
    			W[j, i] = W[i, j]

    	W = np.exp(W)
    

    #calculating affinity matrix using adaptive kernel
    else:

    	if type(k) != int or (type(k) == float and k.is_integer() == False) or k <= 0:
    		
    		sys.exit("k must be a positive integer.")

    	for i in range (n):
    		for j in range (i + 1, n):

    			iarray = np.sort(D[i])
    			sigmaki = iarray[k - 1]

    			jarray = np.sort(D[j])
    			sigmakj = jarray[k - 1]


    			W[i, j] = np.exp(-1 * ((D[i, j] ** 2) / (sigmaki ** 2))) + np.exp(-1 * ((D[i, j] ** 2) / (sigmakj ** 2)))
    			W[i, j] = W[i, j] * 0.5

    			W[j, i] = W[i, j]

    return W


def diff_map_info(W):
    '''
    Construct the information necessary to easily construct diffusion map for any t

    Inputs:
        W           a numpy array of size n x n containing the affinities between points

    Outputs:

        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix

        We assume the convention that the coordinates in the diffusion vectors are in descending order
        according to eigenvalues.
    '''

    n, n2 = W.shape


    #constructing diagonal matrix of row sums
    D = np.zeros((n, n))

    for i in range(n):
    	D[i, i] = np.sum(W[i])
        D[i, i] = D[i, i] ** -0.5


    #powering matrix by -1/2
    Dpow = D


    #computing symmetric matrix MS
    WDpow = np.matmul(W, Dpow)
    MS = np.matmul(Dpow, WDpow)


    diff_eig, diff_vec_init = la.eigh(MS)

    size = np.size(diff_eig)
    #removing trivial eigenvalue
    diff_eig = np.delete(diff_eig, size - 1)
    # print(diff_vec_init)
    # print(diff_eig)

    #normalizing eigenvectors
    diff_vec_t = np.zeros((n, n))
    

    for i in range(n):
    	#isolating vector
    	vec_i = diff_vec_init[:,i]

    	#multiplying by powered D matrix
    	M_vec = np.dot(Dpow, vec_i)


    	#dividing/normalizing
    	scaldiv = la.norm(M_vec)
    	M_vec = np.true_divide(M_vec, scaldiv)

    	diff_vec_t[i] = M_vec

   
    #removing trivial eigenvector
    diff_vec_t = np.delete(diff_vec_t, n - 1, 0)

    #reverting to column form
    diff_vec = np.transpose(diff_vec_t)

    return diff_vec, diff_eig


def get_diff_map(diff_vec, diff_eig, t):
    '''
    Construct a diffusion map at t from eigenvalues and eigenvectors of Markov matrix

    Inputs:
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
        t           diffusion time parameter t

    Outputs:
        diff_map    a numpy array of size n x n-1, the diffusion map defined for t

    '''

    diff_eig = np.power(diff_eig, t)
    m = np.size(diff_eig)
    n = m + 1

    diff_map = np.zeros((n, n - 1))

    for i in range(n - 1):
        for j in range(n):
            diff_map[j, i] = diff_vec[j, i] * diff_eig[i]
    
    
    

    return diff_map




####### EXPERIMENT CODE ########
