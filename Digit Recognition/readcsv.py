import numpy as np
def getcsvdata(filename) :
	a = np.genfromtxt(filename, skip_header = 1, delimiter = ",")
	return a

