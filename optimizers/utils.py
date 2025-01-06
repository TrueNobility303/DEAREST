import numpy as np 

# return the minimum, median, and maximum of an array
def quantile(array):
    sorted_array = np.sort(array)
    mid = int(len(array) / 2)
    return sorted_array[0], sorted_array[mid], sorted_array[-1] 