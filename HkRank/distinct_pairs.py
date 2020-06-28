#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@


# Distinct pairs

# In this challenge, you will be given an array of integers and a target value.
# Determine the number of distinct pairs of elements in the array that sum up to the target value.
# Two pairs (a, b) and (c, d) are considered to be distinct if and only if the values in sorted order do not match i.e. (1, 9) and (9, 1) are indistinct but (1, 9) and (9, 2) are distinct.

# For Example, given the array [1, 2, 3, 6, 7, 8, 9, 1] and a target value of 10, the seven pairs (1, 9), (2, 8), (3, 7), (8, 2), (9, 1), (9, 1) and (1, 9) all sum to 10 and there are only three distinct pairs: (1, 9), (2, 8), and (3, 7).

# Function Description: Complete the function numberOfPairs in the editor below.
# The function must return an integer, the total number of distinct pairs of elements in the array that sum to the target value.

# numberOfPairs has the following parameter(s):

# a[a[0], ..., a[n-1]]: an array of integers to select the pairs from.
# Constraints:

# 1 <= n <= 5 x 10^5
# 0 <= a[i] <= 10^9
# 0 <= k <= 5 x 10^9
# Input Format for Custom Testing

# The first line contains an integer n, the size of the array a.
# The next n lines each contain an element a[i] where 0 <= i < n.
# The next line contains an integer k, the target value.

#def libs
import os
import scipy 
import numpy as np

# input 
input_arr = np.array([1, 2, 3, 6, 7, 8, 9, 1])
target_val = 10

# init 
distinct_pairs = []
for i in range (0, input_arr.shape[0]):

	for j in range(0, input_arr.shape[0]):

		if input_arr[i] + input_arr[j] == target_val:

			element = str(input_arr[i]) + str(input_arr[j])

			if element not in distinct_pairs and element[::-1] not in distinct_pairs:

				distinct_pairs.append(element)

print ('There are', len(distinct_pairs), 'distinct pairs that add to', target_val ,'in this example.')
print ('The pairs are as follows:')
print (distinct_pairs)