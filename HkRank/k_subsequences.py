#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# We define a k-subsequence of an array as follows:

# It is a subsequence of contiguous elements in the array, i.e. a subarray.
# The sum of the subsequence's elements, s, is evenlt divisible by k (i.i: s % k = 0).
# Given an array of integers, determine the number of k-subsequences it contains.
# For example k = 5 and the array nums = [5, 10, 11, 9, 5].
# The 10 k-subsequences are: {5}, {5, 10}, {5, 10, 11, 9}, {5, 10, 11, 9, 5}, {10}, {10, 11, 9}, {10, 11, 9, 5}, {11, 9}, {11, 9, 5}, {5}.

# Function Description: Complete the function kSub in the editor below.
# The function must return an long integer that represents the number of k-subsequences.

# kSub has following parameter(s):

# k: an integer that the sum of the subsequence must be divisible by
# nums[nums[0], ..., nums[n-1]]: an array of integers
# Constraints:

# 1 <= n <= 3 x 10^5
# 1 <= k <= 100
# 1 <= nums[i] <= 10^4
# Input Format for Custom Testing

# The first line contains an integer k, the number the sum of the subsequence must be divisible by.
# The next line contains an integer n, that denotes the number of elements in nums.
# Each line i of the n subsequent lines (where 0 <= i < n) contains an integer that describes nums[i].

#def libs
import itertools
def kSub_all_combinations(input_list, division_value):

    result = []
    for L in range(0, len(list_of_nums)+1):

        for subset in itertools.combinations(list_of_nums, L):

            if sum(subset) % division_value == 0 and len(subset) != 0:

                result.append(subset) 

    return len(result)


def kSub(input_list, division_value):
    
    subsequent_vals = 0
    counter = 0

    for k in range(0, len(list_of_nums)):

        for i in range(0, len(list_of_nums)):

            subsequent_vals = subsequent_vals + list_of_nums[i]
            if subsequent_vals % division_value == 0:
                counter += 1

        del list_of_nums[0]
        subsequent_vals = 0 

    print (counter)


# dummy input
list_of_nums = [3,5,1,2,3,4,1]
k_subsequences_val = 3

kSub(list_of_nums, k_subsequences_val)