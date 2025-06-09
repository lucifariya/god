#!/usr/bin/env python
# coding: utf-8

# # MINOR ASSIGNMENT-2: COMPUTER SCIENCE THINKING: RECURSION, SEARCHING, SORTING AND BIG O  

# <b>Question 1:</b>  
# Write a recursive function power(base, exponent) that, when called, returns
# > base^exponent
#   
# 

# In[1]:


def power(base, exponent):
    if exponent == 1:
        return base
    return base * power(base, exponent - 1)

power(3, 6)


# <b>Question 2:</b>  
# The greatest common divisor of integers x and y is the largest integer that evenly divides into both x
# and y. Write and test a recursive function gcd that returns the greatest common divisor of x and y.
# The gcd of x and y is defined recursively as follows: If y is equal to 0, then gcd(x, y) is x; otherwise,
# gcd(x, y) is gcd(y, x%y).

# In[2]:


def gcd(x, y):
    if y == 0:
        return x
    return gcd(y, x % y)

gcd(48, 18)


# <b>Question 3:</b>  
# Write a recursive function that takes a number n as an input parameter and prints n-digit strictly
# increasing numbers.

# In[3]:


def increase(n,st=1,curr=""):
    if len(curr)==n:
        print(curr,end=" ")
        return
    for i in range(st,10):
        increase(n,i+1,curr+str(i))

increase(3)


# <b>Question 4:</b>  
# Implement a recursive solution for computing the nth Fibonacci number. Then, analyze its time
# complexity. Propose a more efficient solution and compare the two approaches.

# In[4]:


# First Approach:

def fibRec(n):
    if n == 1 or n == 2:
        return 1
    return fibRec(n-1) + fibRec(n-2)

fibRec(20)

# Time Complexity: O(2^n)


# In[5]:


# Second Approach:

def fibMem(n, memo = None):
    if memo is None:
        memo = {}
        
    if n in memo:
        return memo[n]
    
    if n == 1 or n == 2:
        return 1
    
    memo[n] = fibMem(n-1, memo) + fibMem(n-2, memo);
    return memo[n]

fibMem(20)

# Time Complexity: O(n)


# <b>Question 5:</b>  
# Given an array of N elements, not necessarily in ascending order, devised an algorithm to find the kth largest one. It should run in O(N) time on random inputs.

# In[6]:


import random
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quickselect(arr, low, high, k):
    if low <= high:
        pivot_index = random.randint(low, high)
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
        pi = partition(arr, low, high)
        if pi == k:
            return arr[pi]
        elif pi > k:
            return quickselect(arr, low, pi - 1, k)
        else:
            return quickselect(arr, pi + 1, high, k)

def find_kth_largest(arr, k):
    n = len(arr)
    return quickselect(arr, 0, n - 1, n - k)

arr = [12, -3, 5, 7, 19, 8]
k = 4
find_kth_largest(arr, k)


# <b>Question 6:</b>  
# For each of the following code snippets, determine the time complexity in terms of Big O. Explain your answer.  
#   
# **(a) :**  
# ```python
# def example1(n):
#     for i in range(n):
#         for j in range(n):
#             print(i, j)
# ```
# **Ans-** Time Complexity: O(n^2)  
#   
# **(b) :**  
# ```python
# for i in range(n):
#     print(i)
# ```  
# **Ans-** Time Complexity: O(n)  
#   
# **(c) :**  
# ```python
# def recursive_function(n):
#     if n <= 1:
#         return 1
#     return recursive_function(n - 1) + recursive_function(n - 1)
# ```
# **Ans-** Time Complexity: O(2^n)  
#   

# <b>Question 7:</b>  
# Given N points on a circle, centered at the origin, design an algorithm that determines whether there
# are two points that are antipodal, i.e., the line connecting the two points goes through the origin. Your
# algorithm should run in time proportional to NlogN.

# In[7]:


import math
def find_antipodal_points(p):
    a=[]
    for (x,y) in p:
        a.append(math.atan2(y, x))
    a.sort()
    n = len(a)
    for i in range(n):
        t = (a[i] + math.pi) % (2 * math.pi)
        left, right = i + 1, n - 1
        while left <= right:
            m = (left + right) // 2
            if a[m] == t:
                return True
            elif a[m] < t:
                left = m + 1
            else:
                right = m - 1
    return False

p = [(1, 0), (0, 1), (-1, 0), (0, -1)]
find_antipodal_points(p)


# <b>Question 8:</b>  
# The **quicksort algorithm** is a recursive sorting technique that follows these steps:  
# 1. **Partition Step:** Choose the first element of the array as the pivot and determine its final position
# in the sorted array by ensuring all elements to its left are smaller and all elements to its right are larger.  
# 1. **Recursive Step:** Recursively repeat the partitioning process on the subarrays created on either side of the pivot.
#    
# As an example, consider the array [37, 2, 6, 4, 89, 8, 10, 12, 68, 45] with 37 as the pivot. Using the partitioning logic, the pivot eventually moves to its correct position, resulting in two subarrays: [12, 2, 6, 4, 10, 8] and [89, 68, 45]. The algorithm continues recursively until the entire array is sorted.  
#   
# Write a Python function quick_sort that implements the quicksort algorithm. The function should include a helper function quick_sort_helper to handle recursion. The helper function must take a starting and ending index as arguments and sort the array in-place. Demonstrate the function by sorting the given array and printing the sorted output.

# In[8]:


def partition(arr, low, high):
    pivot = arr[low]
    i = low + 1
    for j in range(low + 1, high + 1):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[low], arr[i - 1] = arr[i - 1], arr[low]
    return i - 1

def quick_sort_helper(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort_helper(arr, low, pi - 1)
        quick_sort_helper(arr, pi + 1, high)

def quick_sort(arr):
    quick_sort_helper(arr, 0, len(arr) - 1)

arr = [37, 2, 6, 4, 89, 8, 10, 12, 68, 45]
quick_sort(arr)
arr


# <b>Question 9:</b>  
# You are given the following list of famous personalities with their net worth:  
# • Elon Musk: 433.9 Billion  
# • Jeff Bezos: 239.4 Billion  
# • Mark Zuckerberg: 211.8 Billion  
# • Larry Ellison: 204.6 Billion  
# • Bernard Arnault & Family: 181.3  
# • Larry Page: 161.4 Billion
#      
# Develop a program to sort the aforementioned details on the basis of net worth using  
# a. Selection sort  
# b. Bubble sort  
# c. Insertion sort.  
#   
# The final sorted data should be the same for all cases. After you obtain the sorted data, present the result in the form of the following dictionary:  
# {’name1’:networth1, ’name2’:networth2,...}

# In[16]:


data = [
    ("Elon Musk", 433.9),
    ("Jeff Bezos", 239.4),
    ("Mark Zuckerberg", 211.8),
    ("Larry Ellison", 204.6),
    ("Bernard Arnault & Family", 181.3),
    ("Larry Page", 161.4)
]


# In[10]:


# Selection Sort:
def selection_sort(data):
    for i in range(len(data)):
        min_index = i
        for j in range(i + 1, len(data)):
            if data[j][1] > data[min_index][1]: 
                min_index = j
        data[i], data[min_index] = data[min_index], data[i]
    return data


# In[11]:


selection_sorted = selection_sort(data.copy())
print("Selection Sort Result:")
selection_sorted_dict = {name: networth for name, networth in selection_sorted}
selection_sorted_dict


# In[12]:


# Bubble Sort:
def bubble_sort(data):
    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j][1] < data[j + 1][1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data


# In[14]:


bubble_sorted = bubble_sort(data.copy())
print("\nBubble Sort Result:")
bubble_sorted_dict = {name: networth for name, networth in bubble_sorted}
bubble_sorted_dict


# In[15]:


# Insertion Sort
def insertion_sort(data):
    for i in range(1, len(data)):
        key = data[i]
        j = i - 1
        while j >= 0 and key[1] > data[j][1]:  # Sort by net worth (descending order)
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
    return data


# In[17]:


insertion_sorted = insertion_sort(data.copy())
print("\nInsertion Sort Result:")
insertion_sorted_dict = {name: networth for name, networth in insertion_sorted}
insertion_sorted_dict


# <b>Question 10:</b>  
# Use Merge Sort to sort a list of strings alphabetically. Example:  
# Input: [’apple’, ’orange’, ’banana’, ’grape’]  
# Output: [’apple’, ’banana’, ’grape’, ’orange’]

# In[18]:


def ms(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    return merge(ms(left), ms(right))


# In[19]:


def merge(left, right):
    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    res.extend(left[i:])
    res.extend(right[j:])
    return res


# In[20]:


lst = ['apple', 'orange', 'banana', 'grape']
ms(lst)


# <b>Question 11:</b>  
# Without using the built-in sorted() function, write a Python program to merge two pre-sorted lists into a single sorted list using the logic of Merge Sort. Example:  
# Input: [1, 3, 5, 7] and [2, 4, 6, 8]  
# Output: [1, 2, 3, 4, 5, 6, 7, 8]

# In[21]:


def merge_sorted_lists(list1, list2):
    i = j = 0
    merged_list = []
    
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1
    
    merged_list.extend(list1[i:])
    merged_list.extend(list2[j:])
    
    return merged_list


# In[22]:


list1 = [1, 3, 5, 7]
list2 = [2, 4, 6, 8]
merge_sorted_lists(list1, list2)

