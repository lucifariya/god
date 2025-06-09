#!/usr/bin/env python
# coding: utf-8

# # Computer Science Thinking: Recursion, Searching, Sorting and Big O
# ### github : https://github.com/pdeitel/IntroToPython/tree/master/examples/ch11 
# ### book   : http://localhost:8888/files/2241016309/Python/Python%202/Python%20Book.pdf 
#             (only works in lab comp)

# ### Factorials
# **Iterative Factorial Approach:**

# In[1]:


factorial = 1
for number in range(5, 0, -1):
    factorial *= number
print(factorial)


# <hr>  </hr>  
#   
# **Recursive Faqctorial Approach:**

# In[2]:


def factorial(number):
    """Return factorial of number."""
    if number <= 1:
        return 1
    return number * factorial(number - 1)  # recursive call

for i in range(11):
    print(f'{i}! = {factorial(i)}')


# In[4]:


factorial(50)

# Java’s int type can represent only values in the range –2,147,483,648 to+2,147,483,647.
# But Python allows integers to become arbitrarily large. 


# <hr>  </hr>  
#   
# ### Fibonacci Series: 
# **Recursive Fibonacci Series Example:**

# In[5]:


def fibonacci(n):
    if n in (0, 1):  # base cases
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
    
for n in range(41):
    print(f'Fibonacci({n}) = {fibonacci(n)}')


# > speed of the calculation slows substantially as we get near the end of the loop. The variable n indicates which Fibonacci number to calculate in each iteration of the loop.  
#   
# <hr>  </hr>  
#   
# **Iterative Fibonacci:**

# In[6]:


def iterative_fibonacci(n):
    result = 0
    temp = 1
    for j in range(0, n):
        temp, result = result, result + temp
    return result


# In[7]:


get_ipython().run_line_magic('timeit', 'fibonacci(32)')


# In[8]:


get_ipython().run_line_magic('timeit', 'iterative_fibonacci(32)')


# In[9]:


get_ipython().run_line_magic('timeit', 'fibonacci(34)')


# In[10]:


get_ipython().run_line_magic('timeit', 'iterative_fibonacci(34)')


# <hr>  </hr>  
#   
# ### Searching and Sorting
# **Linear Search Implementation:**

# In[11]:


def linear_search(data, search_key):
    for index, value in enumerate(data):
        if value == search_key:
            return index
    return -1

import numpy as np

np.random.seed(11)

values = np.random.randint(10, 91, 10)

values


# In[12]:


linear_search(values, 23)


# In[13]:


linear_search(values, 61)


# In[14]:


linear_search(values, 34)


# <hr>  </hr>  
#   
#   **Binary Search Implementation:**

# In[3]:


import numpy as np

def binary_search(data, key):
    low = 0
    high = len(data) - 1
    middle = (low + high + 1) // 2
    location = -1

    while low <= high and location == -1:
        print(remaining_elements(data, low, high))
        print('   ' * middle + ' * ')

        if key == data[middle]:
            location = middle
        elif key < data[middle]:
            high = middle - 1
        else:
            low = middle + 1

        middle = (low + high + 1) // 2

    return location

def remaining_elements(data, low, high):
    return '   ' * low + ' '.join(str(s) for s in data[low:high + 1])

def main():
    data = np.random.randint(10, 91, 15)
    data.sort()
    print(data, '\n')

    search_key = int(input('Enter an integer value (-1 to quit): '))
    while search_key != -1:
        location = binary_search(data, search_key)
        if location == -1:
            print(f'{search_key} was not found\n')
        else:
            print(f'{search_key} found in position {location}\n')

        search_key = int(input('Enter an integer value (-1 to quit): '))
        
main()


# <hr>  </hr>  
#   
# **Selection Sort Implementation:**

# In[5]:


def print_pass(data, pass_number, index): 
    label = f'after pass {pass_number}: '
    print(label, end='')

    print('  '.join(str(d) for d in data[:index]), end='  ' if index != 0 else '') 
    print(f'{data[index]}* ', end='')
    print('  '.join(str(d) for d in data[index + 1:]))
    print(f'{" " * len(label)}{"--  " * pass_number}')  


# In[11]:


import numpy as np

def selection_sort(data):
    for index1 in range(len(data) - 1):
        smallest = index1
        for index2 in range(index1 + 1, len(data)): 
            if data[index2] < data[smallest]:
                smallest = index2
        data[smallest], data[index1] = data[index1], data[smallest]  
        print_pass(data, index1 + 1, smallest)

def main(): 
    data = np.array([34, 56, 14, 20, 77, 51, 93, 30, 15, 52])
    print(f'Unsorted array: {data}\n')
    selection_sort(data) 
    print(f'\nSorted array: {data}\n')
    
main()


# **Insertion Sort Implementation:** 

# In[14]:


import numpy as np

def insertion_sort(data):
    for next in range(1, len(data)):
        insert = data[next]
        move_item = next

        while move_item > 0 and data[move_item - 1] > insert:
            data[move_item] = data[move_item - 1]
            move_item -= 1

        data[move_item] = insert
        print_pass(data, next, move_item)

def main(): 
    data = np.array([34, 56, 14, 20, 77, 51, 93, 30, 15, 52])
    print(f'Unsorted array: {data}\n')
    insertion_sort(data)
    print(f'\nSorted array: {data}\n')

main()


# <hr>  </hr>  
#   
# **Merge Sort Implementation:** 

# In[16]:


import numpy as np

def merge_sort(data):
    sort_array(data, 0, len(data) - 1)

def sort_array(data, low, high):
    if (high - low) >= 1:
        middle1 = (low + high) // 2
        middle2 = middle1 + 1

        print(f'split:   {subarray_string(data, low, high)}') 
        print(f'         {subarray_string(data, low, middle1)}') 
        print(f'         {subarray_string(data, middle2, high)}\n') 

        sort_array(data, low, middle1)
        sort_array(data, middle2, high)

        merge(data, low, middle1, middle2, high)

def merge(data, left, middle1, middle2, right):
    left_index = left
    right_index = middle2
    combined_index = left
    merged = [0] * len(data)

    print(f'merge:   {subarray_string(data, left, middle1)}') 
    print(f'         {subarray_string(data, middle2, right)}')

    while left_index <= middle1 and right_index <= right:
        if data[left_index] <= data[right_index]:
            merged[combined_index] = data[left_index]
            left_index += 1
        else:
            merged[combined_index] = data[right_index]
            right_index += 1
        combined_index += 1

    if left_index == middle2:
        merged[combined_index:right + 1] = data[right_index:right + 1]
    else:
        merged[combined_index:right + 1] = data[left_index:middle1 + 1]

    data[left:right + 1] = merged[left:right + 1]

    print(f'         {subarray_string(data, left, right)}\n')

def subarray_string(data, low, high):
    return '   ' * low + ' '.join(str(item) for item in data[low:high + 1])

def main():
    data = np.array([34, 56, 14, 20, 77, 51, 93, 30, 15, 52])
    print(f'Unsorted array: {data}\n')
    merge_sort(data)
    print(f'\nSorted array: {data}\n')

main()


# <hr></hr>
#   
# **Implementing the Selection Sort Animation:**

# In[23]:


"""Functions to play sounds."""
from pysine import sine

TWELFTH_ROOT_2 = 1.059463094359  # 12th root of 2
A3 = 220  # hertz frequency for musical note A from third octave 

def play_sound(i, seconds=0.1):
    """Play a note representing a bar's magnitude. Calculation 
    based on https://pages.mtu.edu/~suits/NoteFreqCalcs.html."""
    sine(frequency=(A3 * TWELFTH_ROOT_2 ** i), duration=seconds)
    
def play_found_sound(seconds=0.1):
    """Play sequence of notes indicating a found item."""
    sine(frequency=523.25, duration=seconds) # C5
    sine(frequency=698.46, duration=seconds) # F5
    sine(frequency=783.99, duration=seconds) # G5

def play_not_found_sound(seconds=0.3):
    """Play a note indicating an item was not found."""
    sine(frequency=220, duration=seconds) # A3


# In[ ]:





# In[ ]:




