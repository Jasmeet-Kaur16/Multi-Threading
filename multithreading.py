import os
import sys
import numpy as np
import multiprocessing
import threading
import time
import random as random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("OS Name: ", sys.platform)

# check the number of cores
numberOfCores = multiprocessing.cpu_count()
print("Number of cores in the system: ", numberOfCores)


# multiply two matrices
def multiply_matrices(mat1, mat2):
    return np.dot(mat1, mat2)


def matrix_multiplication_threading(num_of_matrices, size_matrix, main_matrix, results, lock_result):
    for _ in range(num_of_matrices):
        random_matrix = np.random.rand(*size_matrix)
        result = multiply_matrices(random_matrix, main_matrix)

        with lock_result:
            results.append(result)


def cal_time(num_of_threads):
    size_matrix = (1000, 1000)
    num_of_matrices = 100

    main_matrix = np.random.rand(*size_matrix)

    results = []
    lock_result = threading.Lock()

    threads = []

    matrices_per_thread = num_of_matrices // num_of_threads
    remaining_matrices = num_of_matrices % num_of_threads

    start_time = time.time()

    for _ in range(num_of_threads):
        if remaining_matrices > 0:
            num_of_matrices_this_thread = matrices_per_thread + 1
            remaining_matrices -= 1
        else:
            num_of_matrices_this_thread = matrices_per_thread

        thread = threading.Thread(
            target=matrix_multiplication_threading,
            args=(num_of_matrices_this_thread, size_matrix,
                  main_matrix, results, lock_result),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    end_time = time.time()

    # Calculate total time taken
    total_time = end_time - start_time

    print(f"Number of results: {len(results)}")
    print(
        f"Total time taken with {num_of_threads} threads: {total_time:.4f} seconds")
    return total_time


num_of_threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                  12, 13, 14, 15, 16]
time_taken = []
for i in range(0, len(num_of_threads)):
    time_taken.append(cal_time(num_of_threads[i]))

plt.plot(num_of_threads, time_taken, marker='o', linestyle='-')
plt.title('Execution Time')
plt.xlabel('Number of Threads')
plt.ylabel('Time Taken (seconds)')
plt.xticks(num_of_threads)
plt.show()
