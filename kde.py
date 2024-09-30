import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Kernel functions
def gaussian_kernel(x, h):
    """
    Gaussian kernel function.
    
    Parameters:
    x (float or array-like): Input value(s).
    h (float): Bandwidth parameter.
    
    Returns:
    float or array-like: Kernel density estimate.
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / h) ** 2)

def laplacian_kernel(x, h):
    """
    Laplacian kernel function.
    
    Parameters:
    x (float or array-like): Input value(s).
    h (float): Bandwidth parameter.
    
    Returns:
    float or array-like: Kernel density estimate.
    """
    return (1 / 2 ) * np.exp(-np.abs(x) / h)

def uniform_kernel(x, h):
    """
    Uniform kernel function.
    
    Parameters:
    x (float or array-like): Input value(s).
    h (float): Bandwidth parameter.
    
    Returns:
    float or array-like: Kernel density estimate.
    """
    return np.where(np.abs(x) <= h, 1 / 2, 0)

def epanechnikov_kernel(x, h):
    """
    Epanechnikov kernel function.
    
    Parameters:
    x (float or array-like): Input value(s).
    h (float): Bandwidth parameter.
    
    Returns:
    float or array-like: Kernel density estimate.
    """
    return np.where(np.abs(x) <= h, 3 / 4 * (1 - (x / h) ** 2), 0)

def read_csv_to_array(file_path):
    """
    Read a CSV file and return the first column as a numpy array.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    numpy array: The first column of the CSV file.
    """
    try:
        df = pd.read_csv(file_path, header=None)
        return df.iloc[:, 0].values
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def make_folds(data, k):
    """
    Create K folds for cross-validation.
    
    Parameters:
    data (numpy array): The data to be split into K folds.
    k (int): The number of folds.
    
    Returns:
    list of tuples: Each tuple contains the training and validation sets for one fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    
    for train_index, val_index in kf.split(data):
        train_set, val_set = data[train_index], data[val_index]
        folds.append((train_set, val_set))
    
    return folds

def evaluate(input_value, train_set, kernel, bandwidth):
    """
    Evaluate the density estimate at the input value using KDE.
    
    Parameters:
    input_value (float): The point at which to evaluate the density.
    train_set (numpy array): The training set for KDE.
    kernel (function): The kernel function to use.
    bandwidth (float): The bandwidth parameter for the kernel.
    
    Returns:
    float: The density estimate at the input value.
    """
    n = len(train_set)
    density = np.sum(kernel(input_value - train_set, bandwidth)) / (n * bandwidth)
    return density

def select_optimal_bandwidth(data, kernel, k, bandwidths):
    """
    Select the optimal bandwidth that maximizes the sum of log estimated densities over the validation set.
    
    Parameters:
    data (numpy array): The data to be used for cross-validation.
    kernel (function): The kernel function to use.
    k (int): The number of folds for cross-validation.
    bandwidths (numpy array): The range of bandwidth values to try.
    
    Returns:
    float: The optimal bandwidth value.
    """
    folds = make_folds(data, k)
    best_bandwidth = None
    best_log_likelihood = -np.inf
    epsilon = 1e-10  # Small positive value to prevent log(0)

    for h in bandwidths:
        log_likelihood = 0
        for train_set, val_set in folds:
            for val in val_set:
                density = evaluate(val, train_set, kernel, h)
                log_likelihood += np.log(density + epsilon)  # Add epsilon to prevent log(0)
        
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_bandwidth = h

    return best_bandwidth

print("***1D Kernel Density Estimation***")
data = read_csv_to_array(input("Data file: "))
if data is None:
    exit()

print("Kernel Options")
print("gaussian, laplacian, uniform, epanechnikov")
kernel_t = input("Kernel Type: ").strip().lower().capitalize()
if kernel_t == "Gaussian":
    kernel = gaussian_kernel
elif kernel_t == "Laplacian":
    kernel = laplacian_kernel
elif kernel_t == "Uniform":
    kernel = uniform_kernel
elif kernel_t == "Epanechnikov":
    kernel = epanechnikov_kernel
else:
    print("Invalid kernel type")
    exit()

print("Bandwidth options")
print("auto: (k-fold validation)\nreal number: (h)")
h = input("Bandwidth: ")
if h == "auto":
    try:
        k = int(input("Number of Folds: "))
        bandwidths = np.linspace(0.05, 5.0, 99)
        h = select_optimal_bandwidth(data, kernel, k, bandwidths)
    except ValueError:
        print("Invalid number of folds")
        exit()
else:
    try:
        h = float(h)
    except ValueError:
        print("Invalid bandwidth value")
        exit()

while True:
    try:
        command = input("Enter command: ").split()
        if command[0] == "inference":
            if command[1] == "point":
                input_value = float(command[2])
                density = evaluate(input_value, data, kernel, h)
                print(f"Density estimate at {input_value}: {density}")
            elif command[1] == "range":
                command[2] = command[2].split(",")
                low = float(command[2][0])
                up = float(command[2][1])
                x = np.linspace(low, up, 1000)
                y = [evaluate(xi, data, kernel, h) for xi in x]
                p = np.trapz(y, x)
                print(f"Probability estimate at [{low},{up}]: {p}")
            else:
                print("Invalid command")
                continue

        elif command[0] == "plot":
            x = np.linspace(min(data), max(data), 1000)
            y = [evaluate(xi, data, kernel, h) for xi in x]
            plt.plot(x, y)
            plt.xlabel('Data')
            plt.ylabel('Density')
            plt.title(f'{kernel_t} Kernel Density Estimation: h={round(h, 2)}')
            plt.show()
        elif command[0] == "quit":
            break
        else:
            print("Invalid command")
            continue
    except Exception as e:
        print(f"Error: {e}")
        continue