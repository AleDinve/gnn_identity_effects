##Alphabet one-hot encoding
import numpy as np
from scipy.stats import ortho_group
from string import ascii_uppercase as letters


def one_hot_enc():
    encodings = []
    for value in range(26):
        letter = [0 for _ in range(26)]
        letter[value] = 1
        encodings.append(letter)
    return encodings

def haar_enc():
    encodings = []
    A = ortho_group.rvs(26)
    for i in range(26):
        encodings.append(A[i,:])
    return encodings

def gaussian_enc():
    encodings = []
    dim = 26
    mean = np.zeros((dim))
    cov = np.eye(dim,dim)
    A = np.random.multivariate_normal(mean,cov, size = dim)
    for i in range(26):
        encodings.append(A[i,:])
    return encodings

def gaussian_red_enc(dim=16):
    encodings = []
    mean = np.zeros((dim))
    cov = np.eye(dim,dim)
    A = np.random.multivariate_normal(mean,cov, size = 26)
    for i in range(26):
        encodings.append(A[i,:])
    return encodings

def sanity_check_enc():
    encodings = []
    for value in range(24):
        letter = [0 for _ in range(26)]
        letter[value] = 1
        encodings.append(letter)
    #making Y and Z linear combinations of previous letter encodings 
    y_enc= [0 for _ in range(26)]
    y_enc[0] = 2
    #y_enc[1] = 1
    encodings.append(y_enc)
    z_enc=[0 for _ in range(26)]
    z_enc[1] = 2
    encodings.append(z_enc)
    return encodings

def distr_j_enc(k=26, j=6):
    array_dict = []
    str_dict = {}
    for i, letter in enumerate(letters):
        indexes = np.random.choice(a=k, size=j, replace=False)
        encoding = [1 if i in indexes else 0 for i in range(k)]
        encoding_str = ''.join(str(b) for b in encoding)
        while encoding_str in str_dict.values():
            indexes = np.random.choice(a=k, size=j, replace=False)
            encoding = [1 if i in indexes else 0 for i in range(k)]
            encoding_str = ''.join(str(b) for b in encoding)
        array_dict.append(encoding)
        str_dict[letter] = encoding_str
    return array_dict

def get_encodings_list(j=3, dim=16):
    encodings = {'one-hot': one_hot_enc(),
                 'haar': haar_enc(),
                 'distributed': distr_j_enc(k=26,j=j),
                 'gaussian': gaussian_enc(),
                 'gaussian_red': gaussian_red_enc(dim=dim),
                 'sanity_check':sanity_check_enc()}
    return encodings
