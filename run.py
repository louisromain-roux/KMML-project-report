import pandas as pd
import numpy as np
import cvxopt
from itertools import product

# The KRR implementation given running this script uses the simple_kernel kernel, while the kernel SVM uses the spectrum kernel
# Utility functions are listed first, then the bulk of the algorithms. The main wrappers are at the end of the file.

def parse_x_and_y(file_train_x, file_train_y, file_test_x):
    train_df_x = pd.read_csv(file_train_x)
    train_df_y = pd.read_csv(file_train_y)
    datax = train_df_x.to_numpy()
    datay = train_df_y.to_numpy()
    data_x = np.array([x[1] for x in datax])
    data_y = np.array([2*x[1]-1 for x in datay])

    test_df = pd.read_csv(file_test_x)
    testx = np.array(test_df)
    test_x = [x[1] for x in testx]
    
    return data_x, data_y, test_x


#Functions related to our basic kernel and KRR below

def simple_kernel(s1, s2, alpha=0.95):
  return (alpha)**(sum(c1 != c2 for c1, c2 in zip(s1, s2)))


def build_kernelmatrix(data, kernel, alpha=0.95):
  K = np.zeros((len(data), len(data)))
  for i, s in enumerate(data):
    for i2, s2 in enumerate(data):
      K[i,i2] = kernel(s, s2, alpha)

  return K

def fit_alpha(K, data_y, l=0.01):
  return np.linalg.inv(K+l*np.eye(K.shape[0]))@data_y


def predict_krr(alpha, kernel, data_x, y):
  s = 0
  for i in range(data_x.shape[0]):
    s += alpha[i]*kernel(y, data_x[i])
  return s

# Functions related to the spectrum kernel and kernel SVM below

def precompute_spec(data_x, k=4):
  preindexed_spectrums = []
  for i in range(len(data_x)):
    current_spectrum = {}
    for j in range(len(data_x[i])-k+1):
      if data_x[i][j:j+k] not in current_spectrum:
        current_spectrum[data_x[i][j:j+k]] = 1
      else:
        current_spectrum[data_x[i][j:j+k]] += 1
    preindexed_spectrums.append(current_spectrum)
  return preindexed_spectrums


def spec_kernel(sp1, sp2):
  s=0
  intersection = sp1.keys() & sp2.keys()
  for key in intersection:
    s += sp1[key]*sp2[key]
  return s


def build_specmatrix(preindexed_spectrums):
  K = np.zeros((len(preindexed_spectrums), len(preindexed_spectrums)))
  for i in range(len(preindexed_spectrums)):
    for j in range(len(preindexed_spectrums)):
      K[i,j] = spec_kernel(preindexed_spectrums[i], preindexed_spectrums[j])

  return K


def pred_spec(alpha, preindexed_spectrums, x, k=4):
    current_spectrum = {}
    for j in range(len(x)-k+1):
      if x[j:j+k] not in current_spectrum:
        current_spectrum[x[j:j+k]] = 1
      else:
        current_spectrum[x[j:j+k]] += 1
    s = 0
    for i in range(len(alpha)):
      s += alpha[i]*spec_kernel(preindexed_spectrums[i], current_spectrum)
    return s

def compute_multipliers(y, K, c=100):
    n_samples = y.shape[0]

    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(n_samples))
    G_low = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
    G_hi = cvxopt.matrix(np.diag(np.ones(n_samples)))
    G = cvxopt.matrix(np.vstack((G_low, G_hi)))
    h_low = cvxopt.matrix(np.zeros(n_samples))
    h_hi = cvxopt.matrix(np.ones(n_samples) * c)
    h = cvxopt.matrix(np.vstack((h_low, h_hi)))
    A = cvxopt.matrix(y.astype("float"), (1, n_samples))
    b = cvxopt.matrix(0.0)

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    return np.ravel(sol['x'])

def predict_svm(data_x_sv, data_y_sv, weight_SV, x, bias=0.0):
    result = bias
    for z_i, x_i, y_i in zip(weight_SV,
                              data_x_sv,
                              data_y_sv):
        result += z_i * y_i * simple_kernel(x_i, x)
    return np.sign(result).item()

def compute_bias(data_x_sv, data_y_sv, weight_SV, data_x, data_y):
    return np.mean([data_y[i] - predict_svm(data_x_sv, data_y_sv, weight_SV, data_x[i]) for i in range(len(data_x))])


#Wrappers below

def get_predictions_krr(file_train_x, file_train_y, file_test_x):
    data_x, data_y, test_x = parse_x_and_y(file_train_x, file_train_y, file_test_x)
    print("Building Gram matrix...")
    K = build_kernelmatrix(data_x, simple_kernel)
    print("Done.")
    alpha = fit_alpha(K, data_y)
    
    predictions = []
    print("Beginning predictions...")
    for i in range(len(test_x)):
        if(predict_krr(alpha, simple_kernel, data_x, test_x[i])>0):
          predictions.append(1)
        else:
          predictions.append(0)
    print("Done.")
    return predictions

def get_predictions_svm(file_train_x, file_train_y, file_test_x):
    data_x, data_y, test_x = parse_x_and_y(file_train_x, file_train_y, file_test_x)
    print("Building Gram matrix...")
    preindexed_spectrums = precompute_spec(data_x)
    K = build_specmatrix(preindexed_spectrums)
    print("Done.")
    print("Solving QP...")
    L = compute_multipliers(data_y, K)
    
    data_x_sv = data_x[L>1e-5]
    data_y_sv = data_y[L>1e-5]
    weight_SV = L[L>1e-5]
    bias = compute_bias(data_x_sv, data_y_sv, weight_SV, data_x, data_y)
    print("Done.")
    print("Beginning predictions...")
    predictions = []
    for i in range(len(test_x)):
        if(predict_svm(data_x_sv, data_y_sv, weight_SV, test_x[i], bias)>0):
          predictions.append(1)
        else:
          predictions.append(0)
    print("Done.")
    return predictions
    
try:
    pred_krr = []
    pred_svm = []
    for i in range(3):
        print("Running KRR on dataset n° {}".format(i+1))
        pred_krr += get_predictions_krr("Xtr" + str(i) + ".csv", "Ytr" + str(i) + ".csv", "Xte" + str(i) + ".csv")
        print("Running kernel SVM on dataset n° {}".format(i+1))
        pred_svm += get_predictions_svm("Xtr" + str(i) + ".csv", "Ytr" + str(i) + ".csv", "Xte" + str(i) + ".csv")
    
    f = open("result_KRR.txt", "w+")
    f.write("Id,Bound\n")
    for i in range(len(pred_krr)):
      f.write(str(i)+","+str(pred_krr[i])+"\n")
    f.close()
    f = open("result_SVM.txt", "w+")
    f.write("Id,Bound\n")
    for i in range(len(pred_svm)):
      f.write(str(i)+","+str(pred_svm[i])+"\n")
    f.close()
    
except FileNotFoundError:
    print("Please run in a directory containing the data files.")