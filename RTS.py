import numpy as np
from numpy import random
from numpy.random import randn
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import kf_book.book_plots as bp
from Generator import *

def plot_rts(input_array,K=4, noise=1, Q=0.001, show_velocity=False):
    random.seed(1223)
    fk = KalmanFilter(dim_x=2, dim_z=1)

    fk.x = np.array([0., 1.])      # state (x and dx)

    fk.F = np.array([[1., 1.],
                     [0., 1.]])    # state transition matrix

    fk.H = np.array([[1., 0.]])    # Measurement function
    fk.P = 10.                     # covariance matrix
    fk.R = noise                   # state uncertainty
    fk.Q = Q                       # process uncertainty

    # create noisy data
    zs=np.zeros(len(input_array))
    for t in range(len(input_array)):
        zs[t]=input_array[t]*K + randn()*noise
    #zs = np.asarray([t + randn() * noise for t in range(40)])

    # filter data with Kalman filter, than run smoother on it
    mu, cov, _, _ = fk.batch_filter(zs)
    M, P, C, _ = fk.rts_smoother(mu, cov)
    # plot data
    if show_velocity:
        index = 1
        print('gu')
    else:
        index = 0
    if not show_velocity:
        bp.plot_measurements(zs)
    plt.plot(M[:, index], c='b', label='RTS')
    plt.plot(mu[:, index], c='g', ls='--', label='KF output')
    if not show_velocity:
        N = len(zs)
    plt.legend(loc=4)
    plt.show(block=0)
    return ((M , mu))

def page_on_page(bin_array,x_array,figure_number,is_block):
    number_of_spikes=0
    for i in range(len(bin_array)):
        if(bin_array[i]==1):
            number_of_spikes+=1

    page=np.zeros((number_of_spikes,20))
    row=0
    column=0
    for i in range(len(bin_array)):
        if(bin_array[i]==1):
            if(i!=0):
                page[row,column]=x_array[i-1]
                column+=1
                page[row,column]=x_array[i]
                column+=1
            else:
                page[row, column] = x_array[i]
                column += 1
            j=i+1
            while((bin_array[j] != 1) and (j<len(bin_array)-1) and (column<20)):
                page[row, column] = x_array[j]
                column += 1
                i += 1
                j += 1
            column = 0
            row += 1
    plt.figure(figure_number)
    for i in range(number_of_spikes):
        plt.plot(page[i])

    plt.show(block=is_block)


bin_array = create_binary_samples(10000, 1000)
noisy_bin_array = get_noisy_bin_array(bin_array, 0, 0.25, 0.7)
x_array = create_x_array(noisy_bin_array, 5, 0.5)
print(x_array)
plt.figure(1)
plt.plot(x_array, c='r', ls='--', label='Reference')
plt.plot(bin_array, c='b', ls='--', label='Real Nuironic activity')
plt.plot(noisy_bin_array, c='r', label='Noisy Nuironic activity')
plt.show(block=0)


(RTS, kalman_result) = plot_rts(x_array, 1, Q=1, show_velocity=False)
page_on_page(noisy_bin_array, RTS[:, 0], 2, 0)
page_on_page(noisy_bin_array, kalman_result[:, 0], 3, 1)