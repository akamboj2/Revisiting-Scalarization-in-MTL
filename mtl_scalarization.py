import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from scipy import io
import os.path as osp
import random
from sklearn import preprocessing
import matplotlib
from tqdm import tqdm
import multiprocessing as mp
from matplotlib import animation


matplotlib.rc('font', size=12)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(0)

def load_SARCOS_numpy(file):
    data = io.loadmat(file)[osp.basename(file).split(".")[0]]
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    X = data[:,:21] # 21-dimensional input space (7 joint positions, 7 joint velocities, 7 joint accelerations) 
    y = data[:,[22, 23, 24]] # predicting the torques of arms 3, 4, 5
    return X, y

X_train_npy, y_train_npy = load_SARCOS_numpy("ECE513_FinalProject/sarcos_inv.mat")
X_test_npy, y_test_npy = load_SARCOS_numpy("ECE513_FinalProject/sarcos_inv_test.mat")

# X_train = torch.Tensor(X_train_npy); y_train = torch.Tensor(y_train_npy)
# X_test = torch.Tensor(X_test_npy); y_test = torch.Tensor(y_test_npy)
# print("Training data:", X_train.shape, y_train.shape)
# print("Testing data:", X_test.shape, y_test.shape)


y_hat = X_train_npy @ np.linalg.pinv((X_train_npy.T @ X_train_npy)) @ X_train_npy.T @ y_train_npy

# Generating the linear scalarization coefficients for training. Following the instruction in Appendix D.1. (page 22)
n_experiments = 10000
m = np.random.uniform(low=0.0, high=1.0, size=(n_experiments, 2))
all_w = np.concatenate([
    np.min(m, axis=1, keepdims=True),
    np.max(m, axis=1, keepdims=True) - np.min(m, axis=1, keepdims=True),
    1 - np.max(m, axis=1, keepdims=True)
], axis=1)

MAX_WORKERS = 2

def run(w):
    ld_mat = np.diag(np.sqrt(w))
    Y_hat_ld_mat = y_hat @ ld_mat
    U, S, Vh = np.linalg.svd(Y_hat_ld_mat, full_matrices=False)
    Y_hat_ld_mat_approx = S[0] * np.outer(U[:,0], Vh.T[:,0])
    training_err = ((Y_hat_ld_mat - Y_hat_ld_mat_approx)**2).mean(axis=0) + ((Y_hat_ld_mat - y_train_npy @ ld_mat)**2).mean(axis=0)
    # training_err = training_err/w
    return training_err

run(all_w[0])
if __name__ == '__main__':
    eval_res = []
    for w in tqdm(all_w):
        eval_res.append(run(w))
    # with mp.Pool(processes=MAX_WORKERS) as p:
    #     eval_res = p.map(run, all_w)
    #     # for i, _ in enumerate(p.imap_unordered(run, all_w), 1):
    #     #     sys.stderr.write('\rdone {0:%}'.format(i/len(all_w)))
    eval_res = np.array(eval_res)

    # import ipdb; ipdb.set_trace()
# Save figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(eval_res[:,0], eval_res[:,1], eval_res[:,2])
ax.set_xlabel(r'$MSE_1$')
ax.set_ylabel(r'$MSE_2$')
ax.set_zlabel(r'$MSE_3$')

# ax.set_xlim(0,1.0)
# ax.set_ylim(0,1.0)
# ax.set_zlim(0,1.0)

ax.invert_zaxis()
ax.invert_xaxis()

out_file = "mtl_scalarization"
plt.savefig(osp.join("ECE513_FinalProject", f"{out_file}.png"), bbox_inches="tight")
np.save(osp.join("ECE513_FinalProject", f"{out_file}.npy"), eval_res)

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

def init():
    ax.scatter(eval_res[:,0], eval_res[:,1], eval_res[:,2])
    return fig,

# Animate
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=360, interval=20, blit=True)
# writervideo = animation.PillowWriter(fps=30)
# anim.save("test.mp4", writer=writervideo)
# Save
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])