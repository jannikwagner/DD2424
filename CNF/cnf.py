import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--yolo_dim', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="results_test")
parser.add_argument('--img', type=str, default="imgs/flag.png")
parser.add_argument('--aug_dim', type=int, default=0)
parser.add_argument('--de_augment', type=bool, default=False)
args = parser.parse_args()

print("--niters: " + str(args.niters) + ' --width: '+ str(args.width) + " --hidden_dim: " + str(args.hidden_dim))

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class CNF(nn.Module):
    """ CNF using a hyper network
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

class CNF2(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        blocksize = width * in_out_dim
        self.blocksize = blocksize

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2 * blocksize + width)


    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            params = t.reshape(1, 1)
            params = torch.tanh(self.fc1(params))
            params = self.fc3(params)

            # restructure
            params = params.reshape(-1)
            W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

            U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

            B = params[2 * self.blocksize:].reshape(self.width, 1, 1)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

class CNF3(nn.Module):
    def __init__(self, in_out_dim):
        super().__init__()
        self.in_out_dim = in_out_dim
        layers = [
            nn.Linear(in_out_dim, 2*in_out_dim), nn.BatchNorm1d(2*in_out_dim), nn.ReLU(),
            nn.Linear(2*in_out_dim, in_out_dim)
            ]
        
        self.seq = nn.Sequential(*layers)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            dz_dt = self.seq(z)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

from typing import List
import torch.nn.functional as F

class TimeIncorporatingLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation_fn=F.tanh, batch_norm=False):
        super().__init__()
        self.fc = nn.Linear(in_dim+1, out_dim)
        self.activation_fn = activation_fn
        self.bn = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
    
    def forward(self, x):
        t, z = x
        t_tensor = torch.zeros((z.shape[0], 1)) + t
        
        z = torch.cat((z, t_tensor), 1)
        z = self.fc(z)
        z = self.activation_fn(z)
        z = self.bn(z)

        return t, z


class CNF_2layer(nn.Module):
    def __init__(self, in_out_dim):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.fc1 = nn.Linear(in_out_dim+1, 2*in_out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2*in_out_dim+1, in_out_dim)

    def forward(self, t, states: List[torch.Tensor]):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            t_tensor = torch.zeros((z.shape[0], 1)) + t
            
            x = torch.cat((z, t_tensor), 1)

            x = self.fc1(x)
            x = self.relu(x)

            x = torch.cat((x, t_tensor), 1)

            dz_dt = self.fc2(x)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

class CNF_time_incorp_4layers_narrow(nn.Module):
    """
    CNF with time 4 incoporating layers
    """
    def __init__(self, in_out_dim):
        super().__init__()
        self.in_out_dim = in_out_dim
        layers = [TimeIncorporatingLayer(in_out_dim, 2 * in_out_dim),
                  TimeIncorporatingLayer(2*in_out_dim, 4 * in_out_dim),
                  TimeIncorporatingLayer(4*in_out_dim, 2 * in_out_dim),
                  TimeIncorporatingLayer(2*in_out_dim, in_out_dim, activation_fn=nn.Identity(), batch_norm=False),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, t, states: List[torch.Tensor]):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            _, dz_dt = self.seq((t, z))

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

class CNF_time_incorp_4layers(nn.Module):
    """
    CNF with 4 time incoporating layers
    """
    def __init__(self, in_out_dim):
        super().__init__()
        self.in_out_dim = in_out_dim
        layers = [TimeIncorporatingLayer(in_out_dim, 32),
                  TimeIncorporatingLayer(32, 64),
                  TimeIncorporatingLayer(64, 32),
                  TimeIncorporatingLayer(32, in_out_dim, activation_fn=nn.Identity(), batch_norm=False),
        ]
        self. seq = nn.Sequential(*layers)

    def forward(self, t, states: List[torch.Tensor]):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            _, dz_dt = self.seq((t, z))

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

class CNF_time_incorp_6layers(nn.Module):
    """
    CNF with 6 time incoporating layers. Hidden dimensions: 32, 64, 64, 64, 32
    """
    def __init__(self, in_out_dim):
        super().__init__()
        self.in_out_dim = in_out_dim
        layers = [TimeIncorporatingLayer(in_out_dim, 32),
                  TimeIncorporatingLayer(32, 64),
                  TimeIncorporatingLayer(64, 64),
                  TimeIncorporatingLayer(64, 64),
                  TimeIncorporatingLayer(64, 32),
                  TimeIncorporatingLayer(32, in_out_dim, activation_fn=nn.Identity(), batch_norm=False),
        ]
        self. seq = nn.Sequential(*layers)

    def forward(self, t, states: List[torch.Tensor]):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            _, dz_dt = self.seq((t, z))

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1] - args.aug_dim):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


class HyperNetwork4Layers(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time. With 4 layers

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width, yolo_dim):
        super().__init__()

        blocksize = width * in_out_dim * yolo_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 3 * blocksize + width*yolo_dim)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize
        self.yolo_dim = yolo_dim

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = torch.tanh(self.fc3(params))
        params = self.fc4(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, self.yolo_dim)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, self.yolo_dim, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, self.yolo_dim, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, self.yolo_dim)
        return [W, B, U]


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_batch_from_circles(num_samples):
    # get the points in circles
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return(x, logp_diff_t1)

img = np.array(Image.open(args.img).convert('L'))

h, w = img.shape
xx = np.linspace(-1.5, 1.5, w)
yy = np.linspace(-1.5, 1.5, h)
xx, yy = np.meshgrid(xx, yy)
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)

means = np.concatenate([xx, yy], 1)
img = img.max() - img
probs = img.reshape(-1) / img.sum()

std = np.array([8 / w / 2, 8 / h / 2])

def get_batch(num_samples):
    # get the points from the image specified in args.img
    inds = np.random.choice(int(probs.shape[0]), int(num_samples), p=probs)
    m = means[inds]
    points = np.random.randn(*m.shape) * std + m
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return(x, logp_diff_t1)


if __name__ == '__main__':
    t0 = 0
    t1 = 10
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # model
    func = CNF(in_out_dim = 2 + args.aug_dim, hidden_dim=args.hidden_dim, width=args.width).to(device)
    # func = CNF_time_incorp_6layers(in_out_dim=2 + args.aug_dim).to(device)
    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    if args.de_augment:
        p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 0.0]).to(device),
            covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
        )
    else:
        p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.tensor(torch.zeros(2+args.aug_dim)).to(device),
            covariance_matrix=torch.eye(2+args.aug_dim).to(device)
        )
    loss_meter = RunningAverageMeter()

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))
    # tol = tol_control.get_tol()
    tol = 10**-5
    try:
        filename = '../loss_results/loss_niter_' + str(args.niters) + '_augdim_' + str(args.aug_dim) + "_width" + str(args.width) + "_hidden" + str(args.hidden_dim)
        f = open(filename, 'w', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['iteration', 'running avg loss'])

        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()

            x, logp_diff_t1 = get_batch(args.num_samples)
            x_aug = torch.cat([x, torch.zeros([x.shape[0], args.aug_dim])], 1)

            z_t, logp_diff_t = odeint(
                func,
                (x_aug, logp_diff_t1),
                torch.tensor([t1, t0]).type(torch.float32).to(device),
                atol=tol,
                rtol=tol,
                method='dopri5',
            )

            if args.aug_dim > 0 and args.de_augment:
                z_t = z_t[..., :-args.aug_dim]

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
            # get probability of z_t0 according to p_z0 and add it to difference (due to change of vars) to get probability of x according to p_x
            logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
            loss = -logp_x.mean(0)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())

            print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))
            writer.writerow([itr, loss_meter.avg])

        f.close()

    except KeyboardInterrupt:
        pass
    finally:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.viz:
        viz_samples = 3000
        viz_timesteps = 41
        target_sample, _ = get_batch(viz_samples)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with torch.no_grad():
            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

            z_t0_aug = torch.cat([z_t0, torch.zeros([z_t0.shape[0], args.aug_dim])], 1)
            # z_t0_aug = torch.cat([z_t0, torch.normal(0, 1, [z_t0.shape[0], args.aug_dim]) ], 1)

            z_t_samples, _ = odeint(
                func,
                (z_t0_aug, logp_diff_t0),
                torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )
            if args.aug_dim > 0 and args.de_augment:
                z_t_samples = z_t_samples[..., :-args.aug_dim]

            # Generate evolution of density
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = torch.tensor(points).type(torch.float32).to(device)
            logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)            
            # logp_diff_t1 = torch.normal(0, 1, [z_t1.shape[0], 1]).type(torch.float32).to(device)
            
            z_t1_aug = torch.cat([z_t1, torch.zeros([z_t1.shape[0], args.aug_dim])], 1)
            # z_t1_aug = torch.cat([z_t1, torch.normal(0, 1, [z_t1.shape[0], args.aug_dim])], 1)

            z_t_density, logp_diff_t = odeint(
                func,
                (z_t1_aug, logp_diff_t1),
                torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )            
            if args.aug_dim > 0 and args.de_augment:
                z_t_density = z_t_density[..., :-args.aug_dim]


            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace(t0, t1, viz_timesteps),
                    z_t_samples, z_t_density, logp_diff_t
            ):
                fig = plt.figure(figsize=(12, 4), dpi=200)
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                fig.suptitle(f'{t:.2f}s')

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Target')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('Samples')
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Log Probability')
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])

                ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(args.results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz.gif")))
