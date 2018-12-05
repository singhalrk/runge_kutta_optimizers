# from torch.optim import Optimizer
# import torch

# class RK2_heun(Optimizer):
#     """
#     Arguments:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float, optional): learning rate (default: 1e-3)
#     """
#     def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
#                  weight_decay=0, nesterov=False):

#         defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
#                         weight_decay=weight_decay, nesterov=nesterov)
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")

#         super(RK2_heun, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(RK2_heun, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('nesterov', False)



#     def step(self, closure):
#         if closure is not None: closure()
#         grad_k1, grad_k2= [], []

#         for group in self.param_groups:
#             p_real = [(p) for p in group['params']]
#             weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             dampening = group['dampening']
#             nesterov = group['nesterov']


#         for group in self.param_groups:
#             for i, p in enumerate(group['params']):
#                 if p.grad is None:
#                     continue

#                 grad_k1.append(p.grad.data)
#                 p.data.add_(-group['lr'], grad_k1[i])

#         closure()
#         for group in self.param_groups:
#             for k, p in enumerate(group['params']):
#                 if p.grad is None:
#                     continue

#                 #p.data = p_real[k].data
#                 for group_2 in self.param_groups:
#                     grad_k2.append(group_2['params'][k].grad.data)

#         for group in self.param_groups:

#             for j, p in enumerate(group['params']):
#                 d_p = grad_k1[j].add_(1.0, grad_k2[j])

#                 #print(p.data.shape, p_real[j].data.shape)
#                 if weight_decay != 0:
#                     d_p.add_(weight_decay, p_real[j].data)
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         buf = param_state['momentum_buffer'] = torch.zeros(p_real[j].data.size())
#                         buf.mul_(momentum).add_(d_p.cpu())
#                         #d_p.cuda()
#                         #buf.cuda()
#                     else:
#                         buf = param_state['momentum_buffer']
#                         buf.mul_(momentum).add_(1 - dampening, d_p.cpu())
#                     if nesterov:
#                         d_p = d_p.add(momentum, buf)
#                     else:
#                         d_p = buf

#                 p_real[j].data.add_(-group['lr']/2, d_p.cuda())
#                 p.data = p_real[j].data
#         return closure()

from torch.optim import Optimizer
import torch
from copy import deepcopy

class RK2_heun(Optimizer):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
    """
    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(RK2_heun, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK2_heun, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)



    def step(self, closure):
        if closure is not None: closure()
        grad_k1, grad_k2= [], []

        for group in self.param_groups:
            p_real = [(deepcopy(p)) for p in group['params']]
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']


        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    grad_k1.append(None)
                    continue

                d_p_1 = p.grad.data

                if weight_decay != 0:
                    d_p_1.add_(weight_decay, p.data)

                grad_k1.append(deepcopy(d_p_1))
                p.data.add_(-group['lr'], d_p_1)

        closure()
        for group in self.param_groups:
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    # grad_k2.append(None)
                    continue
        #         grad_k2.append(p.grad.data)

        #         #p.data = p_real[k].data
        #         # for group_2 in self.param_groups:
        #             # grad_k2.append(group_2['params'][k].grad.data)

        # for group in self.param_groups:

        #     for j, p in enumerate(group['params']):
        #         if p.grad is None:
        #             continue
                d_p = grad_k1[j].add_(1.0, p.grad.data)

                #print(p.data.shape, p_real[j].data.shape)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p_real[j].data)
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.zeros(p_real[j].data.size())
                #         buf.mul_(momentum).add_(d_p.cpu())
                #         #d_p.cuda()
                #         #buf.cuda()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p.cpu())
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf

                p_real[j].data.add_(-group['lr']/2, d_p)
                p.data = p_real[j].data
                # p.data = p.data.add_(weight_decay, torch.randn(p.data.size()).cuda())
        return closure()
