from torch.optim import Optimizer
import torch

class RK2_heun_fast(Optimizer):
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

        super(RK2_heun_fast, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK2_heun_fast, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('nesterov', False)



    def step(self, closure):
        if closure is not None: closure()
        grad_k1, grad_k2= [], []

        for group in self.param_groups:
            p_real = [(p) for p in group['params']]
            weight_decay = group['weight_decay']

            grad_k1 = [(p.grad.data) for p in group['params'] if p.grad is not None]

        # for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                # Consider adding weight decay term here as well
                # might just prevent the massive overfitting

                # grad_k1.append(p.grad.data)
                # p.data.add_(-group['lr'], grad_k1[i])

                d_p = grad_k1[i]
                # if weight_decay != 0:
                    # d_p.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], d_p)

        closure()
        for group in self.param_groups:
            # for k, p in enumerate(group['params']):
            #     if p.grad is None:
            #         continue
            grad_k2 = [(q.grad.data) for q in group['params'] if q.grad is not None]
                #p.data = p_real[k].data
                # for group_2 in self.param_groups:
                    # grad_k2.append(group_2['params'][k].grad.data)

        # for group in self.param_groups:

            for j, p in enumerate(group['params']):
                d_p = grad_k1[j].add_(1.0, grad_k2[j])

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

                p_real[j].data.add_(-group['lr']/2, d_p.cuda())
                p.data = p_real[j].data

        return closure()
