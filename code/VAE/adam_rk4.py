from torch.optim import Optimizer
import math

class AdamRK4(Optimizer):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamRK4, self).__init__(params, defaults)


    def step(self, closure):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is not None:
            closure()
        grad_k1, grad_k2, grad_k3, grad_k4 = [], [], [], []

        for group in self.param_groups:
            p_real = [(p) for p in group['params']]

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad_k1.append(p.grad.data)
                p.data.add_(-group['lr'] / 2, grad_k1[i])

        closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                p.data = p_real[i].data
                for group_2 in self.param_groups:
                    grad_k2.append(group_2['params'][i].grad.data)

                p.data.add_(-group['lr'] / 2, grad_k2[i])

        closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                p.data = p_real[i].data
                for group_3 in self.param_groups:
                    grad_k3.append(group_3['params'][i].grad.data)

                p.data.add_(-group['lr'], grad_k3[i])

        closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                for group_4 in self.param_groups:
                    grad_k4.append(group_4['params'][i].grad.data)

        """
        correct adam implementation for adam
        """

        for group in self.param_groups:
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                # changed sign here for rk4_grad
                rk4_grad = grad_k1[j].add_(2, grad_k2[j]).add_(2, grad_k3[j]).add(grad_k4[j])

                #p_real[j].data.add_(group['lr'] / 6, rk4_grad)
                #p.data = p_real[j].data


                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of rk4_gradient values
                    state['exp_avg'] = rk4_grad.new().resize_as_(rk4_grad).zero_()
                    # Exponential moving average of squared rk4_gradient values
                    state['exp_avg_sq'] = rk4_grad.new().resize_as_(rk4_grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    rk4_grad = rk4_grad.add(group['weight_decay'], p_real[j].data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, rk4_grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, rk4_grad, rk4_grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p_real[j].data.addcdiv_(-step_size, exp_avg, denom)
                p.data = p_real[j].data

