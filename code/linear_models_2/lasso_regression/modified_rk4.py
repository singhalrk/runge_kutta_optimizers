from torch.optim import Optimizer
import torch

class RK4(Optimizer):
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

        super(RK4, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK4, self).__setstate__(state)
        for group in self.param_groups:  
            group.setdefault('nesterov', False)

    def step(self, closure):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Note add the following function
        to your train function, so you pass

        for input, target in dataset:
            def closure():
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                return loss
            optimizer.step(closure)
        """
        if closure is not None: closure()
        grad_k1, grad_k2, grad_k3, grad_k4 = [], [], [], []

        for group in self.param_groups:
            p_real = [(p) for p in group['params']]
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']   

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad_k1.append(-p.grad.data)
                p.data.add_(group['lr'] / 2, grad_k1[i])

        closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                p.data = p_real[i].data
                for group_2 in self.param_groups:
                    grad_k2.append(-group_2['params'][i].grad.data)

                p.data.add_(group['lr'] / 2, grad_k2[i])

        closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                p.data = p_real[i].data
                for group_3 in self.param_groups:
                    grad_k3.append(-group_3['params'][i].grad.data)

                p.data.add_(group['lr'], grad_k3[i])

        closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                for group_4 in self.param_groups:
                    grad_k4.append(-group_4['params'][i].grad.data)

        for group in self.param_groups:

            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                d_p = grad_k1[j].add_(2, grad_k2[j]).add_(2, grad_k3[j]).add(grad_k4[j])

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p_real[j].data.size())
                        buf.mul_(momentum).add_(d_p.cpu())
                        #d_p.cuda()
                        #buf.cuda()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p.cpu())
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p_real[j].data.add_(group['lr'] / 6, d_p.cuda())
                p.data = p_real[j].data

        return closure()
