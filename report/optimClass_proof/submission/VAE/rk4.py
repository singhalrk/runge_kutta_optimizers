from torch.optim import Optimizer


class RK4(Optimizer):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
    """
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(RK4, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK4, self).__setstate__(state)

    def step(self, closure):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss

        """
        if closure is not None:
            closure()

        for group in self.param_groups:

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                p_new_real = [p.data]

                grad_k1 = -p.grad.data
                p.data.add_(group['lr'] / 2, grad_k1)
                closure()

                p.data = p_new_real[0]
                for group_2 in self.param_groups:
                    grad_k2 = -group_2['params'][i].grad.data

                p.data.add_(group['lr'] / 2, grad_k2)
                closure()

                p.data = p_new_real[0]
                for group_3 in self.param_groups:
                    grad_k3 = -group_3['params'][i].grad.data

                p.data.add_(group['lr'], grad_k3)
                closure()

                p.data = p_new_real[0]
                for group_4 in self.param_groups:
                    grad_k4 = -group_4['params'][i].grad.data

                p_new_real[0].add_(group['lr'] / 6, grad_k1.add_(2, grad_k2).add_(2, grad_k3).add(grad_k4))

                p.data = p_new_real[0]




