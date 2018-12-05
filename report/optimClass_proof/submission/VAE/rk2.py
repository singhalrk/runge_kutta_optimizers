from torch.optim import Optimizer


class RK2(Optimizer):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
    """
    def __init__(self, params, lr=1e-2,):

        defaults = dict(lr=lr)
        super(RK2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK2, self).__setstate__(state)


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
        grad_k1, grad_k2= [], []

        for group in self.param_groups:
            p_real = [(p) for p in group['params']]

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad_k1.append(p.grad.data)
                p.data.add_(-group['lr']*2/3, grad_k1[i])

        closure()
        for group in self.param_groups:
            for k, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                p.data = p_real[i].data
                for group_2 in self.param_groups:
                    grad_k2.append(group_2['params'][k].grad.data)

        for group in self.param_groups:

            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                rk2_grad = grad_k1[j].add_(3, grad_k2[j])

                p_real[j].data.add_(-group['lr']/4, rk2_grad)
                p.data = p_real[j].data
