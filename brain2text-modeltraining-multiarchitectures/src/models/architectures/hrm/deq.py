import torch


def fixed_point_solver(f, h0, tol=1e-3, max_iter=10):
    """Iterate f until convergence or max_iter, return the fixed-point estimate."""
    h = h0
    for _ in range(max_iter):
        h_new = f(h)
        if (h_new - h).norm() < tol:
            return h_new
        h = h_new
    return h


class OneStepGrad(torch.autograd.Function):
    """1-step DEQ gradient: dL/dtheta ≈ dL/dh* · df(h*;theta)/dtheta.

    Accumulates parameter gradients directly so the backward pass stays O(1)
    w.r.t. sequence length (patches are detached across boundaries).
    """

    @staticmethod
    def forward(ctx, h_star, l_module, patch, h_h, patch_size):
        ctx.save_for_backward(h_star, patch, h_h)
        ctx.l_module = l_module
        ctx.patch_size = patch_size
        return h_star

    @staticmethod
    def backward(ctx, grad_out):
        h_star, patch, h_h = ctx.saved_tensors
        l_module = ctx.l_module
        patch_size = ctx.patch_size
        with torch.enable_grad():
            h_d = h_star.detach().requires_grad_(True)
            h_iter = h_d
            for t in range(patch_size):
                h_iter = l_module(patch[:, t, :], h_iter, h_h.detach())
            param_list = [p for p in l_module.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                h_iter, param_list,
                grad_outputs=grad_out,
                allow_unused=True,
                retain_graph=False,
            )
        for p, g in zip(param_list, grads):
            if g is not None:
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.add_(g)
        # Pass grad_out through to h_star (identity Jacobian approximation)
        return grad_out, None, None, None, None
