import numpy as np
from torch.autograd import Variable
import scipy.optimize
from utils import *


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).data[0]

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).data[0]
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


def trpo_step(policy_net, value_net, states, actions, returns, advantages, max_kl, damping, l2_reg):

    """update critic"""
    values_target = Variable(returns)

    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(Variable(states))
        value_loss = (values_pred - values_target).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.data.cpu().numpy()[0], get_flat_grad_from(value_net).data.cpu().numpy()

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).cpu().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    """update policy"""
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data
    """define the loss function for TRPO"""
    def get_loss(volatile=False):
        log_probs = policy_net.get_log_prob(Variable(states, volatile=volatile), Variable(actions))
        action_loss = -Variable(advantages) * torch.exp(log_probs - Variable(fixed_log_probs))
        return action_loss.mean()

    """use fisher information matrix for Hessian*vector"""
    def Fvp_fim(v):
        cov_inv, mean = policy_net.get_fim(Variable(states))
        param_count = 0
        std_index = 0
        res = zeros(v.shape)
        for name, param in policy_net.named_parameters():
            if name == "action_log_std":
                std_index = param_count
            param_count += param.data.view(-1).shape[0]

        for i in range(mean.size(0)):
            mean_jacobian = zeros(param_count, mean.size(1))
            for j in range(mean.size(1)):
                mean[i, j].backward(retain_graph=True)
                grad = get_flat_grad_from(policy_net)
                mean_jacobian[:, j] += grad.data
                for param in policy_net.parameters():
                    param.grad = None
            res += mean_jacobian.mm(cov_inv.mm(mean_jacobian.t().mm(v.unsqueeze(1)))).squeeze()
        res /= mean.size(0)
        res[std_index: std_index + cov_inv.shape[0]] += 2 * v[std_index: std_index + cov_inv.shape[0]]
        return res + v * damping

    """directly compute Hessian*vector from KL"""
    def Fvp(v):
        kl = policy_net.get_kl(Variable(states))
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
        return flat_grad_grad_kl + v * damping

    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
    stepdir = conjugate_gradients(Fvp_fim, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp_fim(stepdir)))
    lm = math.sqrt(max_kl / shs)
    fullstep = stepdir * lm
    expected_improve = -loss_grad.dot(fullstep)

    prev_params = get_flat_params_from(policy_net)
    success, new_params = line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
    set_flat_params_to(policy_net, new_params)

    return success
