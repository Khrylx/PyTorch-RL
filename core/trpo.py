import numpy as np
from torch.autograd import Variable
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


def trpo_step(policy_net, value_net, optimizer_value, states, actions,
              returns, advantages, max_kl, damping, l2_reg):

    """update critic"""
    values_target = Variable(returns)
    values_pred = value_net(Variable(states))
    value_loss = (values_pred - values_target).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data
    """define the loss function for TRPO"""
    def get_loss(volatile=False):
        log_probs = policy_net.get_log_prob(Variable(states, volatile=volatile), Variable(actions))
        action_loss = -Variable(advantages) * torch.exp(log_probs - Variable(fixed_log_probs))
        return action_loss.mean()

    """define Hessian*vector for KL"""
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
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
    lm = math.sqrt(max_kl / shs)
    fullstep = stepdir * lm
    expected_improve = -loss_grad.dot(fullstep)

    prev_params = get_flat_params_from(policy_net)
    success, new_params = line_search(policy_net, get_loss, prev_params, fullstep,expected_improve)
    set_flat_params_to(policy_net, new_params)

    return success
