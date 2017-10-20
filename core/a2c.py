import torch
from torch.autograd import Variable


def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):

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

    """update policy"""
    log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
    policy_loss = -(log_probs * Variable(advantages)).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    optimizer_policy.step()
