import torch
from utils.math import *

from torch.distributions import MultivariateNormal, Normal

def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, clip_epsilon, l2_reg, scheduler_policy, scheduler_value):

    """update critic"""
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)

        optimizer_value.step()
        # Update LR
        scheduler_value.step()
    """update policy"""
    log_probs , action_mean, action_std = policy_net.get_log_prob(states, actions)
    
    # Calculate the entropy
    dist = Normal(action_mean, action_std)
    entropy = dist.entropy().mean()
    # Calculate the explained_variance
    # print(values_pred.squeeze().shape,returns.squeeze().shape)
    try:
        ev = explained_variance(values_pred.squeeze(),returns.squeeze())
    except:
        ev=np.nan


    ratio = torch.exp(log_probs - fixed_log_probs)
    ## Calculate clipfrac
    #clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
    # clipfrac = tf.reduce_mean(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
    clipfrac =  (torch.gt(torch.abs(ratio - 1), clip_epsilon)).float().mean().item()
    ## Calculate Approx KL
    # approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
    diff = log_probs - fixed_log_probs
    approxkl = .5 * torch.mean(torch.mul(diff,diff))
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    optimizer_policy.step()
    # Update LR
    scheduler_policy.step()

    return policy_surr.item(), value_loss.item(), ev, clipfrac, entropy.item(), approxkl.item()