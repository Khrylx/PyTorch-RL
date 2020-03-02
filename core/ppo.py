import torch
from utils.math import *

from torch.distributions import MultivariateNormal, Normal

def ppo_step_one_loss(policy_net, value_net, unique_optimizer, optim_value_iternum, states, actions,
             returns, advantages,values, fixed_log_probs, clip_epsilon, \
             l2_reg, ent_coef=0.01, vf_coef = 0.5, clip_value=False):


    OLDVPRED = values
    """update critic"""
    for _ in range(optim_value_iternum):
        if clip_value == True:
            values_pred = value_net(states)
            with torch.no_grad():
                # vpredclipped = OLDVPRED + torch.clamp(value_net(states) - OLDVPRED,min =  - clip_epsilon, max =  clip_epsilon)
                vpredclipped = OLDVPRED + torch.clamp(value_net(states) - OLDVPRED,min =  - 1, max =  1)

            vf_losses1 = (values_pred - returns).pow(2)
            # Clipped value
            vf_losses2 = (vpredclipped - returns).pow(2)
            value_loss = .5 * torch.max(vf_losses1, vf_losses2).mean()
        else:
            values_pred = value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()


    """update policy"""
    log_probs , action_mean, action_std = policy_net.get_log_prob(states, actions)
    
    # Calculate the entropy
    dist = Normal(action_mean, action_std)
    entropy = dist.entropy().mean()
    # Calculate the explained_variance

    ratio = torch.exp(log_probs - fixed_log_probs)

    ## Calculate clipfrac
    with torch.no_grad():
        try:
            ev = explained_variance(values_pred.squeeze(),returns.squeeze())
        except:
            ev=np.nan

        clipfrac =  (torch.gt(torch.abs(ratio - 1), clip_epsilon)).float().mean().item()
        ## Calculate Approx KL
        diff = log_probs - fixed_log_probs
        approxkl = .5 * torch.mean(torch.mul(diff,diff))
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Total loss
    loss = policy_loss - (entropy * ent_coef) + (value_loss * vf_coef)
    unique_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
    unique_optimizer.step()

    return policy_loss.item(), value_loss.item(), ev, clipfrac, entropy.item(), approxkl.item()


# Two loss 
def ppo_step_two_losses(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages,values, fixed_log_probs, clip_epsilon, l2_reg, ent_coef=0.01, clip_value=False):


    OLDVPRED = values
    """update critic"""
    for _ in range(optim_value_iternum):
        if clip_value == True:
            values_pred = value_net(states)
            with torch.no_grad():
                # vpredclipped = OLDVPRED + torch.clamp(value_net(states) - OLDVPRED,min =  - clip_epsilon, max =  clip_epsilon)
                vpredclipped = OLDVPRED + torch.clamp(value_net(states) - OLDVPRED,min =  - 1, max =  1)

            vf_losses1 = (values_pred - returns).pow(2)
            # Clipped value
            vf_losses2 = (vpredclipped - returns).pow(2)
            value_loss = .5 * torch.max(vf_losses1, vf_losses2).mean()
        else:
            values_pred = value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
        optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)

        optimizer_value.step()

    """update policy"""
    log_probs , action_mean, action_std = policy_net.get_log_prob(states, actions)
    
    # Calculate the entropy
    dist = Normal(action_mean, action_std)
    entropy = dist.entropy().mean()
    # Calculate the explained_variance
    ratio = torch.exp(log_probs - fixed_log_probs)

    ## Calculate clipfrac
    with torch.no_grad():
        try:
            ev = explained_variance(values_pred.squeeze(),returns.squeeze())
        except:
            ev=np.nan

        clipfrac =  (torch.gt(torch.abs(ratio - 1), clip_epsilon)).float().mean().item()
        ## Calculate Approx KL
        diff = log_probs - fixed_log_probs
        approxkl = .5 * torch.mean(torch.mul(diff,diff))
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    # policy_loss = -torch.min(surr1, surr2).mean()
    policy_loss = -torch.min(surr1, surr2).mean() - (entropy * ent_coef)


    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    optimizer_policy.step()

    return policy_loss.item(), value_loss.item(), ev, clipfrac, entropy.item(), approxkl.item()

    