# https://github.com/optimizedlearning/mechanic
import logging
from typing import Tuple, Any, Callable, Dict
import torch


logger = logging.getLogger(__name__)


def _init_state(
        optimizer: torch.optim.Optimizer,
        p_ref: Dict[torch.Tensor, torch.Tensor],
        s_decay: float,
        betas: Tuple[float],
        s_init: float,
        eps: float,
        store_delta: bool,
        log_every: int,
        force=False):
    '''
    initialized extra state for mechanic.

    Args:
        optimizer: optimizer instance to initialize extra state for.
        p_ref: mapping of parameters to their initial values at the start of optimization.
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets or recompute them on-the-fly.
        log_every: how often to log scale values.
        force: if True, reinitialize the state.
    '''
    if force or '_mechanic' not in optimizer.state:
        optimizer.state['_mechanic'] = {
            's_decay': torch.tensor(s_decay),
            'betas': torch.tensor(betas),
            's_init': torch.tensor(s_init),
            'eps': eps,
            's': torch.zeros(len(betas)),
            'p_ref': {},
            'sum_squared_products': torch.zeros(len(betas)),
            'reward': torch.zeros(len(betas)),
            'max_product': torch.full((len(betas),), 1e-6),
            'iter_count': 0,
            'log_every': log_every,
        }
        _init_reference(optimizer, p_ref, store_delta)

def _init_reference(
        optimizer: torch.optim.Optimizer,
        p_ref: Dict[torch.Tensor, torch.Tensor],
        store_delta: bool):
    '''
    Stores the starting point of the optimization (the "reference").

    Args:
        optimizer: optimizer instance to store reference for.
        p_ref: mapping of parameters to their initial values at the start of optimization.
        store_delta: if true, we should also store the "Delta" value: the
            displacement between the current iterate and the reference.
    '''
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state['_mechanic'][p] = {
                'ref': p_ref[p].clone(),
            }
            if store_delta:
                optimizer.state['_mechanic'][p]['delta'] = torch.zeros_like(p)

def _step(
        optimizer: torch.optim.Optimizer,
        base_step: Callable,
        s_decay: float,
        betas: Tuple[float],
        s_init: float,
        eps: float,
        store_delta: bool=True,
        log_every: int=0,
        closure: Callable=None):
    '''
    runs one step of mechanic.

    Args:
        optimizer: mechanic optimizer instance that we are computing the step for.
        base_step: The "step" function of the base optimizer (e.g. SGD, AdamW etc).
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets between current iterate and reference
            or recompute them on-the-fly.
        force: if True, reinitialize the state.
    Returns:
        loss value
    '''

    prev_grad = torch.is_grad_enabled()

    # we don't wrap the entire function in @torch.no_grad because
    # we want to let the base optimizer differentiate things
    # if it so desires.
    torch.set_grad_enabled(False)


    if closure is not None:
        # if we need to rely on closure to generate gradients
        # then we generate gradient here, but also need to let the
        # base algorithm potentially reevaluate the closure as much
        # as it likes without doubling the gradients the first time it does so.
        # So, we will create a "fake" closure called skip_once_closure
        # to be eventually provided to base_step.
        loss = closure()
        eval_count = 0

        # lie to the base algorithm about first closure eval so that if
        # it thinks that the closure has been evaluated N times at the
        # end of its update, then it will be correct.
        # I'm not sure if this is actually important - might be reasonable
        # to just not do this fake closure stuff.
        def skip_once_closure():
            nonlocal eval_count
            eval_count += 1
            if eval_count == 1:
                return loss
            return closure()
    else:
        skip_once_closure = None

    updates = {}
    grads = {}
    deltas = {}

    global_norm = 0.0
    grad_norm = 0.0

    # store gradients and current parameter values.
    # We need to store the gradients because the base optimizer might
    # change them (for example by adding a weight-decay term).
    # We need to store the current parameter values so that we can
    # compute the "update" generated by the base optimizer by subtracting
    # the "new" values from the current values.
    for group in optimizer.param_groups:
        for p in group['params']:

            if p.grad is None:
                grads[p] = None
            else:
                grads[p] = p.grad.clone()
            updates[p] = p.data.clone()

    # Re-enable gradients and run the base optimizer step
    torch.set_grad_enabled(prev_grad)
    loss = base_step(skip_once_closure)
    torch.set_grad_enabled(False)

    # init state after base_step in case base_step only initializes its
    # own state if self.state is empty.
    # Here, we use the fact that updates[p] is the original value of p before the base step
    _init_state(optimizer, updates, s_decay, betas, s_init, eps, store_delta, log_every)
    mechanic_state = optimizer.state['_mechanic']


    # compute updates and global norms.
    for group in optimizer.param_groups:
        for p in group['params']:
            if grads[p] is None:
                continue

            p_ref = mechanic_state[p]['ref']
            if store_delta:
                deltas[p] = mechanic_state[p]['delta']
            else:
                # Again, we use updates[p] is the original value of p before base_step
                deltas[p] = (updates[p] - p_ref)/(torch.sum(mechanic_state['s']) + mechanic_state['eps'])

            updates[p].copy_(p-updates[p])
            p_flat = p.flatten()
            global_norm += torch.dot(p_flat, p_flat)

            g_flat = grads[p].flatten()
            grad_norm += torch.dot(g_flat, g_flat)



    global_norm = torch.sqrt(global_norm)
    grad_norm = torch.sqrt(grad_norm)
    inner_product = 0.0

    # compute inner_product (h in paper pseudocode)
    for group in optimizer.param_groups:
        for p in group['params']:

            if grads[p] is None:
                continue

            grad = grads[p]

            delta = deltas[p]

            decay = mechanic_state['s_decay'] * p.flatten() \
                * torch.sum(mechanic_state['s']) * grad_norm / (global_norm + mechanic_state['eps'])

            inner_product += torch.dot(
                delta.flatten(),
                grad.flatten() + decay.flatten())

            delta.add_(updates[p])

    device = inner_product.device

    for key in mechanic_state:
        try:
            if mechanic_state[key].device != device:
                mechanic_state[key] = mechanic_state[key].to(device)
        except:
            pass


    # Run the "tuner" step of Mechanic to compute the new s values.
    s = mechanic_state['s']
    s_decay = mechanic_state['s_decay']                             # called "lambda" in paper
    s_init = mechanic_state['s_init']
    betas = mechanic_state['betas']
    eps = mechanic_state['eps']
    max_product = mechanic_state['max_product']                     # called "m" in paper
    reward = mechanic_state['reward']                               # called "r" in paper
    sum_squared_products = mechanic_state['sum_squared_products']   # called "v" in paper

    mechanic_state['iter_count'] += 1
    log_every = mechanic_state['log_every']


    max_product.copy_(torch.maximum(
        (betas * max_product), torch.abs(inner_product)))

    sum_squared_products.mul_(
        betas**2).add_(torch.square(inner_product))
    reward.mul_(betas).sub_(s * inner_product)
    reward.copy_(torch.clamp(reward, min=torch.zeros_like(reward)))

    wealth = max_product * s_init / len(betas) + reward

    s.copy_(wealth / (torch.sqrt(sum_squared_products) + eps))


    if log_every > 0 and mechanic_state['iter_count']%log_every == 0:
        logging.info(f"(k={mechanic_state['iter_count']}), s_sum (global scaling): {torch.sum(s).item()}")

    for group in optimizer.param_groups:
        for p in group['params']:

            if grads[p] is None:
                continue

            p_ref = mechanic_state[p]['ref']
            delta = deltas[p]
            p.copy_(p_ref + delta * max(torch.sum(s), 0.0))

    torch.set_grad_enabled(prev_grad)

    return loss


# Empty class used so that we can do isinstance(mechanize(SGD), Mechanic)
class Mechanic:
    pass

def is_mechanized(opt):
    return isinstance(opt, Mechanic)

def mechanize(
        Base: Any,
        s_decay: float = 0.01,
        betas: Tuple[float] = (0.9, 0.99, 0.999, 0.9999,
                               0.99999, 0.999999),
        s_init: float = 1e-8,
        eps: float = 1e-8,
        store_delta: bool = False,
        log_every: int = 0):
    '''
    Wrap a base optimizer class in a mechanic tuner. The mechanized optimizer
    is a subclass of the base optimizer class in order to minimize disruption
    to subsequent code.

    Args:
        Base: base optimizer class to convert into a mechanic instance (e.g. torch.optim.SGD)
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets or recompute them on-the-fly.
        log_every: how often (in steps) to log the scale values computed by mechanic.
    
    Returns: a new class Mechanized that tunes the base class.

    For example, instead of

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    you could do:

    optimizer = mechanize(torch.optim.SGD)(model.parameters(), lr=0.01)

    The rest of your code should ideally not need to change, even if it accesses
    the internal state of optimizer under the assumption that it is an unadulterated
    instance of torch.optim.SGD.

    Note that this may not always hold: certain libraries like DeepSpeed seem to make
    significant enough assumptions about how the optimizer will work that they may do
    incorrect things.
    '''

    class Mechanized(Base, Mechanic):
        '''
        Wraps a base algorithm as a Mechanic instance.
        '''

        def step(self, closure=None):

            return _step(self, super().step, s_decay, betas, s_init, eps, store_delta, log_every, closure)

    Mechanized.__name__ += Base.__name__

    return Mechanized
