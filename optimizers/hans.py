import torch

class HANS(torch.optim.Optimizer):
    """
    Hierarchical Adaptive Momentum with Nesterov Scaling (HANS) Optimizer
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta_fast: float = 0.9,
        beta_slow: float = 0.99,
        beta_adaptive: float = 0.999,
        alpha: float = 0.7,
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta_fast < 1.0:
            raise ValueError(f"Invalid beta_fast parameter: {beta_fast}")
        if not 0.0 <= beta_slow < 1.0:
            raise ValueError(f"Invalid beta_slow parameter: {beta_slow}")
        if not 0.0 <= beta_adaptive < 1.0:
            raise ValueError(f"Invalid beta_adaptive parameter: {beta_adaptive}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            beta_adaptive=beta_adaptive,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super(HANS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HANS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            m_fast_states = []
            m_slow_states = []
            v_states = []
            max_v_states = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.dtype in {torch.float16, torch.bfloat16}:
                        grads.append(p.grad.float())
                    else:
                        grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['m_fast'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['m_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    m_fast_states.append(state['m_fast'])
                    m_slow_states.append(state['m_slow'])
                    v_states.append(state['v'])
                    if group['amsgrad']:
                        max_v_states.append(state['max_v'])

                    state['step'] += 1
                    state_steps.append(state['step'])

            hans_update(
                params_with_grad,
                grads,
                m_fast_states,
                m_slow_states,
                v_states,
                max_v_states,
                state_steps,
                amsgrad=group['amsgrad'],
                beta_fast=group['beta_fast'],
                beta_slow=group['beta_slow'],
                beta_adaptive=group['beta_adaptive'],
                alpha=group['alpha'],
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps']
            )
        return loss


def hans_update(
    params,
    grads,
    m_fast_states,
    m_slow_states,
    v_states,
    max_v_states,
    state_steps,
    *,
    amsgrad: bool,
    beta_fast: float,
    beta_slow: float,
    beta_adaptive: float,
    alpha: float,
    lr: float,
    weight_decay: float,
    eps: float
):
    for i, param in enumerate(params):
        grad = grads[i]
        m_fast = m_fast_states[i]
        m_slow = m_slow_states[i]
        v = v_states[i]
        step = state_steps[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        m_fast.mul_(beta_fast).add_(grad, alpha=1 - beta_fast)
        m_slow.mul_(beta_slow).add_(grad, alpha=1 - beta_slow)

        bias_correction1_fast = 1 - beta_fast ** step
        bias_correction1_slow = 1 - beta_slow ** step
        m_fast_corrected = m_fast / bias_correction1_fast
        m_slow_corrected = m_slow / bias_correction1_slow
        m_combined = alpha * m_fast_corrected + (1 - alpha) * m_slow_corrected

        effective_grad = grad
        if step > 1:
            momentum_norm_sq = torch.sum(m_combined * m_combined)
            grad_norm_sq = torch.sum(grad * grad)
            if momentum_norm_sq > eps and grad_norm_sq > eps:
                momentum_grad_dot = torch.sum(m_combined * grad)
                momentum_norm = torch.sqrt(momentum_norm_sq)
                grad_norm = torch.sqrt(grad_norm_sq)
                cos_sim = momentum_grad_dot / (momentum_norm * grad_norm + eps)
                if cos_sim > 0.1:
                    nesterov_factor = 1.0 + 0.1 * cos_sim
                    effective_grad = nesterov_factor * grad

        v.mul_(beta_adaptive).addcmul_(effective_grad, effective_grad, value=1 - beta_adaptive)
        bias_correction2 = 1 - beta_adaptive ** step
        v_corrected = v / bias_correction2

        if amsgrad:
            max_v = max_v_states[i]
            torch.maximum(max_v, v_corrected, out=max_v)
            denom = max_v.sqrt().add_(eps)
        else:
            denom = v_corrected.sqrt().add_(eps)

        param.addcdiv_(m_combined, denom, value=-lr)
