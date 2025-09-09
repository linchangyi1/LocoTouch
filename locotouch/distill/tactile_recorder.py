import torch


class TactileRecorder:
    def __init__(self, device, env_num, tactile_shape, min_delay=3, max_delay=7):
        self.device = device
        self.env_num = env_num
        self.tactile_shape = (tactile_shape,) if isinstance(tactile_shape, int) else tactile_shape
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.tactile_expand_shape = (-1, self.max_delay, *self.tactile_shape)
        self.tactile_buffer = torch.zeros((self.env_num, self.max_delay, *self.tactile_shape), dtype=torch.float32, device=self.device)
        self.first_signal_recorded = torch.ones((self.env_num,), dtype=torch.bool, device=self.device)
        self.delay_steps = torch.zeros((self.env_num,), dtype=torch.long, device=self.device)
        self.env_idx = torch.arange(env_num, device=self.device)
        self.reset()

    def reset(self, env_idx=None):
        env_idx = env_idx if env_idx is not None else self.env_idx
        self.tactile_buffer[env_idx] = 0.0
        self.first_signal_recorded[env_idx] = True
        self.delay_steps[env_idx] = torch.randint(low=self.min_delay, high=self.max_delay, size=(env_idx.shape), device=self.device)

    def record_new_tactile_signals(self, tactile_signals: torch.Tensor):
        self.tactile_buffer[:, 1:] = self.tactile_buffer[:, :-1].clone()
        self.tactile_buffer[:, 0] = tactile_signals.clone()
        
        if self.first_signal_recorded.any():
            first_signal_idx = self.first_signal_recorded.nonzero(as_tuple=False).flatten()
            self.tactile_buffer[first_signal_idx] = tactile_signals[first_signal_idx].unsqueeze(1).expand(self.tactile_expand_shape).clone()
            self.first_signal_recorded[first_signal_idx] = False

    def get_tactile_signals(self):
        return self.tactile_buffer[self.env_idx, self.delay_steps]

