"""Training configuration of the cross-embodiment FB-CPR agent."""

from dataclasses import dataclass, field


def default_aux_reward_scales() -> dict[str, float]:
    """Return BFM-Zero's weights (paper Table 4) for the embodiment-independent penalties."""
    return {
        "limits_dof_pos": -10.0,
        "penalty_action_rate": -0.1,
        "penalty_slippage": -2.0,
        "penalty_undesired_contact": -1.0,
        "penalty_feet_ori": -0.4,
    }


@dataclass
class FbCprConfig:
    """Hyperparameters of ``DaGetFbCprAgent``; defaults are BFM-Zero's G1 values.

    Attributes:
        lr_f (float): Forward map learning rate.
        lr_b (float): Backward map learning rate.
        lr_actor (float): Actor learning rate.
        lr_discriminator (float): Discriminator learning rate.
        lr_critic (float): Discriminator-reward critic learning rate.
        lr_aux_critic (float): Auxiliary critic learning rate.
        weight_decay (float): Adam weight decay of F, B, actor and critics.
        clip_grad_norm (float): Gradient clipping of F, B and actor; 0 disables it.
        fb_target_tau (float): Soft-update rate of the F and B targets.
        critic_target_tau (float): Soft-update rate of the critic targets.
        ortho_coef (float): Weight of the orthonormality loss of B.
        train_goal_ratio (float): Fraction of training z encoded from replay states.
        expert_asm_ratio (float): Fraction of training z encoded from expert windows.
        relabel_ratio (float): Fraction of transitions relabeled with a new z.
        fb_pessimism_penalty (float): Ensemble pessimism in the FB target.
        actor_pessimism_penalty (float): Ensemble pessimism in the actor objective.
        critic_pessimism_penalty (float): Ensemble pessimism in the critic target.
        aux_critic_pessimism_penalty (float): Same for the auxiliary critic.
        stddev_clip (float): Clip of the policy noise when sampling actions.
        batch_size (int): Transitions per update; a multiple of the model's ``seq_length``.
        discount (float): Discount factor.
        reg_coeff (float): Weight of the discriminator critic in the actor loss.
        reg_coeff_aux (float): Weight of the auxiliary critic in the actor loss.
        scale_reg (bool): Scale the regularizers by ``|Q_fb|``.
        grad_penalty_discriminator (float): WGAN gradient-penalty weight; 0 disables it.
        within_robot_pairs (bool): Compare F and B only between states of the same robot.
            A state of another robot is unreachable, so comparing them would teach B to
            separate robots, and z could not be shared across them.
        train_encoder (bool): Tune the policy encoder during RL (off: keep it frozen).
        encoder_rl_gradients (bool): Let the F and critic losses reach the encoder. The
            actor loss never does, so that the actor cannot move the critics' inputs.
        lr_encoder (float): Policy encoder learning rate (not from BFM-Zero, which has none).
        lr_physics_heads (float): Learning rate of the physics pretraining heads.
        physics_loss_weight (float): Weight of the physics auxiliary losses on the encoder.
        use_mix_rollout (bool): Roll out with z from the training distribution.
        update_z_every_step (int): Control steps between rollout z changes.
        z_buffer_size (int): Stored training z per robot.
        rollout_expert_trajectories (bool): Some environments roll out expert-encoded z.
        rollout_expert_trajectories_length (int): Length of those rollouts in steps.
        rollout_expert_trajectories_percentage (float): Fraction of such environments.
        aux_reward_scales (dict[str, float]): Weight of each auxiliary reward.
    """

    lr_f: float = 3e-4
    lr_b: float = 1e-5
    lr_actor: float = 3e-4
    lr_discriminator: float = 1e-5
    lr_critic: float = 3e-4
    lr_aux_critic: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    critic_target_tau: float = 0.005
    ortho_coef: float = 100.0
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6
    relabel_ratio: float = 0.8
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    critic_pessimism_penalty: float = 0.5
    aux_critic_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    batch_size: int = 1024
    discount: float = 0.98
    reg_coeff: float = 0.05
    reg_coeff_aux: float = 0.02
    scale_reg: bool = True
    grad_penalty_discriminator: float = 10.0
    within_robot_pairs: bool = True
    train_encoder: bool = True
    encoder_rl_gradients: bool = True
    lr_encoder: float = 3e-5
    lr_physics_heads: float = 1e-4
    physics_loss_weight: float = 1.0
    use_mix_rollout: bool = True
    update_z_every_step: int = 100
    z_buffer_size: int = 8192
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    aux_reward_scales: dict[str, float] = field(default_factory=default_aux_reward_scales)

    def __post_init__(self) -> None:
        """Reject mixture ratios that cannot be sampled."""
        if self.train_goal_ratio < 0 or self.expert_asm_ratio < 0 or self.train_goal_ratio + self.expert_asm_ratio > 1:
            raise ValueError("train_goal_ratio and expert_asm_ratio must be non-negative and sum to at most 1")
        if not 0 <= self.relabel_ratio <= 1:
            raise ValueError("relabel_ratio must be in [0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
