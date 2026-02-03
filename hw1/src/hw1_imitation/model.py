"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        #create the model here
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h #then we change the in_dim to be the hidden dimension size
        layers.append(nn.Linear(in_dim, action_dim * chunk_size))
        self.model = nn.Sequential(*layers)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.model(state) #lets get the new state output
        pred = pred.view(-1, self.chunk_size, self.action_dim) #gets this back into shape of (batch, chunk dim, action dim)
        loss = torch.nn.functional.mse_loss(pred, action_chunk)
        return loss
        #create the loss here

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        actions = self.model(state)
        actions = actions.view(-1, self.chunk_size, self.action_dim)
        return actions #this should be in the shape of (batch, chunk, action)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        #lets create the actual model here
        layers = []
        in_dim = state_dim + action_dim * chunk_size + 1 #actually need to take in state_dim as well as action_dim and chunk size?
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h #then we change the in_dim to be the hidden dimension size
        layers.append(nn.Linear(in_dim, action_dim * chunk_size))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self, 
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        action_chunk = action_chunk.view(batch_size, -1) #this flattens it into batch_size x chunk_dim * acttion_dim
        action_noise = torch.randn_like(action_chunk)
        t = torch.rand(batch_size, 1, device=state.device)
        xt = action_noise * (1-t) + action_chunk * t
        predicted_velocity = self.model(torch.cat([state, xt, t], dim=1))
        vector_field = action_chunk - action_noise
        loss = torch.nn.functional.mse_loss(predicted_velocity, vector_field)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        #need to sample actions
        chunk_dim = self.action_dim * self.chunk_size
        batch_size = state.shape[0]
        action_noise = torch.randn(batch_size, chunk_dim, device=state.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=state.device)
            input = torch.cat([state, action_noise, t], dim=1)
            velocity_output = self.model(input)
            action_noise = action_noise + dt * velocity_output
        return action_noise.view(batch_size, self.chunk_size, self.action_dim)
        #do euler integration to get the actions:



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
