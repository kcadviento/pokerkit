import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

class StrategyNetwork(nn.Module):
    """Neural network for strategy prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(StrategyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class AdvantageNetwork(nn.Module):
    """Neural network for advantage prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(AdvantageNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DeepCFRAgent:
    """Deep Counterfactual Regret Minimization (Deep CFR) Agent for poker."""
    
    def __init__(self, name: str = "Deep-CFR-Agent", num_actions: int = 5):
        self.name = name
        self.num_actions = num_actions
        
        # Neural networks for value and strategy
        self.value_network = self._build_value_network()
        self.strategy_network = self._build_strategy_network()
        
        # Optimizers
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)
        self.strategy_optimizer = optim.Adam(self.strategy_network.parameters(), lr=0.001)
        
        # Memory buffers
        self.value_memory = []
        self.strategy_memory = []
        
        # Training parameters
        self.batch_size = 32
        self.iterations = 0
        self.regrets = defaultdict(lambda: np.zeros(num_actions))
        self.strategy_sum = defaultdict(lambda: np.zeros(num_actions))
    
    def _build_value_network(self) -> nn.Module:
        """Build the value network for estimating counterfactual values."""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
    
    def _build_strategy_network(self) -> nn.Module:
        """Build the strategy network for action probabilities."""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Softmax(dim=-1)
        )
    
    def get_info_key(self, obs: Dict, player_idx: int) -> str:
        """Convert an observation to a string key for the info set."""
        # Get player's hole cards
        player_states = obs.get("player_states", [])
        hole_cards = []
        if player_idx < len(player_states):
            hole_cards = sorted(player_states[player_idx].get("hole_cards", []))
        
        # Get community cards
        community_cards = sorted(obs.get("community_cards", []))
        
        # Get game phase and pot
        phase = obs.get("game_phase", "preflop")
        pot = obs.get("pot", 0)
        
        # Get player's stack and position
        stack = 0
        position = 0
        if player_idx < len(player_states):
            stack = player_states[player_idx].get("stack", 0)
            position = (player_idx - obs.get("dealer_position", 0)) % len(player_states)
        
        # Combine information into a key
        key = f"{hole_cards}|{community_cards}|{phase}|{pot}|{stack}|{position}"
        return key
    
    def _encode_state(self, obs: Dict, player_idx: int) -> torch.Tensor:
        """Encode the game state into a neural network input."""
        # This is a placeholder - implement proper state encoding
        return torch.zeros(128)
    
    def get_strategy(self, obs: Dict, valid_actions: List[int], player_idx: int) -> np.ndarray:
        """Get the current strategy for an information set."""
        # Encode state
        state = self._encode_state(obs, player_idx)
        
        # Get strategy from network
        with torch.no_grad():
            strategy = self.strategy_network(state)
        
        # Convert to numpy and mask invalid actions
        strategy = strategy.numpy()
        action_mask = np.zeros(self.num_actions)
        for action in valid_actions:
            action_mask[action] = 1.0
        
        strategy = strategy * action_mask
        strategy = strategy / np.sum(strategy)  # Renormalize
        
        return strategy
    
    def update_strategy_sum(self, obs: Dict, strategy: np.ndarray, player_idx: int):
        """Update the cumulative strategy for average strategy computation."""
        info_key = self.get_info_key(obs, player_idx)
        self.strategy_sum[info_key] += strategy
        self.iterations += 1
    
    def get_average_strategy(self, obs: Dict, valid_actions: List[int], player_idx: int) -> np.ndarray:
        """Get the average strategy across all iterations."""
        info_key = self.get_info_key(obs, player_idx)
        if info_key not in self.strategy_sum:
            return np.ones(self.num_actions) / self.num_actions
        
        strategy_sum = self.strategy_sum[info_key]
        total_sum = np.sum(strategy_sum)
        
        if total_sum > 0:
            return strategy_sum / total_sum
        else:
            return np.ones(self.num_actions) / self.num_actions
    
    def train_value_network(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Train the value network on a batch of experiences."""
        if not batch:
            return
        
        states, targets = zip(*batch)
        states = torch.stack(states)
        targets = torch.stack(targets)
        
        self.value_optimizer.zero_grad()
        predictions = self.value_network(states)
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        self.value_optimizer.step()
    
    def train_strategy_network(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Train the strategy network on a batch of experiences."""
        if not batch:
            return
        
        states, targets = zip(*batch)
        states = torch.stack(states)
        targets = torch.stack(targets)
        
        self.strategy_optimizer.zero_grad()
        predictions = self.strategy_network(states)
        loss = nn.KLDivLoss()(predictions.log(), targets)
        loss.backward()
        self.strategy_optimizer.step()
    
    def act(self, obs: Dict, valid_actions: List[int], player_idx: int = 0) -> int:
        """Choose an action based on the current strategy."""
        strategy = self.get_strategy(obs, valid_actions, player_idx)
        
        # Update the strategy sum for average strategy computation
        self.update_strategy_sum(obs, strategy, player_idx)
        
        # Choose an action based on the strategy
        valid_probs = np.array([strategy[a] for a in valid_actions])
        valid_probs = valid_probs / np.sum(valid_probs)  # Renormalize
        
        chosen_idx = np.random.choice(len(valid_actions), p=valid_probs)
        return valid_actions[chosen_idx]
    
    def train(self, env, num_iterations: int = 1000):
        """Train the Deep CFR agent through self-play."""
        for iteration in tqdm(range(num_iterations)):
            # Reset the environment
            obs = env.reset()
            done = False
            
            # Initialize reach probabilities
            reach_probs = {i: 1.0 for i in range(env.num_players)}
            
            while not done:
                current_player = obs.get("current_player", 0)
                valid_actions = env.action_wrapper.get_valid_actions(obs)
                
                if not valid_actions:
                    break
                
                # Get strategy and choose action
                strategy = self.get_strategy(obs, valid_actions, current_player)
                action_probs = np.array([strategy[a] for a in valid_actions])
                action_probs = action_probs / np.sum(action_probs)
                action_idx = np.random.choice(len(valid_actions), p=action_probs)
                action = valid_actions[action_idx]
                
                # Update reach probabilities
                for i in range(env.num_players):
                    if i != current_player:
                        reach_probs[i] *= strategy[action]
                
                # Take action and get next state
                next_obs, rewards, done, info = env.step(action)
                
                # Store experience for value network
                state = self._encode_state(obs, current_player)
                target = torch.tensor(rewards[current_player], dtype=torch.float32)
                self.value_memory.append((state, target))
                
                # Store experience for strategy network
                strategy_target = torch.tensor(strategy, dtype=torch.float32)
                self.strategy_memory.append((state, strategy_target))
                
                obs = next_obs
            
            # Train networks on collected experiences
            if len(self.value_memory) >= self.batch_size:
                value_batch = random.sample(self.value_memory, self.batch_size)
                self.train_value_network(value_batch)
            
            if len(self.strategy_memory) >= self.batch_size:
                strategy_batch = random.sample(self.strategy_memory, self.batch_size)
                self.train_strategy_network(strategy_batch)
            
            # Clear memory if it gets too large
            if len(self.value_memory) > 10000:
                self.value_memory = self.value_memory[-10000:]
            if len(self.strategy_memory) > 10000:
                self.strategy_memory = self.strategy_memory[-10000:]
    
    def save(self, filepath: str):
        """Save the agent's networks and strategy to a file."""
        torch.save({
            'value_network_state_dict': self.value_network.state_dict(),
            'strategy_network_state_dict': self.strategy_network.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'strategy_optimizer_state_dict': self.strategy_optimizer.state_dict(),
            'strategy_sum': dict(self.strategy_sum),
            'iterations': self.iterations
        }, filepath)
    
    def load(self, filepath: str):
        """Load the agent's networks and strategy from a file."""
        checkpoint = torch.load(filepath)
        
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.strategy_network.load_state_dict(checkpoint['strategy_network_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer_state_dict'])
        
        self.strategy_sum = defaultdict(lambda: np.zeros(self.num_actions))
        for key, value in checkpoint['strategy_sum'].items():
            self.strategy_sum[key] = value
            
        self.iterations = checkpoint['iterations'] 