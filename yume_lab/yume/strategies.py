"""
Latent space exploration strategies.
Different methods for navigating the seed/latent space.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
import itertools


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""
    
    @abstractmethod
    def next_seed(self, iteration: int) -> int:
        """Generate next seed based on strategy."""
        pass
    
    @abstractmethod
    def next_prompt_variation(self, base_prompt: str, iteration: int) -> str:
        """Generate next prompt variation."""
        pass


class RandomStrategy(ExplorationStrategy):
    """
    Random exploration - uniform sampling.
    Good for: Broad coverage, no assumptions.
    """
    
    def __init__(self, seed_range: Tuple[int, int] = (0, 2**31 - 1)):
        self.seed_range = seed_range
        self.prompt_modifiers = [
            "dramatic lighting", "soft lighting", "golden hour",
            "cinematic", "highly detailed", "ethereal",
            "warm tones", "cool tones", "vibrant colors",
            "misty", "foggy", "hazy", "atmospheric",
            "sharp focus", "shallow depth of field", "bokeh",
            "film grain", "vintage", "modern",
        ]
    
    def next_seed(self, iteration: int) -> int:
        return np.random.randint(*self.seed_range)
    
    def next_prompt_variation(self, base_prompt: str, iteration: int) -> str:
        # Random number of modifiers (0-3)
        num_mods = np.random.randint(0, 4)
        if num_mods == 0:
            return base_prompt
        
        mods = np.random.choice(self.prompt_modifiers, num_mods, replace=False)
        return f"{base_prompt}, {', '.join(mods)}"


class LinearWalkStrategy(ExplorationStrategy):
    """
    Linear walk through seed space.
    Good for: Smooth transitions, animations.
    """
    
    def __init__(self, start_seed: int = 0, step_size: int = 1000):
        self.start_seed = start_seed
        self.step_size = step_size
        self.prompt_modifiers = [
            "dramatic lighting", "soft lighting", "golden hour",
            "cinematic", "atmospheric", "ethereal",
        ]
    
    def next_seed(self, iteration: int) -> int:
        return (self.start_seed + iteration * self.step_size) % (2**31)
    
    def next_prompt_variation(self, base_prompt: str, iteration: int) -> str:
        # Cycle through modifiers
        mod_idx = iteration % len(self.prompt_modifiers)
        return f"{base_prompt}, {self.prompt_modifiers[mod_idx]}"


class GridStrategy(ExplorationStrategy):
    """
    Grid-based exploration.
    Good for: Systematic coverage, reproducibility.
    """
    
    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.prompt_modifiers = [
            "dramatic lighting", "soft lighting",
            "warm tones", "cool tones",
            "highly detailed", "minimalist",
        ]
    
    def next_seed(self, iteration: int) -> int:
        x = iteration % self.grid_size
        y = iteration // self.grid_size
        return x * 1000000 + y
    
    def next_prompt_variation(self, base_prompt: str, iteration: int) -> str:
        # Grid position determines modifiers
        x = iteration % self.grid_size
        y = iteration // self.grid_size
        
        # Map grid position to modifier combinations
        mod_x = self.prompt_modifiers[x % len(self.prompt_modifiers)]
        mod_y = self.prompt_modifiers[y % len(self.prompt_modifiers)]
        
        if mod_x == mod_y:
            return f"{base_prompt}, {mod_x}"
        else:
            return f"{base_prompt}, {mod_x}, {mod_y}"


class EvolutionaryStrategy(ExplorationStrategy):
    """
    Evolutionary/genetic algorithm approach.
    Good for: Finding optimal regions, refinement.
    
    Maintains population of high-scoring seeds,
    generates children via mutation/crossover.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Population: list of (seed, score)
        self.population: List[Tuple[int, float]] = []
        
        self.prompt_modifiers = [
            "dramatic lighting", "soft lighting", "golden hour",
            "cinematic", "highly detailed", "ethereal",
        ]
    
    def update_population(self, seed: int, score: float):
        """Add/update population with new scored candidate."""
        self.population.append((seed, score))
        
        # Keep only top N
        self.population.sort(key=lambda x: x[1], reverse=True)
        self.population = self.population[:self.population_size]
    
    def next_seed(self, iteration: int) -> int:
        if len(self.population) < 2:
            # Not enough population, use random
            return np.random.randint(0, 2**31)
        
        # Evolutionary operators
        if np.random.random() < self.crossover_rate:
            # Crossover: combine two parent seeds
            parent1, _ = self.population[np.random.randint(0, len(self.population))]
            parent2, _ = self.population[np.random.randint(0, len(self.population))]
            
            # Simple crossover: average seeds
            child = (parent1 + parent2) // 2
        else:
            # Mutation: perturb existing seed
            parent, _ = self.population[np.random.randint(0, len(self.population))]
            mutation = np.random.randint(-10000, 10000)
            child = (parent + mutation) % (2**31)
        
        return child
    
    def next_prompt_variation(self, base_prompt: str, iteration: int) -> str:
        # Evolve prompts too
        num_mods = min(3, len(self.population) // 5)
        if num_mods == 0:
            return base_prompt
        
        mods = np.random.choice(self.prompt_modifiers, num_mods, replace=False)
        return f"{base_prompt}, {', '.join(mods)}"


class TemperatureScheduleStrategy(ExplorationStrategy):
    """
    Temperature-based exploration (simulated annealing).
    Good for: Finding good regions then refining.
    
    Starts with high exploration (wild), gradually focuses.
    """
    
    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        decay_rate: float = 0.9999,
    ):
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.final_temp = final_temp
        self.decay_rate = decay_rate
        
        self.best_seed = None
        self.best_score = 0.0
        
        self.prompt_modifiers = [
            "dramatic lighting", "soft lighting", "golden hour",
            "cinematic", "highly detailed", "ethereal",
            "warm tones", "cool tones", "vibrant colors",
        ]
    
    def update_best(self, seed: int, score: float):
        """Update best candidate."""
        if score > self.best_score:
            self.best_seed = seed
            self.best_score = score
    
    def next_seed(self, iteration: int) -> int:
        # Decay temperature
        self.current_temp = max(
            self.final_temp,
            self.current_temp * self.decay_rate
        )
        
        if self.best_seed is None:
            # No best yet, use random
            return np.random.randint(0, 2**31)
        
        # Perturb best seed, scaled by temperature
        perturbation = int(np.random.normal(0, 100000 * self.current_temp))
        return (self.best_seed + perturbation) % (2**31)
    
    def next_prompt_variation(self, base_prompt: str, iteration: int) -> str:
        # Number of modifiers scales with temperature
        max_mods = int(3 * self.current_temp) + 1
        num_mods = np.random.randint(0, max_mods)
        
        if num_mods == 0:
            return base_prompt
        
        mods = np.random.choice(
            self.prompt_modifiers, 
            min(num_mods, len(self.prompt_modifiers)), 
            replace=False
        )
        return f"{base_prompt}, {', '.join(mods)}"


class ClusterStrategy(ExplorationStrategy):
    """
    Cluster-based exploration.
    Good for: Finding multiple good regions.
    
    Maintains multiple clusters of high-scoring seeds,
    explores around each cluster center.
    """
    
    def __init__(self, num_clusters: int = 5, cluster_radius: int = 50000):
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius
        
        # Cluster centers: list of (seed, score, members)
        self.clusters: List[Tuple[int, float, List[int]]] = []
        
        self.prompt_modifiers = [
            "dramatic lighting", "soft lighting", "golden hour",
            "cinematic", "highly detailed",
        ]
    
    def update_clusters(self, seed: int, score: float):
        """Add seed to nearest cluster or create new one."""
        if len(self.clusters) == 0:
            # First cluster
            self.clusters.append((seed, score, [seed]))
            return
        
        # Find nearest cluster
        distances = [abs(seed - cluster[0]) for cluster in self.clusters]
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        
        if nearest_dist < self.cluster_radius:
            # Add to existing cluster
            center, center_score, members = self.clusters[nearest_idx]
            members.append(seed)
            
            # Update center to average of top members
            top_members = sorted(members, key=lambda s: s)[:10]
            new_center = int(np.mean(top_members))
            
            self.clusters[nearest_idx] = (new_center, max(center_score, score), members)
        else:
            # Create new cluster
            if len(self.clusters) < self.num_clusters:
                self.clusters.append((seed, score, [seed]))
            else:
                # Replace worst cluster
                self.clusters.sort(key=lambda x: x[1])
                self.clusters[0] = (seed, score, [seed])
    
    def next_seed(self, iteration: int) -> int:
        if len(self.clusters) == 0:
            return np.random.randint(0, 2**31)
        
        # Pick random cluster, perturb its center
        cluster_center, _, _ = self.clusters[np.random.randint(0, len(self.clusters))]
        perturbation = np.random.randint(-self.cluster_radius, self.cluster_radius)
        return (cluster_center + perturbation) % (2**31)
    
    def next_prompt_variation(self, base_prompt: str, iteration: int) -> str:
        num_mods = np.random.randint(1, 3)
        mods = np.random.choice(self.prompt_modifiers, num_mods, replace=False)
        return f"{base_prompt}, {', '.join(mods)}"


def get_strategy(name: str, **kwargs) -> ExplorationStrategy:
    """Factory function to get strategy by name."""
    strategies = {
        'random': RandomStrategy,
        'linear_walk': LinearWalkStrategy,
        'grid': GridStrategy,
        'evolutionary': EvolutionaryStrategy,
        'temperature': TemperatureScheduleStrategy,
        'cluster': ClusterStrategy,
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Choose from {list(strategies.keys())}")
    
    return strategies[name](**kwargs)