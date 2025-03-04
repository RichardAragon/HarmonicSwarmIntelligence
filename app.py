import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import random
from enum import Enum, auto
import time

class SensoryType(Enum):
    """Different types of sensory capabilities for agents"""
    FREQUENCY_ANALYZER = auto()
    WAVE_INTERFERENCE = auto()
    TOPOLOGICAL_DISTORTION = auto()
    GEOMETRIC_MEASUREMENT = auto()
    
class Agent:
    """An agent with unique sensory capabilities exploring a hyperdimensional structure"""
    
    def __init__(self, agent_id, sensory_type, position, dimension=4):
        self.id = agent_id
        self.sensory_type = sensory_type
        self.position = np.array(position)
        self.dimension = dimension
        self.confidence = 0.5  # Initial confidence in measurements
        self.history = []  # History of readings
        self.knowledge = np.zeros((10, 10, 10))  # Local knowledge representation (simplified)
        
    def ping(self, structure, noise_level=0.1):
        """Send a ping into the structure and measure the response"""
        distance = np.linalg.norm(self.position - structure.center)
        
        # Base signal strength diminishes with distance
        base_signal = max(0, 1 - (distance / structure.radius))
        
        # Different sensing types interact differently with the structure
        if self.sensory_type == SensoryType.FREQUENCY_ANALYZER:
            # Detects frequency components of the structure
            frequency_match = np.cos(distance * structure.frequency)
            response = base_signal * frequency_match
            
        elif self.sensory_type == SensoryType.WAVE_INTERFERENCE:
            # Measures interference patterns
            interference = np.sin(distance * structure.density)
            response = base_signal * interference
            
        elif self.sensory_type == SensoryType.TOPOLOGICAL_DISTORTION:
            # Senses topological features
            distortion = structure.complexity * np.exp(-distance / structure.radius)
            response = base_signal * distortion
            
        elif self.sensory_type == SensoryType.GEOMETRIC_MEASUREMENT:
            # Direct geometric measurements
            structural_feature = structure.measure_at_point(self.position)
            response = base_signal * structural_feature
        
        # Add some noise to the measurement
        response += np.random.normal(0, noise_level)
        
        # Update confidence based on consistency of readings
        if len(self.history) > 0:
            consistency = 1 - min(1, abs(response - np.mean(self.history)) / max(1e-5, np.std(self.history) if len(self.history) > 1 else 1))
            self.confidence = 0.8 * self.confidence + 0.2 * consistency
        
        self.history.append(response)
        
        # Update local knowledge representation
        x, y, z = np.clip((self.position[:3] + 5).astype(int), 0, 9)
        self.knowledge[x, y, z] = response
        
        return {
            'agent_id': self.id,
            'sensory_type': self.sensory_type,
            'position': self.position,
            'reading': response,
            'confidence': self.confidence
        }
    
    def move(self, direction, structure_bounds=5.0):
        """Move the agent in a given direction"""
        # Normalize direction
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        # Apply movement
        self.position += direction
        
        # Keep within bounds
        self.position = np.clip(self.position, -structure_bounds, structure_bounds)
        
        return self.position
    
    def recalibrate(self):
        """Adjust sensing parameters when readings are inconsistent"""
        self.confidence = max(0.2, self.confidence * 0.8)
        self.history = self.history[-5:] if len(self.history) > 5 else self.history

class HyperdimensionalStructure:
    """A simplified representation of a hyperdimensional structure"""
    
    def __init__(self, dimension=4, complexity=0.7, frequency=2.0, density=3.0):
        self.dimension = dimension
        self.center = np.zeros(dimension)
        self.radius = 4.0
        self.complexity = complexity  # How intricate the structure is
        self.frequency = frequency    # Base resonant frequency
        self.density = density        # Material density affecting wave propagation
        
        # Generate some random features for the structure
        self.features = []
        for _ in range(10):
            position = np.random.uniform(-self.radius, self.radius, size=dimension)
            strength = np.random.uniform(0.5, 1.0)
            self.features.append((position, strength))
    
    def measure_at_point(self, point):
        """Measure a structural feature at a specific point"""
        # Calculate influence of each feature
        total_influence = 0
        for position, strength in self.features:
            distance = np.linalg.norm(point[:len(position)] - position)
            if distance < self.radius:
                influence = strength * (1 - distance/self.radius)
                total_influence += influence
        
        return min(1.0, total_influence)

class SwarmQueen:
    """Central intelligence that integrates sensory inputs from all agents"""
    
    def __init__(self, dimension=4):
        self.dimension = dimension
        self.global_model = np.zeros((20, 20, 20))  # Simplified 3D representation of higher-dimensional space
        self.agent_weights = {}  # How much to trust each agent
        self.signal_history = {}  # Track signals over time
        self.convergence = 0.0  # How well the model has converged
    
    def process_signal(self, signal_data):
        """Process incoming signals from agents"""
        agent_id = signal_data['agent_id']
        reading = signal_data['reading']
        position = signal_data['position']
        confidence = signal_data['confidence']
        
        # Initialize tracking for new agents
        if agent_id not in self.agent_weights:
            self.agent_weights[agent_id] = 0.5
            self.signal_history[agent_id] = []
            
        # Track signal history
        self.signal_history[agent_id].append(reading)
        
        # Limit history length
        if len(self.signal_history[agent_id]) > 20:
            self.signal_history[agent_id] = self.signal_history[agent_id][-20:]
            
        # Adjust agent weight based on signal consistency and confidence
        if len(self.signal_history[agent_id]) > 3:
            signal_std = np.std(self.signal_history[agent_id])
            consistency = 1 / (1 + signal_std)  # Higher consistency if lower standard deviation
            self.agent_weights[agent_id] = 0.7 * self.agent_weights[agent_id] + 0.3 * consistency * confidence
        
        # Update global model - convert position to indices for our simplified 3D representation
        x, y, z = np.clip((position[:3] + 10).astype(int), 0, 19)
        current_value = self.global_model[x, y, z]
        weight = self.agent_weights[agent_id]
        
        # Weighted average update
        self.global_model[x, y, z] = (1 - weight) * current_value + weight * reading
        
        # Calculate model convergence
        non_zero = self.global_model > 0
        if np.sum(non_zero) > 0:
            model_variance = np.var(self.global_model[non_zero])
            self.convergence = 1 / (1 + model_variance)
        
        return {
            'agent_id': agent_id,
            'weight': self.agent_weights[agent_id],
            'model_update_strength': weight * abs(reading),
            'convergence': self.convergence
        }
    
    def suggest_movement(self, agent, other_agents):
        """Suggest where an agent should move next based on current understanding"""
        # Get current position
        position = agent.position.copy()
        
        # Calculate gradient in the model to find interesting areas
        x, y, z = np.clip((position[:3] + 10).astype(int), 0, 19)
        gradients = []
        
        # Check surrounding areas (simplified gradient)
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = np.clip(x + dx, 0, 19), np.clip(y + dy, 0, 19), np.clip(z + dz, 0, 19)
            if self.global_model[nx, ny, nz] == 0:  # Unexplored area
                gradients.append(((nx - 10, ny - 10, nz - 10) + (0,) * (agent.dimension - 3), 2.0))
            else:
                gradient_value = self.global_model[nx, ny, nz] - self.global_model[x, y, z]
                gradients.append(((nx - 10, ny - 10, nz - 10) + (0,) * (agent.dimension - 3), abs(gradient_value)))
        
        # Sort gradients by strength
        gradients.sort(key=lambda g: g[1], reverse=True)
        
        # Also factor in separation from other agents
        other_positions = [a.position for a in other_agents if a.id != agent.id]
        
        # Find the best direction balancing gradient strength and agent separation
        best_direction = None
        best_score = -float('inf')
        
        for gradient_dir, gradient_strength in gradients:
            direction = np.array(gradient_dir) - position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
                # Calculate separation score
                separation_score = 0
                for other_pos in other_positions:
                    projected_pos = position + direction
                    distance = np.linalg.norm(projected_pos - other_pos)
                    separation_score += min(2.0, distance)
                
                # Combined score: gradient strength + separation
                score = gradient_strength + 0.5 * separation_score / max(1, len(other_positions))
                
                if score > best_score:
                    best_score = score
                    best_direction = direction
        
        if best_direction is None:
            # Fallback: random direction
            best_direction = np.random.uniform(-1, 1, size=agent.dimension)
            if np.linalg.norm(best_direction) > 0:
                best_direction = best_direction / np.linalg.norm(best_direction)
        
        return best_direction

    def visualize_model(self):
        """Visualize the current state of the global model"""
        fig = plt.figure(figsize=(12, 8))
        
        # 3D visualization
        ax1 = fig.add_subplot(121, projection='3d')
        x, y, z = np.where(self.global_model > 0)
        values = self.global_model[x, y, z]
        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
        
        scatter = ax1.scatter(x, y, z, c=normalized_values, cmap='viridis', alpha=0.6, s=50)
        ax1.set_title('Swarm Queen\'s Global Model')
        ax1.set_xlabel('X Dimension')
        ax1.set_ylabel('Y Dimension')
        ax1.set_zlabel('Z Dimension')
        
        # Heatmap slice through middle of Z dimension
        ax2 = fig.add_subplot(122)
        mid_z = self.global_model.shape[2] // 2
        heatmap = ax2.imshow(self.global_model[:, :, mid_z], cmap='viridis', origin='lower')
        ax2.set_title(f'Signal Strength at Z={mid_z}')
        ax2.set_xlabel('X Dimension')
        ax2.set_ylabel('Y Dimension')
        plt.colorbar(heatmap, ax=ax2, label='Signal Strength')
        
        plt.tight_layout()
        plt.show()

class AcousticSwarmSimulation:
    """Main simulation class for the acoustic swarm intelligence"""
    
    def __init__(self, dimension=4, num_agents=20):
        self.dimension = dimension
        self.structure = HyperdimensionalStructure(dimension=dimension)
        self.queen = SwarmQueen(dimension=dimension)
        
        # Create diversified agents
        self.agents = []
        sensory_types = list(SensoryType)
        
        for i in range(num_agents):
            # Distribute agents with different sensory types
            sensory_type = sensory_types[i % len(sensory_types)]
            
            # Random initial position
            position = np.random.uniform(-3, 3, size=dimension)
            
            agent = Agent(i, sensory_type, position, dimension=dimension)
            self.agents.append(agent)
    
    def run_simulation(self, steps=100, visualize_every=20):
        """Run the swarm simulation for a specified number of steps"""
        convergence_history = []
        agent_weight_history = {agent.id: [] for agent in self.agents}
        
        for step in range(steps):
            print(f"Step {step+1}/{steps}, Convergence: {self.queen.convergence:.4f}")
            
            # Each agent pings the structure
            for agent in self.agents:
                signal_data = agent.ping(self.structure)
                feedback = self.queen.process_signal(signal_data)
                
                # Store agent weights for analysis
                agent_weight_history[agent.id].append(feedback['weight'])
            
            # The queen suggests movements
            for agent in self.agents:
                direction = self.queen.suggest_movement(agent, self.agents)
                agent.move(direction)
                
                # Occasionally recalibrate underperforming agents
                if self.queen.agent_weights[agent.id] < 0.3:
                    agent.recalibrate()
            
            # Track convergence
            convergence_history.append(self.queen.convergence)
            
            # Visualize every N steps
            if (step + 1) % visualize_every == 0 or step == steps - 1:
                self.visualize_simulation(step, convergence_history, agent_weight_history)
    
    def visualize_simulation(self, step, convergence_history, agent_weight_history):
        """Visualize the current state of the simulation"""
        plt.figure(figsize=(18, 10))
        
        # Convergence plot
        plt.subplot(2, 3, 1)
        plt.plot(convergence_history)
        plt.title('Model Convergence')
        plt.xlabel('Simulation Step')
        plt.ylabel('Convergence Score')
        
        # Agent weights
        plt.subplot(2, 3, 2)
        for agent_id, weights in agent_weight_history.items():
            agent = next(a for a in self.agents if a.id == agent_id)
            sensory_label = agent.sensory_type.name
            plt.plot(weights, label=f"Agent {agent_id} ({sensory_label})")
        plt.title('Agent Trust Weights')
        plt.xlabel('Simulation Step')
        plt.ylabel('Weight')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        # Agent positions in 3D
        ax = plt.subplot(2, 3, 3, projection='3d')
        
        # Plot the structure's center
        ax.scatter([self.structure.center[0]], 
                  [self.structure.center[1]], 
                  [self.structure.center[2]], 
                  color='red', marker='*', s=200, label='Structure Center')
        
        # Plot the agents
        for agent in self.agents:
            color = {
                SensoryType.FREQUENCY_ANALYZER: 'blue',
                SensoryType.WAVE_INTERFERENCE: 'green',
                SensoryType.TOPOLOGICAL_DISTORTION: 'purple',
                SensoryType.GEOMETRIC_MEASUREMENT: 'orange'
            }[agent.sensory_type]
            
            ax.scatter([agent.position[0]], 
                      [agent.position[1]], 
                      [agent.position[2]], 
                      color=color, marker='o', s=50)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Frequency Analyzer'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Wave Interference'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Topological Distortion'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Geometric Measurement'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Structure Center')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        ax.set_title('Agent Positions')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])
        
        # Global model visualization
        plt.subplot(2, 3, 4)
        mid_z = self.queen.global_model.shape[2] // 2
        plt.imshow(self.queen.global_model[:, :, mid_z], cmap='viridis', origin='lower')
        plt.title(f'Global Model (Z={mid_z})')
        plt.colorbar(label='Signal Strength')
        
        # Agent confidence analysis
        plt.subplot(2, 3, 5)
        
        # Group agents by sensory type
        confidence_by_type = {sensory_type: [] for sensory_type in SensoryType}
        for agent in self.agents:
            confidence_by_type[agent.sensory_type].append(agent.confidence)
        
        # Create box plot
        sensory_labels = [t.name.replace('_', '\n') for t in SensoryType]
        confidence_data = [confidence_by_type[t] for t in SensoryType]
        
        plt.boxplot(confidence_data)
        plt.xticks(range(1, len(sensory_labels) + 1), sensory_labels)
        plt.title('Agent Confidence by Sensory Type')
        plt.ylabel('Confidence Score')
        
        # 3D visualization of the knowledge model
        ax2 = plt.subplot(2, 3, 6, projection='3d')
        x, y, z = np.where(self.queen.global_model > 0.2)  # Only show significant signals
        values = self.queen.global_model[x, y, z]
        
        scatter = ax2.scatter(x, y, z, c=values, cmap='viridis', alpha=0.6, s=30)
        ax2.set_title('3D Global Model Visualization')
        plt.colorbar(scatter, ax=ax2, label='Signal Strength')
        
        plt.tight_layout()
        plt.savefig(f'swarm_simulation_step_{step+1}.png')
        plt.show()

# Run the simulation
def main():
    print("Initializing Acoustic Swarm Intelligence Simulation...")
    simulation = AcousticSwarmSimulation(dimension=4, num_agents=16)
    print(f"Created simulation with {len(simulation.agents)} agents exploring a {simulation.dimension}D structure")
    print("Structure properties: complexity={:.2f}, frequency={:.2f}, density={:.2f}".format(
        simulation.structure.complexity,
        simulation.structure.frequency,
        simulation.structure.density
    ))
    
    print("\nStarting simulation...")
    simulation.run_simulation(steps=50, visualize_every=10)
    print("Simulation complete!")
    
    print("\nFinal model convergence: {:.4f}".format(simulation.queen.convergence))
    
    # Analyze agent performance
    print("\nAgent Performance Analysis:")
    for agent in simulation.agents:
        weight = simulation.queen.agent_weights[agent.id]
        print(f"Agent {agent.id} ({agent.sensory_type.name}): Weight={weight:.4f}, Confidence={agent.confidence:.4f}")
    
    # Visualize final state
    simulation.queen.visualize_model()

if __name__ == "__main__":
    main()
