import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot_training_curve():
    # Load training data
    try:
        data = np.load("training_data.npy", allow_pickle=True).item()
        timesteps = data['timesteps']
        rewards = data['rewards']
        lengths = data['lengths']
    except FileNotFoundError:
        print("Error: training_data.npy not found. Run the training script first.")
        return
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.scatter(timesteps, rewards, alpha=0.3, s=10, label='Episode reward')
    
    # Calculate and plot moving average
    if len(rewards) > 100:
        smoothed_rewards = gaussian_filter1d(rewards, sigma=5)
        plt.plot(timesteps, smoothed_rewards, 'r-', linewidth=2, label='Smoothed reward')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot episode lengths
    plt.subplot(1, 2, 2)
    plt.scatter(timesteps, lengths, alpha=0.3, s=10, label='Episode length')
    
    # Calculate and plot moving average for lengths
    if len(lengths) > 100:
        smoothed_lengths = gaussian_filter1d(lengths, sigma=5)
        plt.plot(timesteps, smoothed_lengths, 'b-', linewidth=2, label='Smoothed length')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()


def plot_episode_rewards():
    # Load training data
    try:
        data = np.load("training_data.npy", allow_pickle=True).item()
        rewards = data['rewards']
    except FileNotFoundError:
        print("Error: training_data.npy not found. Run the training script first.")
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot raw episode rewards
    episodes = np.arange(1, len(rewards) + 1)
    plt.scatter(episodes, rewards, alpha=0.3, s=10, label='Episode reward')
    
    # Calculate and plot moving average
    if len(rewards) > 100:
        # Use a window of 100 episodes for moving average
        window_size = 100
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Plot the moving average
        plt.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=2, 
                label=f'{window_size}-episode moving avg')
    
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Reward')
    plt.title('Training Progress: Reward per Episode')
    plt.legend()
    plt.grid(True)
    
    # Save and show plot
    plt.tight_layout()
    plt.savefig('episode_rewards.png')
    plt.show()


if __name__ == "__main__":
    plot_training_curve()
    plot_episode_rewards()




