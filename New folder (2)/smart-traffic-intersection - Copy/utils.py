"""
Utility functions for the traffic simulation.
"""
import time

EPOCHS = 100  # Number of training epochs

def log_data(episode, state, action, reward, next_state, ambulance):
    """
    Log training data to a CSV file.
    
    Args:
        episode (int): Current episode number
        state (int): Current state index
        action (int): Action taken
        reward (float): Reward received
        next_state (int): Resulting state index
        ambulance (bool): Whether an ambulance was detected
    """
    with open(f"training_log_ep{episode}.csv", "a") as f:
        f.write(f"{time.time()},{state},{action},{reward},{next_state},{ambulance}\n")

def load_all_logs():
    """
    Load all training log data.
    
    Returns:
        list: List of training data from logs
    """
    # Implementation needed
    pass

def sample_batch(data, batch_size=64):
    """
    Sample a batch from the training data.
    
    Args:
        data (list): Complete training data
        batch_size (int): Size of the batch to sample
        
    Returns:
        list: Sampled batch of experience tuples
    """
    # Implementation needed
    pass