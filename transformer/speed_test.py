import time
import numpy as np
from model_testing import CommandCorrector
import random
import string

def generate_test_commands(num_commands=100):
    """Generate a mix of realistic and synthetic commands for testing"""

    prefixes = ['git', 'docker', 'npm', 'python', 'pip', 'ls', 'cd', 'rm', 'cp', 'mv']
    suffixes = ['status', 'commit', 'push', 'pull', 'install', 'run', 'build', 'test', 'init', 'update']
    
    commands = []
    for _ in range(num_commands):
        if random.random() < 0.7:
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            # Sometimes add a typo
            if random.random() < 0.3:
                suffix = suffix[:-1] + random.choice(string.ascii_lowercase)
            command = f"{prefix} {suffix}"
        else:
            command = ' '.join([''.join(random.choices(string.ascii_lowercase, k=random.randint(2, 8))) 
                              for _ in range(random.randint(2, 4))])
        commands.append(command)
    return commands

def measure_performance(corrector, commands, num_runs=3):
    """Measure various performance metrics."""
    total_tokens = 0
    total_time = 0
    times_per_command = []
    tokens_per_command = []
    
    # Warmup run
    print("Warming up model...")
    for _ in range(5):
        corrector.correct("git status")
    
    print("\nStarting performance measurements...")
    for run in range(num_runs):
        run_start = time.time()
        for cmd in commands:
            # Measure tokenization time
            token_start = time.time()
            encoded = corrector.tokenizer(
                cmd,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
            )
            token_time = time.time() - token_start
            
            # Measure inference time
            infer_start = time.time()
            _ = corrector.correct(cmd)
            infer_time = time.time() - infer_start
            
            # Record metrics
            num_tokens = len(encoded['input_ids'][0])
            total_tokens += num_tokens
            total_time += token_time + infer_time
            times_per_command.append(token_time + infer_time)
            tokens_per_command.append(num_tokens)
            
        print(f"Run {run + 1}/{num_runs} completed")
    
    # Calculate statistics
    avg_time_per_command = np.mean(times_per_command)
    avg_time_per_token = total_time / total_tokens
    std_time_per_command = np.std(times_per_command)
    
    # Predict time for 5000 commands
    predicted_time_5k = avg_time_per_command * 5000
    
    print("\nPerformance Results:")
    print(f"Average time per command: {avg_time_per_command*1000:.2f}ms")
    print(f"Average time per token: {avg_time_per_token*1000:.2f}ms")
    print(f"Standard deviation per command: {std_time_per_command*1000:.2f}ms")
    print(f"\nPredicted time for 5000 commands: {predicted_time_5k:.2f} seconds")
    print(f"Predicted time for 5000 commands: {predicted_time_5k/60:.2f} minutes")
    
    return {
        'avg_time_per_command': avg_time_per_command,
        'avg_time_per_token': avg_time_per_token,
        'std_time_per_command': std_time_per_command,
        'predicted_time_5k': predicted_time_5k
    }

if __name__ == "__main__":
    # Initialize the model
    print("Initializing model...")
    corrector = CommandCorrector()
    
    # Generate test commands
    test_commands = generate_test_commands(100)
    
    # Run performance tests
    results = measure_performance(corrector, test_commands) 