import kagglehub
import pandas as pd
from pathlib import Path
from transformer_testing import CommandCorrector
import numpy as np
from tqdm import tqdm

def load_and_process_datasets(selected_files=None):
    """
    Load selected datasets from the Linux/MacOS commands dataset and process them through the transformer.
    """
    # Download the dataset
    dataset_path = kagglehub.dataset_download("vaibhavdlights/linuxcmdmacos-commands")
    dataset_path = Path(dataset_path)
    
    if selected_files is None:
        selected_files = ['linux-commands.csv', 'macos-commands.csv', 'linux-commands-description.csv']
    
    print("Initializing transformer...")
    corrector = CommandCorrector()
    
    all_embeddings = []
    all_commands = []
    
    for file_name in selected_files:
        print(f"\nProcessing {file_name}...")
        file_path = dataset_path / file_name
        
        df = pd.read_csv(file_path)
        commands = df['name'].tolist()
        
        batch_size = 32
        for i in tqdm(range(0, len(commands), batch_size)):
            batch_commands = commands[i:i + batch_size]
            
            for cmd in batch_commands:
                try:
                    output = corrector.correct(cmd)
                    # Store the CLS token
                    embedding = output[0][0][0]
                    all_embeddings.append(embedding)
                    all_commands.append(cmd)
                except Exception as e:
                    print(f"Error processing command '{cmd}': {str(e)}")
        print(f"Sample embedding: {all_embeddings[-1]}")
        print(f"All embeddings shape: {np.array(all_embeddings).shape}")
    
    all_embeddings = np.array(all_embeddings)
    all_commands = np.array(all_commands)
    
    print(f"\nProcessed {len(all_commands)} commands")
    print(f"Embedding shape: {all_embeddings.shape}")
    
    return all_embeddings, all_commands

if __name__ == "__main__":

    selected_files = ['linux_commands.csv', 'macos_commands.csv', 'cmd_commands.csv']
    embeddings, commands = load_and_process_datasets(selected_files)
    
    np.save('cli_commands/command_embeddings.npy', embeddings)
    np.save('cli_commands/command_names.npy', commands)
