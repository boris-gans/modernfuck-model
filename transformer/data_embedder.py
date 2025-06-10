import kagglehub
import pandas as pd
from pathlib import Path
from transformer_testing import CommandCorrector
import numpy as np
from tqdm import tqdm

def mean_pooling(output, attention_mask):
    """
    Perform mean pooling on the output embeddings, taking into account the attention mask.
    This averages all token embeddings except padding tokens.
    
    Args:
        output: tensor of shape [batch_size, sequence_length, hidden_size]
        attention_mask: tensor of shape [batch_size, sequence_length]
    """

    # Expand attention mask to match output dimensions
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    masked_output = output * mask_expanded
    sum_embeddings = np.sum(masked_output, axis=1) #[batch size, hidden_size]
    

    sum_mask = np.sum(mask_expanded, axis=1)
    
    # Avoid division by zero
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings

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
                    # Get model output and attention mask
                    embeddings, attention_mask = corrector.correct(cmd)

                    # Apply mean pooling
                    pooled_embedding = mean_pooling(embeddings, attention_mask)
                    
                    all_embeddings.append(pooled_embedding[0])  # Take first (and only) item from batch
                    all_commands.append(cmd)
                except Exception as e:
                    print(f"Error processing command '{cmd}': {str(e)}")
                    print(f"Embeddings shape: {embeddings.shape}")
                    print(f"Attention mask shape: {attention_mask.shape}")
        
        if all_embeddings:
            print(f"Sample embedding shape: {all_embeddings[-1].shape}")
    
    all_embeddings = np.array(all_embeddings)
    all_commands = np.array(all_commands)
    
    print(f"\nProcessed {len(all_commands)} commands")
    print(f"Embedding shape: {all_embeddings.shape}")
    
    return all_embeddings, all_commands

if __name__ == "__main__":
    # selected_files = ['linux_commands.csv', 'macos_commands.csv', 'cmd_commands.csv']
    selected_files = ['macos_commands.csv']

    embeddings, commands = load_and_process_datasets(selected_files)
    
    np.save('cli_commands/command_embeddings.npy', embeddings)
    np.save('cli_commands/command_names.npy', commands)
