import kagglehub
import pandas as pd
from pathlib import Path
from transformer.transformer_testing import CommandCorrector
import numpy as np
from tqdm import tqdm

class DataEmbedder:
    def __init__(self, CommandCorrector,selected_files=None):
        self.selected_files = selected_files
        self.corrector = CommandCorrector

        self.all_embeddings = []
        self.all_commands = []

    def load_and_process_datasets(self):
        dataset_path = kagglehub.dataset_download("vaibhavdlights/linuxcmdmacos-commands")
        dataset_path = Path(dataset_path)
        
        if self.selected_files is None:
            self.selected_files = ['linux-commands.csv', 'macos-commands.csv', 'linux-commands-description.csv']
        

        for file_name in self.selected_files:
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
                        embeddings, attention_mask = self.corrector.correct(cmd)

                        # Apply mean pooling
                        pooled_embedding = self.mean_pooling(embeddings, attention_mask)
                        
                        self.all_embeddings.append(pooled_embedding[0])  # Take first (and only) item from batch
                        self.all_commands.append(cmd)
                    except Exception as e:
                        print(f"Error processing command '{cmd}': {str(e)}")
                        print(f"Embeddings shape: {embeddings.shape}")
                        print(f"Attention mask shape: {attention_mask.shape}")
            
            if self.all_embeddings:
                print(f"Sample embedding shape: {self.all_embeddings[-1].shape}")

        self.all_embeddings = np.array(self.all_embeddings)
        self.all_commands = np.array(self.all_commands)
        
        print(f"\nProcessed {len(self.all_commands)} commands")
        print(f"Embedding shape: {self.all_embeddings.shape}")
    
    def mean_pooling(self, output, attention_mask):

        mask_expanded = np.expand_dims(attention_mask, axis=-1)
        masked_output = output * mask_expanded
        sum_embeddings = np.sum(masked_output, axis=1) #[batch size, hidden_size]
        

        sum_mask = np.sum(mask_expanded, axis=1)
        
        # Avoid division by zero
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings
        
if __name__ == "__main__":
    print("Loading corrector...")
    corrector = CommandCorrector(model_name="microsoft/unixcoder-base", encoder_only=True)

    # selected_files = ['linux_commands.csv', 'macos_commands.csv', 'cmd_commands.csv']
    selected_files = ['macos_commands.csv']

    print("Loading data embedder...")
    data_embedder = DataEmbedder(corrector, selected_files)

    print("Loading and processing datasets...")
    data_embedder.load_and_process_datasets()
    embeddings = data_embedder.all_embeddings
    commands = data_embedder.all_commands

    print("Done, saving embeddings and commands...")
    np.save('transformer/cli_commands/command_embeddings.npy', embeddings)
    np.save('transformer/cli_commands/command_names.npy', commands)