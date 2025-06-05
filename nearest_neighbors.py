# from annoy import AnnoyIndex
import annoy
import numpy as np
from transformer_testing import CommandCorrector

def load_embeddings(path):
    embeddings = np.load(path)
    return embeddings

def create_annoy_index(embeddings, n_trees=10):
    """
    Create and build an Annoy index from the embeddings.
    """

    dim = embeddings.shape[1]
    index = annoy.AnnoyIndex(dim, 'angular')
    
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    
    #Build the annoy index
    index.build(n_trees)
    return index

def find_similar_commands(input_command, index, embeddings, command_names, n_neighbors=3):
    """
    Find the most similar commands to the input command.
    
    Args:
        input_command: The command to find similar commands for
        index: The Annoy index
        embeddings: The original embeddings array
        command_names: Array of command names
        n_neighbors: Number of similar commands to return
    """
    corrector = CommandCorrector()
    
    output = corrector.correct(input_command)
    input_embedding = output[0][0][0]  #Get the CLS token
    
    # Find the nearest neighbors
    indices, distances = index.get_nns_by_vector(
        input_embedding, 
        n_neighbors, 
        include_distances=True
    )
    
    # Get the corresponding commands and their distances
    similar_commands = []
    for idx, distance in zip(indices, distances):
        similar_commands.append({
            'command': command_names[idx],
            'distance': distance
        })
    
    return similar_commands

if __name__ == "__main__":
    # Load the embeddings and command names
    embeddings = load_embeddings('cli_commands/command_embeddings.npy')
    command_names = np.load('cli_commands/command_names.npy')
    
    # Create the Annoy index
    print("Creating Annoy index...")
    index = create_annoy_index(embeddings)
    
    # Test with some example commands
    test_commands = [
        "git stat",  
        "dockr ps", 
        "npm instl",
        "alis",
        "brik",
        "ce"
    ]
    
    print("\nTesting command similarity search:")
    for cmd in test_commands:
        print(f"\nInput command: {cmd}")
        similar = find_similar_commands(cmd, index, embeddings, command_names)
        print("Similar commands:")
        for result in similar:
            print(f"- {result['command']} (distance: {result['distance']:.4f})")

