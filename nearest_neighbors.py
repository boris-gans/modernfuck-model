# from annoy import AnnoyIndex
import annoy
import numpy as np
from transformer_testing import CommandCorrector
from sklearn.neighbors import NearestNeighbors

def load_embeddings(path):
    embeddings = np.load(path)
    return embeddings

def mean_pooling(output, attention_mask):
    """
    Perform mean pooling on the output embeddings, taking into account the attention mask.
    This averages all token embeddings except padding tokens.
    
    Args:
        output: tensor of shape [batch_size, sequence_length, hidden_size]
        attention_mask: tensor of shape [batch_size, sequence_length]
    """

    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(output * mask_expanded, axis=1)
    
    sum_mask = np.sum(attention_mask, axis=1, keepdims=True)
    sum_mask = np.clip(sum_mask, 1e-9, None)
    
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings

def create_knn_index(embeddings, n_neighbors=3):
    """
    Create and build a KNN index from the embeddings.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine').fit(embeddings)
    return nbrs

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

def find_similar_commands(input_command, index, embeddings, command_names, use_annoy=True, n_neighbors=3):
    """
    Find the most similar commands to the input command using either Annoy or sklearn's KNN.
    
    Args:
        input_command: The command to find similar commands for
        index: The Annoy or KNN index
        embeddings: The original embeddings array
        command_names: Array of command names
        use_annoy: Whether to use Annoy (True) or sklearn's KNN (False)
        n_neighbors: Number of similar commands to return
    """
    corrector = CommandCorrector()
    embeddings_tensor, attention_mask = corrector.correct(input_command)
    
    input_embedding = mean_pooling(embeddings_tensor, attention_mask)[0]
    
    if use_annoy:
        indices, distances = index.get_nns_by_vector(
            input_embedding, 
            n_neighbors, 
            include_distances=True
        )
    else:
        distances, indices = index.kneighbors(input_embedding.reshape(1, -1))
        indices = indices[0]
        distances = distances[0]
    
    similar_commands = []
    for idx, distance in zip(indices, distances):
        similar_commands.append({
            'command': command_names[idx],
            'distance': float(distance)  # Convert to float for JSON serialization
        })
    
    return similar_commands

if __name__ == "__main__":
    # Load the embeddings and command names
    embeddings = load_embeddings('cli_commands/command_embeddings.npy')
    command_names = np.load('cli_commands/command_names.npy')
    
    # Create both indices
    print("Creating indices...")
    annoy_index = create_annoy_index(embeddings)
    knn_index = create_knn_index(embeddings)
    
    # Test with some example commands
    test_commands = [
        "git stat",  
        "dockr ps", 
        "npm instl",
        "alis",
        "brik",
        "ciksum",
        "breew"
    ]
    
    print("\nTesting command similarity search with Annoy:")
    for cmd in test_commands:
        print(f"\nInput command: {cmd}")
        similar = find_similar_commands(cmd, annoy_index, embeddings, command_names, use_annoy=True)
        print("Similar commands:")
        for result in similar:
            print(f"- {result['command']} (distance: {result['distance']:.4f})")
    
    print("\n\nTesting command similarity search with KNN:")
    for cmd in test_commands:
        print(f"\nInput command: {cmd}")
        similar = find_similar_commands(cmd, knn_index, embeddings, command_names, use_annoy=False)
        print("Similar commands:")
        for result in similar:
            print(f"- {result['command']} (distance: {result['distance']:.4f})")

