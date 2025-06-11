import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from pathlib import Path

from transformer.transformer_loading import Transformer
from transformer.transformer_testing import CommandCorrector
from transformer.data_embedder import DataEmbedder

# inits
model_name = "microsoft/unixcoder-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# for running all models and tests
if __name__ == "__main__":
    print(f"Loading {model_name}...")
    transformer = Transformer(model_name=model_name, tokenizer=tokenizer, model=model)
    print(f"Exporting {model_name} to ONNX...")
    transformer.export_to_onnx("git status")
    print(f"Quantizing {model_name}...")
    transformer.quantize(Path("onnx/"))

    print(f"Loading corrector...")
    corrector = CommandCorrector(model_name=model_name, encoder_only=True)
    print(f"Testing corrector...")
    corrector.test()


    print(f"Loading data embedder...")
    selected_files = ['macos_commands.csv']
    data_embedder = DataEmbedder(corrector, selected_files)

    print("Loading and processing datasets...")
    data_embedder.load_and_process_datasets()
    embeddings = data_embedder.all_embeddings
    commands = data_embedder.all_commands

    print("Done, saving embeddings and commands...")
    np.save('transformer/cli_commands/command_embeddings.npy', embeddings)
    np.save('transformer/cli_commands/command_names.npy', commands)