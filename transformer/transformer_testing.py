import onnxruntime as ort
from transformers import AutoTokenizer, RobertaTokenizer
import numpy as np

class CommandCorrector:
    def __init__(self, model_name, encoder_only=False):
        self.session = ort.InferenceSession("onnx/quantized/model.quant.onnx", providers=["CPUExecutionProvider"])
        self.encoder_only = encoder_only

        if encoder_only:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # System prompt to guide the model's behavior
        self.system_prompt = """You are a CLI command correction assistant. 
        Your task is to correct shell commands and provide suggestions.
        Format your response as a JSON with the following structure:
        {
            "corrected_command": "the corrected command",
            "confidence": 0.95,
            "explanation": "brief explanation of the correction"
        }"""
        self.test_command = "git stat"

    def test(self):
        print("Testing command correction with:", self.test_command)
        print(f"Model type: {'Encoder-only' if self.encoder_only else 'Encoder-decoder'}")
        result = self.correct(self.test_command)
        
        if self.encoder_only:
            cls_representation, attention_mask = result
            print("\nCLS token representation shape:", cls_representation.shape)
            print("First few values of CLS representation:", cls_representation[0][:5])
        else:
            last_hidden_state, attention_mask = result
            print("\nModel output shape:", last_hidden_state.shape)
            print("First few values of output:", last_hidden_state[0][0][:5])

    def correct(self, text, context=None):
        # Combine system prompt with the command
        full_prompt = f"{self.system_prompt}\nCommand to correct: {text}"
        if context:
            full_prompt += f"\nContext: {context}"
            
        # Tokenize with proper padding and no truncation
        encoded = self.tokenizer(
            full_prompt,
            return_tensors="np",
            padding="max_length",
            truncation=False,
            max_length=512,
            add_special_tokens=True  # Ensure CLS and SEP tokens are added
        )
        
        # Prepare model inputs based on model type
        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
        
        # Only add decoder_input_ids for encoder-decoder models
        if not self.encoder_only:
            # For encoder-decoder models, start with the decoder start token (usually 0 or pad_token_id)
            decoder_start_token = getattr(self.tokenizer, 'decoder_start_token_id', 0)
            if decoder_start_token is None:
                decoder_start_token = 0
            decoder_input_ids = np.array([[decoder_start_token]])
            model_inputs['decoder_input_ids'] = decoder_input_ids
        
        # Run inference
        outputs = self.session.run(None, model_inputs)
        last_hidden_state = outputs[0]
        attention_mask = encoded["attention_mask"]
        
        # Extract CLS token representation for all models
        if self.encoder_only:
            # For encoder-only models, extract CLS token (first token) representation
            cls_representation = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            return cls_representation, attention_mask
        else:
            # For encoder-decoder models, return the full output
            # Note: You might want to extract specific tokens here based on your use case
            return last_hidden_state, attention_mask
    
    def _process_output(self, model_output):
        # This method can be expanded to process the CLS token or decoder output
        # into meaningful command corrections
        return {
            "corrected_command": "",
            "confidence": 0.0,
            "explanation": ""
        }

if __name__ == "__main__":
    corrector = CommandCorrector(model_name="microsoft/unixcoder-base", encoder_only=True)
    corrector.test()
    
    print("\n" + "="*50 + "\n")
    
    # Uncomment to test with encoder-decoder model
    # corrector_seq2seq = CommandCorrector(model_name="t5-small", encoder_only=False)
    # corrector_seq2seq.test()