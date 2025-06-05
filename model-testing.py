import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

class CommandCorrector:
    def __init__(self):
        self.session = ort.InferenceSession("onnx/quantized/model.quant.onnx", providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
        
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
        result = self.correct(self.test_command)
        print("\nModel output shape:", result[0].shape)
        print("First few values of output:", result[0][0][:5])

    def correct(self, text, context=None):
        # Combine system prompt with the command
        full_prompt = f"{self.system_prompt}\nCommand to correct: {text}"
        if context:
            full_prompt += f"\nContext: {context}"
            
        # Tokenize with proper padding and truncation
        encoded = self.tokenizer(
            full_prompt,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Prepare only the required inputs
        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
        
        # Run inference
        outputs = self.session.run(None, model_inputs)
        
        # For now, just return the raw output for inspection
        return outputs
    
    def _process_output(self, model_output):
        # This method will need to be implemented based on your model's
        # output format after fine-tuning
        # For now, it's a placeholder
        return {
            "corrected_command": "",
            "confidence": 0.0,
            "explanation": ""
        }

if __name__ == "__main__":
    corrector = CommandCorrector()
    corrector.test()