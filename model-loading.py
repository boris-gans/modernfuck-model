# MiniLM-L6-H256 from nreimers
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import torch

model_name = "nreimers/MiniLM-L6-H384-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model_path = Path("onnx/")
model_path.mkdir(exist_ok=True)

# Export to ONNX
dummy_input = tokenizer("Hello, I'm a test sentence", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    model_path / "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"}
    },
    opset_version=14
)

# Model quantization
from onnxruntime.quantization import quantize_dynamic, QuantType

quantized_model_path = Path("onnx/quantized/")
quantized_model_path.mkdir(exist_ok=True)
quantize_dynamic(
    str(model_path / "model.onnx"),
    str(quantized_model_path / "model.quant.onnx"),
    weight_type=QuantType.QUInt8
)