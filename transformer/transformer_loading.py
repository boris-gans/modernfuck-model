import torch
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

class Transformer:
    def __init__(self, model_name, tokenizer, model):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
    
    def export_to_onnx(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        onnx_path = Path("onnx/")
        onnx_path.mkdir(exist_ok=True)
        onnx_model_path = onnx_path / "model.onnx"
        if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
            decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]])
            torch.onnx.export(
                self.model,
                (input_ids, attention_mask, decoder_input_ids),
                onnx_model_path,
                input_names=["input_ids", "attention_mask", "decoder_input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "decoder_input_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=14
            )
        else:
            # Encoder-only model
            print("Detected encoder-only model.")
            torch.onnx.export(
                self.model,
                (input_ids, attention_mask),
                onnx_model_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "last_hidden_state": {0: "batch", 1: "sequence"}
                },
                opset_version=14
            )
    def quantize(self, model_path):
        quantized_model_path = Path("onnx/quantized/")
        quantized_model_path.mkdir(exist_ok=True)
        quantize_dynamic(
            str(model_path / "model.onnx"),
            str(quantized_model_path / "model.quant.onnx"),
            weight_type=QuantType.QUInt8
        )

if __name__ == "__main__":
    # UnixCoder
    model_name = "microsoft/unixcoder-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    # MiniLM
    # model_name = "nreimers/MiniLM-L6-H384-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)

    # CodeT5
    # model_name = "Salesforce/codet5-small"
    # tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name)

    transformer = Transformer(model_name, tokenizer, model)
    transformer.export_to_onnx("git status")
    transformer.quantize(Path("onnx/"))

# model_path = Path("onnx/")
# model_path.mkdir(exist_ok=True)

# # new
# inputs = tokenizer("git status", return_tensors="pt")
# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]

# # Export model to ONNX
# onnx_model_path = model_path / "model.onnx"
# if hasattr(model, "encoder") and hasattr(model, "decoder"):
#     # T5-style encoder-decoder
#     print("Detected encoder-decoder model. Adding decoder_input_ids for ONNX export.")
#     decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

#     torch.onnx.export(
#         model,
#         (input_ids, attention_mask, decoder_input_ids),
#         onnx_model_path,
#         input_names=["input_ids", "attention_mask", "decoder_input_ids"],
#         output_names=["logits"],
#         dynamic_axes={
#             "input_ids": {0: "batch", 1: "sequence"},
#             "attention_mask": {0: "batch", 1: "sequence"},
#             "decoder_input_ids": {0: "batch", 1: "sequence"},
#             "logits": {0: "batch", 1: "sequence"},
#         },
#         opset_version=14
#     )
# else:
#     # Encoder-only model
#     print("Detected encoder-only model.")
#     torch.onnx.export(
#         model,
#         (input_ids, attention_mask),
#         onnx_model_path,
#         input_names=["input_ids", "attention_mask"],
#         output_names=["last_hidden_state"],
#         dynamic_axes={
#             "input_ids": {0: "batch", 1: "sequence"},
#             "attention_mask": {0: "batch", 1: "sequence"},
#             "last_hidden_state": {0: "batch", 1: "sequence"}
#         },
#         opset_version=14
#     )




# Export to ONNX
# dummy_input = tokenizer("git status", return_tensors="pt").input_ids
# torch.onnx.export(
#     model,
#     (dummy_input["input_ids"], dummy_input["attention_mask"]),
#     model_path / "model.onnx",
#     input_names=["input_ids", "attention_mask"],
#     output_names=["last_hidden_state"],
#     dynamic_axes={
#         "input_ids": {0: "batch", 1: "sequence"},
#         "attention_mask": {0: "batch", 1: "sequence"},
#         "last_hidden_state": {0: "batch", 1: "sequence"}
#     },
#     opset_version=14
# )

# Model quantization

# quantized_model_path = Path("onnx/quantized/")
# quantized_model_path.mkdir(exist_ok=True)
# quantize_dynamic(
#     str(model_path / "model.onnx"),
#     str(quantized_model_path / "model.quant.onnx"),
#     weight_type=QuantType.QUInt8
# )