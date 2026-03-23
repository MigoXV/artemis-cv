from transformers import AutoModel

model = AutoModel.from_pretrained(
    "./model-bin/artimes-yolov8n-260323-1629",
    trust_remote_code=True,
)
print(model)
