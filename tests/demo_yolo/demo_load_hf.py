from transformers import AutoModel

model = AutoModel.from_pretrained(
    "artimes/artimes-yolov8n-260323-1629",
    trust_remote_code=True,
)
print(model)
