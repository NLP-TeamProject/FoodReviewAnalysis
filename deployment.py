from transformers import HfApi
import os

# Set your model name and task name
model_name = "Sentiment_application"
task_name = "senti_app"

api = HfApi()
result = api.create_repo(
    name=model_name,
    tags=[task_name],
    private=False,
    organization="huggingface",
)

# Push your Panel app to the newly created repository
os.system(f"transformers-cli repo create {model_name}")
os.system(f"transformers-cli repo upload {model_name}")