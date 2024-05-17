from google.cloud import aiplatform

client = aiplatform.gapic.ModelServiceClient()

# Specify the parent resource
parent = "projects/YOUR_PROJECT_ID/locations/YOUR_LOCATION"

# List models
models = client.list_models(parent=parent)

# Print model details
for model in models:
    print(f"Model name: {model.name}")
    print(f"Model display name: {model.display_name}")
    print(f"Model supported methods: {model.supported_export_formats}")