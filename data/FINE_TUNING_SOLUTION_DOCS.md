# 🎉 Gemini 2.5 Fine-Tuning Solution

## 🔍 Problem Summary

After extensive debugging, we discovered the root cause of the fine-tuning error:

**Error Message:**
```
InvalidArgument: 400 Row: 0. Missing required `contents` field.
```

## 💡 Root Cause

The Google sample dataset at `gs://cloud-samples-data/vertex-ai/model-evaluation/peft_train_sample.jsonl` uses an **OUTDATED FORMAT** that is incompatible with Gemini 2.5 models.

### ❌ Old Format (Doesn't Work)
```json
{
  "input_text": "Patient has asthma...",
  "output_text": "Allergy / Immunology"
}
```

### ✅ New Format (Required for Gemini 2.5)
```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "What is machine learning?"}]
    },
    {
      "role": "model",
      "parts": [{"text": "Machine learning is..."}]
    }
  ]
}
```

## 🎯 Complete Solution

### Step 1: Create Properly Formatted Dataset

```python
import json

training_examples = [
    {
        "contents": [
            {"role": "user", "parts": [{"text": "What is machine learning?"}]},
            {"role": "model", "parts": [{"text": "Machine learning is a subset of AI that enables computers to learn from experience..."}]}
        ]
    },
    # Add more examples...
]

# Save to JSONL file
with open('training_data.jsonl', 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\\n')
```

### Step 2: Upload to Google Cloud Storage

**IMPORTANT**: Training dataset MUST be on GCS (local files don't work!)

```python
from google.cloud import storage

# Upload to GCS
PROJECT_ID = 'your-project-id'
BUCKET_NAME = f'{PROJECT_ID}-gemini-tuning'

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.create_bucket(BUCKET_NAME, location='us-central1')

# Upload file
blob = bucket.blob('training_data/training.jsonl')
blob.upload_from_filename('training_data.jsonl')

GCS_PATH = f'gs://{BUCKET_NAME}/training_data/training.jsonl'
print(f"Uploaded to: {GCS_PATH}")
```

### Step 3: Run Fine-Tuning

```python
import vertexai
from vertexai.preview.tuning import sft

PROJECT_ID = 'your-project-id'
REGION = 'us-central1'  # Required for fine-tuning

vertexai.init(project=PROJECT_ID, location=REGION)

tuning_job = sft.train(
    source_model='gemini-2.5-pro',  # or gemini-2.5-flash
    train_dataset=GCS_PATH,  # MUST be GCS path (gs://...)
    tuned_model_display_name='my-tuned-model-v1'
)

# Wait for completion (15-30 minutes)
tuning_job.wait()

# Get tuned model endpoint
tuned_model_endpoint = tuning_job.tuned_model_endpoint_name
print(f"Tuned model: {tuned_model_endpoint}")
```

### Step 4: Use Tuned Model

```python
from vertexai.generative_models import GenerativeModel

tuned_model = GenerativeModel(tuned_model_endpoint)
response = tuned_model.generate_content("What is machine learning?")
print(response.text)
```

## 📊 What We Learned

1. ✅ **Gemini 2.5 models ARE available** in all regions (us-central1, europe-west1, etc.)
2. ✅ **Models support fine-tuning**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
3. ❌ **Google's sample dataset uses wrong format** for Gemini 2.5
4. ✅ **Solution**: Create properly formatted dataset with `contents` field
5. ⚠️ **Training data MUST be on GCS** - local files are not accepted

## 🔧 Key Differences

| Aspect | Old Format | New Format |
|--------|-----------|------------|
| Field names | `input_text`, `output_text` | `contents` |
| Structure | Flat key-value | Nested conversation |
| Roles | Not specified | `user`, `model` |
| Parts | Direct text | Array of parts with text |

## 💾 Dataset Format Reference

### Single Turn Example
```json
{
  "contents": [
    {"role": "user", "parts": [{"text": "Question"}]},
    {"role": "model", "parts": [{"text": "Answer"}]}
  ]
}
```

### Multi-Turn Example
```json
{
  "contents": [
    {"role": "user", "parts": [{"text": "First question"}]},
    {"role": "model", "parts": [{"text": "First answer"}]},
    {"role": "user", "parts": [{"text": "Follow-up question"}]},
    {"role": "model", "parts": [{"text": "Follow-up answer"}]}
  ]
}
```

## 🚀 Quick Start

The updated notebook (`whitepapers_exercises.ipynb`) now includes:
- ✅ Properly formatted training dataset
- ✅ Correct model configuration
- ✅ Working fine-tuning code
- ✅ Complete error handling

Just run the cells in order and fine-tuning should work!

## 📚 References

- [Vertex AI Fine-tuning Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning)
- [Dataset Format Specification](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning)
- [Gemini API Reference](https://ai.google.dev/gemini-api/docs/model-tuning)

---

**Status**: ✅ **RESOLVED** - Fine-tuning now works with proper dataset format!

