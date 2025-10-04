# Setup Guide for Google Generative AI Course

This guide will help you set up your environment for the 5-Day Google Generative AI Intensive Course.

## Prerequisites

### 1. Kaggle Account Setup
1. Visit [Kaggle](https://www.kaggle.com/) and create an account
2. **Important**: Complete phone verification - this is required for the codelabs
3. Familiarize yourself with Kaggle Notebooks if you haven't used them before
4. Link your Discord account at [kaggle.com/discord/confirmation](https://kaggle.com/discord/confirmation)

### 2. AI Studio Account
1. Visit [AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Generate an API key for Gemini 2.0 access
4. Store your API key securely (we'll use environment variables)

### 3. Discord Community
1. Join the [Kaggle Discord server](https://discord.gg/kaggle)
2. Introduce yourself in the course channel
3. Connect with fellow learners

## Local Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/Tuminha/kaggle_google_generative_ai.git
cd kaggle_google_generative_ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory:
```bash
# AI Studio API Key
GOOGLE_API_KEY=your_api_key_here

# Optional: Other API keys
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### 5. Verify Installation
```python
import google.generativeai as genai
import pandas as pd
import numpy as np
print("✅ All packages installed successfully!")
```

## Course Structure

```
kaggle_google_generative_ai/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── day1_prompt_engineering.ipynb
│   ├── day2_embeddings_rag.ipynb
│   ├── day3_ai_agents.ipynb
│   ├── day4_domain_llms.ipynb
│   └── day5_mlops.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── images/
└── docs/
    └── setup_guide.md
```

## Troubleshooting

### Common Issues

1. **Phone Verification Error**: Ensure you've completed phone verification on Kaggle
2. **API Key Issues**: Double-check your AI Studio API key in the `.env` file
3. **Import Errors**: Make sure you're in the correct virtual environment
4. **Kaggle Access**: Some codelabs require Kaggle account linking

### Getting Help

1. Check the [Kaggle Troubleshooting Guide](https://kaggle.com/docs/notebooks)
2. Post in the Discord course channel
3. Review the course whitepapers and podcast episodes

## Next Steps

1. Complete the setup verification
2. Start with Day 1 materials
3. Join the community discussions
4. Begin your generative AI journey!

---

**Good luck with your course! 🚀**
