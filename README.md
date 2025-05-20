# CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models

📷 This is the code repository for the paper: CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models. **ACL 2025 Main**.

<div>
<img src="./image/image_div.png" width="100%">
</div>
👀 Presentation of data in CCHall.

## Dataset Structure

The dataset contains a single split:
```yaml
- `test`: Contains 3600 examples.
```

Each example has the following fields:
```yaml
- `question_id`: (integer) Unique identifier for the question.
- `dataset_source`: (string) Origin dataset (e.g., "AMBER", "GQA").
- `image_id`: (string) Identifier for the image.
- `question`: (string) The question asked about the image.
- `response1`: (string) First response to the question.
- `response2`: (string) Second response (often in a different language).
- `choice`: (string) Choices for hallucination type.
- `label`: (string) Label for hallucination type.
- `language`: (string) Language code (e.g., "sw", "en").
- `language_category`: (string) Category like "low resource".
- `image`: (binary) The binary content of the image.
```

## 🎯 Installation

### 📦 1. Dataset Preparation
#### Load Dataset from Huggingface
```python 
import datasets
dataset = load_dataset("67L11/CCHall", split="test")
```

#### Load Dataset from Google Drive 
Please download the corresponding dataset from [🤗HuggingFace](https://drive.google.com/drive/folders/110bWAuA7sHHWfbtJPcjRgnzl1NuOcj-I?usp=drive_link)/[☁️Google Drive](https://drive.google.com/drive/folders/110bWAuA7sHHWfbtJPcjRgnzl1NuOcj-I?usp=drive_link) and place the unzipped content in the `data` folder.

📁 This download includes:

- ✅ `test.parquet`:  
  A processed Parquet file containing all merged data entries and metadata from multiple datasets.

- ✅ Original image archives (compressed folders) for the following 4 data sources:
  - `AMBER`
  - `GQA`
  - `xFlickrCO`
  - `XM3600`

#### 📂 Expected directory structure after unzipping:
```yaml
project-root/
├── data/
│ ├── amber_image/
│ ├── gqa_image/
│ ├── xflickr_image/
│ ├── xm3600_image/
│ └── test.parquet
├── your_code.py
└── README.md
```
Once the files are in place, you can run the data processing script or load `test.parquet` directly for analysis or modeling.


```python 
import datasets
dataset = load_dataset('parquet', data_files='data/test.parquet')
```

### 2. Evaluation for reproduction on GPT-4o
```python 
import base64
import os
import time
from openai import OpenAI
from datasets import load_dataset
import io

# --- Configuration ---
API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key
# Or use environment variable: os.getenv("OPENAI_API_KEY")
API_BASE = "https://api.openai.com/v1"
GPT_MODEL = "gpt-4o"
DATASET_REPO_ID = "67L11/CCHall"

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def get_prompt(question, answer1, answer2, choice):
    prompt = f"""You are a hallucination detection system tasked with categorizing hallucinations based on images , questions and answers in two different languages. Your job is to analyze the provided image, question, and both language answers to determine the type of hallucination, if any.

Explanation of Hallucination Types
Non-hallucination: Both the English and the other language answers are consistent with the content of the image, and they correctly describe what is present in the image.
Cross-modal, non-cross-lingual hallucination: Both the English and the other language answers are inconsistent with the image content, specifically regarding the existence of objects. The key point is that the both languages make the same error, mentioning the existence of the same object(s) that do not appear in the image.
Cross-lingual, non-cross-modal hallucination: The English answer is consistent with the image, but the answer in the other language contains an error regarding the existence of objects.
Cross-modal, cross-lingual hallucination: Both the English and the other language answers are inconsistent with the image, and they differ in the objects they describe as being present. In other words, both languages make errors, but they reference different nonexistent objects.

Task
Based on the following information, identify the type of hallucination:
Image: [The provided image]
Question: {question}
Answer1: {answer1}
Answer2: {answer2}
Options: {choice}

Output Format
**At the end of the analysis, provide a final, concise, and definitive answer in the following format:**
**Final Answer: [Option]. [Option Content].**"""
    return prompt


def encode_image_bytes(image_bytes):
    """Encodes image bytes to a base64 string."""
    if image_bytes is None:
        return None
    return base64.b64encode(image_bytes).decode('utf-8')


def get_gpt4o_response(prompt_text, base64_image_data):
    """Sends the prompt and image to GPT-4o and returns the response content."""
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_data}",
                        },
                    },
                ],
            }
        ],
        max_tokens=256,
    )
    return response.choices[0].message.content


def process_dataset_with_gpt4o():
    dataset = load_dataset(DATASET_REPO_ID, split="test")

    for item in dataset:
        image_id = item['image_id']
        question = item['question']
        answer1 = item['response1']
        answer2 = item['response2']
        choice = item['choice']  
        true_label = item['label']  
        image_bytes = item['image']['bytes'] 

        # 1. Encode the image bytes to base64
        base64_image = encode_image_bytes(image_bytes)

        # 2. Get the prompt
        prompt_text = get_prompt(question, answer1, answer2, choice)

        # 3. Get response from GPT-4o
        gpt4o_full_response = get_gpt4o_response(prompt_text, base64_image)

if __name__ == "__main__":
    process_dataset_with_gpt4o()
```


## 💬 Contact

If you have any questions or suggestions, please create a GitHub issue or email [Yongheng Zhang](mailto:zyhbrz@gmail.com).
