import os
import json
import re
import spacy
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import snapshot_download

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

app = FastAPI(title="Resume Parser via ChatGPT API")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = " "
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
            
        text = text.strip()
        text = ' '.join(text.split())
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF file: {str(e)}")


@app.post("/api/resume-v1")
async def parse_resume_chatgpt(file: UploadFile = File(...)):
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    pdf_bytes = await file.read()
    
    cv_text = extract_text_from_pdf(pdf_bytes)
    
    if not cv_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

    prompt = f"""
    You are an expert ATS (Applicant Tracking System) parser.
    Extract the following information from the provided CV text.
    Your response MUST be strict JSON matching this exact structure:
    
    {{
        "Name": "Full name of the candidate",
        "Location": "City, State, or Country",
        "Email Address": "Email address",
        "College Name": ["List of colleges/universities attended"],
        "Degree": ["List of degrees obtained"],
        "Graduation Year": ["List of graduation years"],
        "Companies worked at": ["List of companies the candidate worked for"],
        "Designation": ["List of job titles / positions held"],
        "Skills": ["List of all skills mentioned, cleanly separated into single tools/languages"]
    }}
    
    Rules:
    - If a field is not found in the CV, return an empty string "" for single values, or an empty list [] for array values.
    - Do not invent information. Extract it exactly as it appears in the text.
    - Expand the skills into a nice array (e.g. ["Python", "FastAPI", "MongoDB"]).
    - Return ONLY valid JSON.
    
    CV TEXT TO PARSE:
    {cv_text}
    """
    
    try:
        # Using OpenAI's JSON Mode
        response = client.chat.completions.create(
            model="gpt-4o",  # You can change this to "gpt-4o-mini" for faster/cheaper parsing
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        
        predicted_json = json.loads(response.choices[0].message.content)
        
        return {
            "status": "success",
            "data": predicted_json
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="ChatGPT did not return valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from OpenAI API: {str(e)}")


# try:
#     nlp = spacy.load("en_pipeline_name")
# except OSError:
#     nlp = None
#     print("WARNING: Could not load the fine-tuned spaCy model. Update the path in app.py!")

try:
    print('loading model')
    model_path = snapshot_download(repo_id="ducnv123/resume-praser")
    nlp = spacy.load(model_path)
    print('model loaded')
except Exception as e:
    print(f"Error loading model: {e}")
    nlp = None

# @app.post("/api/resume-v2")
async def parse_resume_finetune(file: UploadFile = File(...)):
    
    if not nlp:
        raise HTTPException(status_code=500, detail="The spaCy model failed to load. Check server logs.")
        
    # if not file.filename.endswith(".pdf"):
    #     raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    pdf_bytes = await file.read()
    
    cv_text = extract_text_from_pdf(pdf_bytes)
    
    if not cv_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

    doc = nlp(cv_text)
    extracted_data = {}

    for ent in doc.ents:
        label = ent.label_
        value = ent.text.strip()

        if label not in extracted_data:
            extracted_data[label] = []

    if label == "Skills":
      skills = value.split(',')
      for skill in skills:
        extracted_data[label].append(skill.strip())
    else:
      extracted_data[label].append(value)

    json_output = json.dumps(extracted_data, indent=4, ensure_ascii=False)

    return json_output


@app.get("/")
def health_check():
    return {"status": "running"}

