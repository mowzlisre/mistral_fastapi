from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json 
from models.mistral_model import generateQuestions, generateTopics, generateMCQ, generateMAQ, evaluateResponse
from nltk.tokenize import sent_tokenize
from typing import List
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

class MainInferenceRequest(BaseModel):
    context: str

class QuestionInferenceRequest(BaseModel):
    context: str
    question: str = None
    answer: str = None

def sliding_window_chunks(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


@app.post("/generate/qa")
async def generate_question_answer(request: MainInferenceRequest):
    context = request.context
    result = generateQuestions(context)
    return json.loads(result)

@app.post("/generate/mcq")
async def generate_mcq(request: QuestionInferenceRequest):
    context = request.context
    question = request.question
    answer = request.answer
    result = generateMCQ(context, question, answer)
    return json.loads(result)

@app.post("/generate/maq")
async def generate_mcq(request: QuestionInferenceRequest):
    context = request.context
    question = request.question
    answer = request.answer
    result = generateMAQ(context, question, answer)
    return json.loads(result)

@app.post("/generate/topics")
async def generate_topics(request: MainInferenceRequest):
    context = request.context

    try:
        chunks = sliding_window_chunks(context, chunk_size=500, overlap=100)
        all_topics = []

        for chunk in chunks:
            topics_output = generateTopics(chunk)
            try:
                topics_array = json.loads(topics_output)
                if isinstance(topics_array, list):
                    all_topics.extend(topics_array)
            except json.JSONDecodeError:
                continue  # Skip malformed chunk output

        # Deduplicate topics
        unique_topics = list(set(topic.strip() for topic in all_topics if isinstance(topic, str)))
        return unique_topics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during topic generation: {str(e)}")
    

class EvaluationRequestItem(BaseModel):
    id: int
    context: str
    question: str
    answer: str
    max_points: int
    feedback: Optional[str] = None
    points: Optional[int] = None

@app.post("/evaluate")
async def evaluate_batch(request: List[EvaluationRequestItem]):
    for item in request:
        result = evaluateResponse(item)
        item.feedback = result["feedback"]
        item.points = result["points"]

    return request