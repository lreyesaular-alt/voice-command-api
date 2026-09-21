
import json
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import APIRouter, HTTPException, status
from groq import Groq

from src.app.schemas.voice import InstructionPayload, InstructionRequest

router = APIRouter(tags=["instruction"])


@router.post("/instruction", response_model=InstructionPayload)
def route_instruction(
    payload: InstructionRequest,
) -> InstructionPayload:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is not configured",
        )

    client = Groq(api_key=api_key)

    system_prompt = """
You are an assistant that converts Spanish voice commands
into structured task-management instructions.

Return ONLY valid JSON with this structure:
{
  "endpoint": "/tasks",
  "method": "POST",
  "params": {}
}

Supported methods:
- GET: list tasks
- POST: create a task
- PUT: replace a task
- PATCH: update a task
- DELETE: delete a task

For task creation, use:
{
  "title": "task title",
  "done": false
}

For updates and deletions, include:
{
  "task_id": 1
}

For partial updates, include only the fields that must change.
Do not include Markdown or additional explanations.
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            ),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": payload.transcription,
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response from Groq")

        instruction = json.loads(content)

        return InstructionPayload(**instruction)

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Instruction processing failed: {error}",
        ) from error
