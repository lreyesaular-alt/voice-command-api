import os

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from groq import Groq

from src.app.api.routes.instruction import route_instruction
from src.app.api.routes.tasks import (
    create_task,
    delete_task,
    get_tasks,
    replace_task,
    update_task,
)
from src.app.schemas.voice import (
    InstructionRequest,
    TaskCreate,
    TaskReplace,
    TaskUpdate,
    TranscribeFlowResponse,
)

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(
    file: UploadFile = File(...),
) -> TranscribeFlowResponse:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is not configured",
        )

    try:
        audio_content = await file.read()

        client = Groq(api_key=api_key)

        transcription_response = client.audio.transcriptions.create(
            file=(
                file.filename or "audio.webm",
                audio_content,
            ),
            model="whisper-large-v3-turbo",
            response_format="text",
            language="es",
        )

        transcription = str(transcription_response).strip()

        if not transcription:
            raise ValueError("Empty transcription from Groq")

        instruction = route_instruction(
            InstructionRequest(transcription=transcription)
        )

        endpoint = instruction.endpoint
        method = instruction.method.upper()
        params = instruction.params

        if endpoint != "/tasks":
            raise ValueError(
                f"Unsupported endpoint: {endpoint}"
            )

        if method == "GET":
            result = get_tasks()

        elif method == "POST":
            result = create_task(
                TaskCreate(**params)
            )

        elif method == "PUT":
            task_id = params.pop("task_id")

            result = replace_task(
                task_id=task_id,
                payload=TaskReplace(**params),
            )

        elif method == "PATCH":
            task_id = params.pop("task_id")

            result = update_task(
                task_id=task_id,
                payload=TaskUpdate(**params),
            )

        elif method == "DELETE":
            task_id = params["task_id"]

            result = delete_task(task_id)

        else:
            raise ValueError(
                f"Unsupported method: {method}"
            )

        return TranscribeFlowResponse(
            transcription=transcription,
            instruction=instruction,
            result=result,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription flow failed: {error}",
        ) from error