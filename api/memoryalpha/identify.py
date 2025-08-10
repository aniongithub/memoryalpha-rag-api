from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
import tempfile
import os
from .rag import MemoryAlphaRAG

router = APIRouter()

# Singleton or global instance for demo; in production, manage lifecycle properly
rag_instance = MemoryAlphaRAG()

@router.post("/memoryalpha/rag/identify", summary="Multimodal Image Search")
def identify_endpoint(
    file: UploadFile = File(...),
    top_k: int = Query(5, description="Number of results to return")
):
    """
    Accepts an image file upload, performs multimodal image search, and returns results.
    """
    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            tmp.write(file.file.read())
            image_path = tmp.name
        # Perform image search
        results = rag_instance.search_image(image_path, top_k=top_k)
        # Clean up temp file
        os.remove(image_path)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
