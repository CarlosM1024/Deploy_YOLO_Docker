# FastAPI inference server

"""Primero, agregue las importaciones y configure el registro para Vertex AI. 
Debido a que Vertex AI trata stderr como salida de error, 
tiene sentido canalizar los registros a stdout."""

import sys
import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Any, Dict, Optional
from app import is_model_ready, run_inference, get_image_from_bytes, get_annotated_image, get_bytes_from_image
import uvicorn
import base64

from loguru import logger

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

# Create FastAPI app instance
app = FastAPI(title="YOLO11 Inference Server", version="1.0.0")

"""Para una conformidad completa con Vertex AI, 
defina los puntos finales requeridos en las variables de entorno y 
establezca el límite de tamaño para las solicitudes. 
Se recomienda utilizar puntos finales privados de Vertex AI para implementaciones de producción. 
De esta manera, tendrá un límite de carga útil de solicitud más alto 
(10 MB en lugar de 1,5 MB para los puntos finales públicos), 
junto con una seguridad robusta y control de acceso."""

# Vertex AI environment variables
AIP_HTTP_PORT = int(os.getenv("AIP_HTTP_PORT", "8080"))
AIP_HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")

# Request size limit (10 MB for private endpoints, 1.5 MB for public)
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB in bytes

#Agregue dos modelos Pydantic para validar sus solicitudes y respuestas:
# Pydantic models for request/response
class PredictionRequest(BaseModel):
    instances: list
    parameters: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    predictions: list


"""Agregue el punto final de verificación de estado para verificar la preparación de su modelo. 
Esto es importante para Vertex AI, ya que sin una verificación de estado dedicada, 
su orquestador hará ping a sockets aleatorios y no podrá determinar si el modelo está listo para la inferencia. 
Su verificación debe devolver 200 OK para éxito y 503 Service Unavailable para fallo:"""

# Health check endpoint
@app.get(AIP_HEALTH_ROUTE, status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for Vertex AI."""
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "healthy"}

"""Ahora tiene todo para implementar el punto final de predicción que gestionará las solicitudes de inferencia. 
Aceptará un archivo de imagen, ejecutará la inferencia y devolverá los resultados. 
Tenga en cuenta que la imagen debe estar codificada en base64, 
lo que aumenta adicionalmente el tamaño de la carga útil hasta en un 33%."""

@app.post(AIP_PREDICT_ROUTE, response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint for Vertex AI."""
    try:
        predictions = []

        for instance in request.instances:
            if isinstance(instance, dict):
                if "image" in instance:
                    image_data = base64.b64decode(instance["image"])
                    input_image = get_image_from_bytes(image_data)
                else:
                    raise HTTPException(status_code=400, detail="Instance must contain 'image' field")
            else:
                raise HTTPException(status_code=400, detail="Invalid instance format")

            # Extract YOLO11 parameters if provided
            parameters = request.parameters or {}
            confidence_threshold = parameters.get("confidence", 0.5)
            return_annotated_image = parameters.get("return_annotated_image", False)

            # Run inference with YOLO11n model
            result = run_inference(input_image, confidence_threshold=confidence_threshold)
            detections_list = result["detections"]

            # Format predictions for Vertex AI
            detections = []
            for detection in detections_list:
                formatted_detection = {
                    "class": detection["name"],
                    "confidence": detection["confidence"],
                    "bbox": {
                        "xmin": detection["xmin"],
                        "ymin": detection["ymin"],
                        "xmax": detection["xmax"],
                        "ymax": detection["ymax"],
                    },
                }
                detections.append(formatted_detection)

            # Build prediction response
            prediction = {"detections": detections, "detection_count": len(detections)}

            # Add annotated image if requested and detections exist
            if (
                return_annotated_image
                and result["results"]
                and result["results"][0].boxes is not None
                and len(result["results"][0].boxes) > 0
            ):

                annotated_image = get_annotated_image(result["results"])
                img_bytes = get_bytes_from_image(annotated_image)
                prediction["annotated_image"] = base64.b64encode(img_bytes).decode("utf-8")

            predictions.append(prediction)

        logger.info(
            f"Processed {len(request.instances)} instances, found {sum(len(p['detections']) for p in predictions)} total detections"
        )

        return PredictionResponse(predictions=predictions)

    except HTTPException:
        # Re-raise HTTPException as-is (don't catch and convert to 500)
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

#Finalmente, agregue el punto de entrada de la aplicación para ejecutar el servidor FastAPI.

if __name__ == "__main__":

    logger.info(f"Starting server on port {AIP_HTTP_PORT}")
    logger.info(f"Health check route: {AIP_HEALTH_ROUTE}")
    logger.info(f"Predict route: {AIP_PREDICT_ROUTE}")
    uvicorn.run(app, host="0.0.0.0", port=AIP_HTTP_PORT)

