# 🖼️ CNN Image Classification Service

## Descripción
Servicio de clasificación de imágenes usando Redes Neuronales Convolucionales.

## Características
- Clasificación de 3 clases: perro, gato, ave
- 4 capas convolucionales
- Filtros: Gaussian Blur, Edge Detection, Sharpening, Emboss
- Registro en MLflow

## Limitaciones
**IMPORTANTE**: El modelo SOLO reconoce 3 clases específicas.
- God
- Cat  
- Bird

## Endpoints

### POST /classify
Clasifica una imagen.

**Request:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8003/classify
```

### POST /apply-filters
Aplica filtros convolucionales.

### GET /info
Muestra capacidades y limitaciones.

## Ejecución
```bash
pip install -r requirements.txt
python -m app.main
```