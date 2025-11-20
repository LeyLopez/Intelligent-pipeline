import requests
import os
from typing import Dict, Tuple
import io
from PIL import Image

LLM_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
SKLEARN_URL = os.getenv("SKLEARN_SERVICE_URL", "http://localhost:8002")
CNN_URL = os.getenv("CNN_SERVICE_URL", "http://localhost:8003")


def check_service_health()-> Dict[str, bool]:
    services = {
        "LLM": LLM_URL,
        "Sklearn": SKLEARN_URL,
        "CNN": CNN_URL
    }

    health_status = {}
    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            health_status[name] = response.status_code == 200
        except requests.exceptions.RequestException:
            health_status[name] = False

    return health_status



def chat_with_llm(message:str, history:list)-> str:

    try:
        response = requests.post(
            f"{LLM_URL}/chat",
            json={
                "prompt": message,
                "context": None if not history else history,
                "max_tokens": 500
            },
            timeout=10
        )


        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Error: No response from LLM.")
        else:
            return f"Error: {response.status_code} - {response.text}"
        
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to LLM service."
    except Exception as e:
        return f"Error: {str(e)}"
    


def predict_with_sklearn(
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
)->str:
    try:
        features = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width
        }

        response = requests.post(
            f"{SKLEARN_URL}/predict",
            json={"features": features},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()

            output = output = f"""
### Prediction: {result['prediction_label']}

**Probabilities:**
"""
            for i, prob in enumerate(result['probability']):
                output += f"\n- Class {i}: {prob:.2%}"
            
            output += f"\n\n**Confidence:** {max(result['probability']):.2%}"
            output += f"\n**Status:** {result['status']}"
            
            return output
        else:
            return f" Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to Sklearn service."
    except Exception as e:
        return f"Error: {str(e)}"
    



def classify_image(image) -> Tuple[str, Dict]:

    try:

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        

        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(
            f"{CNN_URL}/classify",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            output = f"""
### Classification Result

**Prediction:** {result['predicted_class']}
**Confidence:** {result['confidence']:.2%}

**All Probabilities:**
"""
            for class_name, prob in result['all_probabilities'].items():
                output += f"\n- {class_name}: {prob:.2%}"
            
            output += f"\n\n### Limitations\n{result['limitations']}"
            
            return output, result['all_probabilities']
        else:
            return f"Error: {response.status_code}", {}
            
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to CNN service.", {}
    except Exception as e:
        return f"Error: {str(e)}", {}




def get_model_info() -> str:

    try:

        response = requests.get(f"{CNN_URL}/info", timeout=5)

        cnn_info = response.json() if response.status_code == 200 else {}
        
        info_text = """
# Sistem information

## Services status
"""
        
        health = check_service_health()
        for service, status in health.items():
            emoji = "OK" if status else "ERROR"
            info_text += f"\n{emoji} **{service}**: {'Active' if status else 'Inactive'}"
        
        if cnn_info:
            info_text += f"""

## CNN Model

**Supported classes:** {', '.join(cnn_info.get('capabilities', {}).get('supported_classes', []))}

**Available filters:** {', '.join(cnn_info.get('capabilities', {}).get('filters', []))}

### Important Limitations:
"""
            for limitation in cnn_info.get('limitations', {}).get('restrictions', []):
                info_text += f"\n- {limitation}"
        
        info_text += """

## Sklearn Model

**Type:** Random Forest Classifier
**Dataset:** Iris (3 classes)
**Features:** 4 features

## LLM Model

**Provider:** Ollama
**Model:** LLaMA 2
**Capability:** General and contextual conversation
"""
        
        return info_text
        
    except Exception as e:
        return f"Error obtaining information: {str(e)}"


