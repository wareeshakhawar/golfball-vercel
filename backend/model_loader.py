import torch
from ultralytics import YOLO
from typing import Optional
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_model(model_path=None):
    try:
        # Use environment variable if no path provided
        model_path = model_path or os.getenv('MODEL_PATH', 'model/best.pt')
        logger.info(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return None
            
        model = YOLO(model_path)
        logger.info("Model loaded successfully with default settings")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def load_model_with_alternative_method(model_path: str) -> Optional[YOLO]:
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found at: {os.path.abspath(model_path)}")
            return None

        print(f"Attempting to load model from: {os.path.abspath(model_path)}")
        
        # Try alternative loading method
        print("Attempting alternative loading method...")
        try:
            ckpt = torch.load(model_path, weights_only=False, map_location='cpu')
            model = YOLO(task='detect')
            model.model.load_state_dict(ckpt['model'].state_dict())
            print("Model loaded successfully with alternative method")
            return model
        except Exception as e2:
            print(f"Failed alternative loading method: {str(e2)}")
            return None
                
    except Exception as e:
        print(f"Unexpected error loading model: {str(e)}")
        return None 