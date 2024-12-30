import base64
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from processing import ImageProcessor 
from PIL import Image, UnidentifiedImageError
import io
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Path to the model
GENERATOR_PATH = "Colourized_model.pth"

# Initializing the image processor
try:
    processor = ImageProcessor(generator_path=GENERATOR_PATH)
    logger.info("Image processor initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize the image processor: {str(e)}")
    raise RuntimeError(f"Failed to initialize the image processor: {str(e)}")

# Serve the page
@app.get("/")
async def serve_page():
    return templates.TemplateResponse("pages.html", {"request": {}})

# Handliing image colorization
@app.post("/colorize")
async def colorize_image(image: UploadFile):
    try:
        # Validate file extension
        if not image.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            logger.warning(f"Invalid file type uploaded: {image.filename}")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .png, .jpg, or .jpeg file.")

        # Loading the uploaded image
        try:
            input_image = Image.open(io.BytesIO(image.file.read()))
            logger.info(f"Image {image.filename} loaded successfully.")
        except UnidentifiedImageError:
            logger.error(f"Uploaded file is not a valid image: {image.filename}")
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        #colorization
        try:
            colorized_image = processor.colorize_image(input_image)
            logger.info("Image colorization completed successfully.")
        except Exception as e:
            logger.error(f"Error during image processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during image processing: {str(e)}")

        # Converting the colorized image to base64
        buffered = io.BytesIO()
        colorized_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return JSON response with base64 image data
        return {"message": "Image colorized successfully!", "image_data": img_str}

    except HTTPException as e:
        # Handling HTTP exceptions
        logger.error(f"HTTP exception: {str(e.detail)}")
        raise e
    except Exception as e:
        # Handling unexpected errors
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
