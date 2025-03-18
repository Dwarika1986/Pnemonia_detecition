import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 Classification Model
MODEL_PATH = "/Users/dwarikamohanty/Downloads/yolov8n-cls.pt"  # Ensure the path is correct
model = YOLO(MODEL_PATH)

def index(request):  
    """ Renders the home page with an image upload form. """
    return render(request, 'pneumonia/index.html')

def predict(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # ✅ Ensure the media directory exists
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)

        # ✅ Save uploaded image correctly in MEDIA folder
        file_name = uploaded_file.name
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        # ✅ Store file
        saved_path = default_storage.save(file_name, ContentFile(uploaded_file.read()))

        # ✅ Load image for inference
        img = Image.open(os.path.join(settings.MEDIA_ROOT, saved_path))

        # ✅ Perform classification using YOLOv8
        results = model(img)
        predicted_class = "Unknown"  # Default value

        if hasattr(results[0], 'probs') and hasattr(results[0].probs, 'top1'):
            predicted_class_idx = results[0].probs.top1  
            predicted_class = results[0].names.get(predicted_class_idx, "Unknown")  # Use `.get()` to avoid KeyError

        return render(request, 'pneumonia/index.html', {
            'predicted_class': predicted_class,
            'image_url': f'{settings.MEDIA_URL}{saved_path}'  # ✅ Serve media correctly
        })

    return render(request, 'pneumonia/index.html', {'error': 'No file uploaded'})  

