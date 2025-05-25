from ultralytics import YOLO


model = YOLO(r"<Model Path>") 

# Inference on an image
img_path = r"<Image Path>"  # replace with your image path
results = model(img_path, conf=0.1, show=True, device='cpu') 
results[0].save(filename='predicted_image.jpg')  

# Inference on a video
# video_path = '<Video Path>'  
# results = model(video_path, save=True, show=True, device='cpu')  
