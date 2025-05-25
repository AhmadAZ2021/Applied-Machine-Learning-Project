from ultralytics import YOLO

# Load your trained model (replace with your actual model path)
model = YOLO(r"C:\Users\20248106\Desktop\mapillary\runs\detect\yolo8s-traffic-sign7\weights\best.pt")  # or wherever your best.pt is

# 🔍 Inference on an image
img_path = r"C:\Users\20248106\Desktop\TestDataBatch2\0000261.jpg"  # replace with your image path
results = model(img_path, conf=0.1, show=True, device='cpu')  # show=True will open image with predictions in Colab or local window
results[0].save(filename='predicted_image.jpg')  # saves the output image with boxes

# # 🎥 Inference on a video
# video_path = 'C:/Users/Ahmad/Desktop/yolo test/traffic signs on German roads.mp4'  # replace with your video path
# results = model(video_path, save=True, show=True, device='cpu')  # auto-saves to 'runs/detect/predict'