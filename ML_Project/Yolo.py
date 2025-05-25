from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO(r"C:\Users\20248106\Desktop\mapillary\runs\detect\yolo8s-traffic-sign6\weights\best.pt")  # Use yolov8m.pt or yolov8l.pt for stronger models if GPU allows

    # Train the model
    model.train(
        data=r'C:\Users\20248106\Desktop\mapillary\dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=32,  
        optimizer='AdamW',
        patience=10,
        save=True,
        save_period=10,  # Save every 10 epochs
        name='yolo8s-traffic-sign'  # Will save checkpoints like: yolo8s-traffic-sign/weights/epoch10.pt
    )

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Required for Windows
    main()
