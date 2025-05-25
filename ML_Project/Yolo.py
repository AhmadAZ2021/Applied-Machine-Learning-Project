from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO(r"<Yolo Model>")  

    
    model.train(
        data=r'<Yaml file Path>',
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
    multiprocessing.set_start_method('spawn')  
    main()
