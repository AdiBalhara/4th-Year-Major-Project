from ultralytics import YOLO

def predict(image_path):
    model = YOLO("models/yolo_best.pt")
    results = model(image_path, save=True)
    return results

if __name__ == "__main__":
    predict("sample.jpg")
