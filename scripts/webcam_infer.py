import cv2
import torch
import argparse
import numpy as np
from torchvision import transforms
from cardnet.model import build_resnet18
from cardnet.utils import load_class_names
from ultralytics import YOLO

def find_card_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_candidate = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter: must be 4-sided and large enough
        if len(approx) == 4 and cv2.contourArea(contour) > 10000:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Card held vertical: ~0.7, horizontal: ~1.4
            if 0.6 < aspect_ratio < 0.75 or 1.3 < aspect_ratio < 1.7:
                best_candidate = approx
                break  # Stop if we find a good one

    return best_candidate

def preprocess_frame(frame, input_size=224):
    # Center crop square + resize + normalize
    # h, w, _ = frame.shape
    # min_dim = min(h, w)
    # start_x = (w - min_dim) // 2
    # start_y = (h - min_dim) // 2
    # cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    # resized = cv2.resize(cropped, (input_size, input_size))
    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # ✅ BGR → RGB fix

    resized = frame

    debug_img = resized.astype(np.uint8)
    cv2.imshow("Debug - Preprocessed RGB Input", debug_img)

    # Convert to tensor
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((224, 224)),
    ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ResNet ImageNet
    #                          std=[0.229, 0.224, 0.225])
    # ])
    tensor = transform(resized).unsqueeze(0)  # shape: [1, 3, 224, 224]
    return tensor

def draw_prediction(frame, text):
    h, w, _ = frame.shape
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Draw center square (ROI)
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    end_x = start_x + min_dim
    end_y = start_y + min_dim
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

def main(model_path, class_names_path):
    # Load model
    class_names = load_class_names(class_names_path)
    num_classes = len(class_names)
    model = build_resnet18(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # YOLO detector
    detector = YOLO("/home/alex/Documents/projects/label_studio/runs/obb/train/weights/best.pt")

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO prediction
        # Run YOLO detection
        results = detector(frame, verbose=False)
        # print(f'number of results: {len(results)}')

        if len(results) > 0:
          for i, result in enumerate(results):
            # print(i)
            # print(result)
            if result.obb.xyxy.nelement() > 0:
              x1, y1, x2, y2 = result.obb.xyxy[0].int().tolist()
              confidence = result.obb.conf[0].item()
              print(f"  [{i}] Box: ({x1},{y1}) → ({x2},{y2}) | conf: {confidence:.2f}")

              # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
              # cv2.putText(frame, f"card ({confidence:.2f})", (x1, y1 - 10),
              #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

              # Crop ROI
              roi = frame[y1:y2, x1:x2]
              input_tensor = preprocess_frame(roi).to(device)

              # Run classifier
              with torch.no_grad():
                  output = model(input_tensor)
                  pred = output.argmax(dim=1).item()
                  pred_class = class_names[pred]

              # Annotate frame
              label = f"{pred_class} ({confidence:.2f})"
              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
              cv2.putText(frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No card detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Card Classifier (YOLO + ResNet)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Preprocess + predict
        # input_tensor = preprocess_frame(frame).to(device)
        # contour = find_card_contour(frame)
        # if contour is not None:
        #     x, y, w, h = cv2.boundingRect(contour)
            
        #     roi = frame[y:y+h, x:x+w]
        #     input_tensor = preprocess_frame(roi).to(device)

        #     cv2.imshow("Model Input (ROI)", roi)

        #     with torch.no_grad():
        #         output = model(input_tensor)
        #         pred = output.argmax(dim=1).item()
        #         pred_class = class_names[pred]

        #     # Draw the box
        #     cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        #     draw_prediction(frame, f"Prediction: {pred_class}")
        # else:
        #     draw_prediction(frame, "No card detected")

        # with torch.no_grad():
        #     output = model(input_tensor)
        #     pred = output.argmax(dim=1).item()
        #     pred_class = class_names[pred]

        # Show on frame
        # draw_prediction(frame, f"Prediction: {pred_class}")
        cv2.imshow("Card Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--classes", type=str, required=True, help="Path to class_names.txt")
    args = parser.parse_args()

    main(args.model, args.classes)
