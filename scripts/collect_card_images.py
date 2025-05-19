import cv2
import os
import argparse
from datetime import datetime

def collect_images(label, output_dir="data/front_images", camera_id=0):
    save_dir = os.path.join(output_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print(f"📸 Press SPACE to capture '{label}' image. Press ESC to exit.")

    counter = len(os.listdir(save_dir))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Show the frame
        # Make a copy for display
        preview = frame.copy()
        cv2.putText(preview, f"Label: {label} | Saved: {counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Card Capture", preview)

        key = cv2.waitKey(1)

        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACE to save image
            filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(save_dir, filename)
            cv2.imwrite(path, frame)
            print(f"✅ Saved {path}")
            counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True,
                        help="Card label (e.g. 'a_c', 'k_s')")
    parser.add_argument("--output_dir", type=str, default="data/front_images",
                        help="Directory to save images")
    args = parser.parse_args()

    collect_images(label=args.label, output_dir=args.output_dir)
