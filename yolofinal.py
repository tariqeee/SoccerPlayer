import cv2
import os
import time
from ultralytics import YOLO

model = YOLO("best.pt")

# Initialize metrics variables
tp, fp, fn, total_frames, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

# Initialize variables for calculating precision, recall, and F1 score
total_frames = 0
precision = 0
recall = 0
f1_score = 0

vid = cv2.VideoCapture("shortvideo.mp4")

# Create output and frames directory if they don't exist
for dir_name in ['output', 'frames']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



# VideoWriter setup
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_video = cv2.VideoWriter('output/output_video.avi', fourcc, 20.0, (1080, 720))


# Frame directory setup
frame_count = 0
def drawBoxes(frame, results):
    global tp, fp, total_frames


    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confi = results[0].boxes.conf.cpu().numpy()
    names = results[0].boxes.cls.cpu().numpy()

    for box, name, conf in zip(boxes, names, confi):
        cls = results[0].names[name]
        confidence = conf
        text = f"{cls} {confidence:.2f}"

        if confidence > 0.5 and cls == "Player":
            tp += 1  # For the purpose of this example, treating every detection as TP.

        color = (0, 0, 255)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(frame, p1, p2, color, 2)

        tf = 1
        w, h = cv2.getTextSize(text, 0, fontScale=tf / 3, thickness=1)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, tf / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

    total_frames += 1
    return frame


start_time = time.time()  
while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        print("Error reading frame or video ended.")
        break

    try:
        height, width, _ = frame.shape
        if width < 1080 or height < 720:
            frame = cv2.resize(frame, (1080, 720))
        res = model.predict(frame, conf=0.3, imgsz=1088)  # Adjusted to multiple of 32
        final_IMG = drawBoxes(frame, res)

        elapsed_time = time.time() - start_time
        current_fps = 1 / elapsed_time
        start_time = time.time()
        cv2.putText(final_IMG, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
        # Display the resulting frame in real time
        cv2.imshow('Result', final_IMG)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # Save the frame as an image
        cv2.imwrite(os.path.join('frames', f'frame_{frame_count}.jpg'), final_IMG)
        frame_count += 1



        # Write the frame to the output video
        out_video.write(final_IMG)
        
        # Save the frame as an image
        cv2.imwrite(os.path.join('frames', f'frame_{frame_count}.jpg'), final_IMG)
        frames_dir = 'frames'


    except cv2.error as e:  
        print("OpenCV Error:", e)
        print("Shape of original frame:", frame.shape)
        print("Shape of final_IMG:", final_IMG.shape)
    except Exception as e:  
        print(f"Error occurred: {e}")

# After the while loop ends
vid.release()
out_video.release()
cv2.destroyAllWindows()




# Compute metrics
if tp + fp > 0:
    precision = tp / (tp + fp)
if tp + fn > 0:
    recall = tp / (tp + fn)
if precision + recall > 0:
    f1_score = 2 * precision * recall / (precision + recall)
fn = fp  # As an example, equating FP to FN.

print(f"Total number of frames tested: {total_frames}")
print(f"True positives: {tp}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 score: {f1_score:.2%}")
