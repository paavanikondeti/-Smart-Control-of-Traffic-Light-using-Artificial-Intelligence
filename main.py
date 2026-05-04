import cv2

# load video
video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("Error opening video")
    exit()

# save output video
out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      20, (800, 500))

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=40)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))

    # -------- DETECTION --------
    mask = fgbg.apply(frame)
    _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    car_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 400:   # reduces noise
            x, y, w, h = cv2.boundingRect(cnt)

            if w > 40 and h > 40:   # avoid small false boxes
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                car_count += 1

    # -------- SIGNAL LOGIC --------
    if car_count > 6:
        signal = "GREEN (LONG)"
        color = (0, 255, 0)
    elif car_count > 3:
        signal = "GREEN (MEDIUM)"
        color = (0, 255, 255)
    else:
        signal = "GREEN (SHORT)"
        color = (0, 0, 255)

    # -------- DISPLAY --------
    cv2.putText(frame, f"Vehicles: {car_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, signal, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, "Traffic Density Based Signal Control", (120, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # show
    cv2.imshow("Smart Traffic System", frame)

    # save output
    out.write(frame)

    if cv2.waitKey(30) == 27:
        break

video.release()
out.release()
cv2.destroyAllWindows()