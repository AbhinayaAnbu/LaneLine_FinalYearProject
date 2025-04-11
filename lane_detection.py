import cv2
import numpy as np

def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(100, height), (1180, height), (700, 400), (600, 400)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0: slope = 0.1  # Avoid divide by zero
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_line = make_coordinates(image, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(image, np.mean(right_fit, axis=0)) if right_fit else None
    return np.array([left_line, right_line])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def draw_lane_fill(image, lines):
    fill_image = np.zeros_like(image)
    if lines[0] is not None and lines[1] is not None:
        pts = np.array([[
            (lines[0][0], lines[0][1]),
            (lines[0][2], lines[0][3]),
            (lines[1][2], lines[1][3]),
            (lines[1][0], lines[1][1])
        ]], dtype=np.int32)
        cv2.fillPoly(fill_image, pts, (0, 255, 255))  # Yellow fill
    return fill_image

def main():
    cap = cv2.VideoCapture('test2.mp4')  # Your test video file

    # Optional: Save output
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter('lane_detected_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        canny = canny_edge(frame)
        cropped = region_of_interest(canny)
        lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        fill_image = draw_lane_fill(frame, averaged_lines)

        combo = cv2.addWeighted(frame, 0.8, fill_image, 0.4, 1)
        combo = cv2.addWeighted(combo, 1, line_image, 1, 1)

        cv2.imshow("Lane Detection", combo)
        out.write(combo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
