import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        # Green
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Yellow
        lower_yellow = np.array([193, 193, 0])
        upper_yellow = np.array([255, 255, 0])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Pink
        lower_pink = np.array([140, 50, 50])
        upper_pink = np.array([170, 255, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Blue
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Combine all masks
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, pink_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"centroid: ({cx}, {cy})", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show the result
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
