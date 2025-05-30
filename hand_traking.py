import mediapipe as mp
import cv2
import numpy as np




def detect_hand_keypoints(image):
    """
    Detects hand keypoints on a single image and displays the drawn image.
    Returns the 2D coordinates of the thumb tip (point 4) and thumb IP joint (point 3),
    along with the hand name (Left or Right).

    :param image: Input image as a NumPy array.
    :return: Dictionary with coordinates of thumb tip (4) and thumb IP joint (3),
             along with the hand name (Left or Right).
    """
    # Initialize Mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, 
                           min_detection_confidence=0.1)
    mp_draw = mp.solutions.drawing_utils

    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    keypoints = {}

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_label = handedness.classification[0].label  # "Left" or "Right"

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            thumb_ip_x, thumb_ip_y = int(thumb_ip.x * w), int(thumb_ip.y * h)

            keypoints[idx] = {
                "hand": hand_label,
                "thumb_tip": (thumb_tip_x, thumb_tip_y),
                "thumb_ip": (thumb_ip_x, thumb_ip_y)
            }
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        print("No hand detected.")

    # Display the image with drawn landmarks.
    cv2.imshow("Hand Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return keypoints



# def detect_hand_keypoints(image):
#     """
#     Detects hand keypoints on a single image and displays the drawn image.
#     Returns the 2D coordinates of the thumb tip (point 4) and thumb IP joint (point 3).

#     :param image_path: Path to the image file.
#     :return: Dictionary with coordinates of thumb tip (4) and thumb IP joint (3).
#     """
#     # Initialize Mediapipe hands module
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, 
#                            min_detection_confidence=0.1)
#     mp_draw = mp.solutions.drawing_utils

#     h, w, _ = image.shape
#     img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)

#     keypoints = {}

#     if results.multi_hand_landmarks:
#         for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#             thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

#             thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
#             thumb_ip_x, thumb_ip_y = int(thumb_ip.x * w), int(thumb_ip.y * h)

#             keypoints[idx] = {
#                 "thumb_tip": (thumb_tip_x, thumb_tip_y),
#                 "thumb_ip": (thumb_ip_x, thumb_ip_y)
#             }
#             mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     else:
#         print("No hand detected.")

#     # Display the image with drawn landmarks.
#     cv2.imshow("Hand Keypoints", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return keypoints



def sample_points_on_line(p1, p2, mask, n=100):
    """
    Sample all points between p1 and p2 and extrapolate up to 'n' pixels on each side.
    If extrapolation moves out of the mask, it stops before going out.

    :param p1: Tuple (x1, y1) - Start point
    :param p2: Tuple (x2, y2) - End point
    :param mask: Binary mask (H, W) where valid regions are >0
    :param n: Max pixels to extrapolate beyond p1 and p2 within the mask
    :return: List of valid points within the mask and the computed contact point
    """
    h, w = mask.shape[:2]
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    d = p2 - p1  # Direction vector

    if np.allclose(d, 0):  # If both points are the same
        contact_point = (int(round(p1[0])), int(round(p1[1])))
        valid_points = [contact_point] if mask[contact_point[1], contact_point[0]] > 0 else []
        return valid_points, contact_point

    # Compute segment points (including p1 and p2)
    length = int(np.linalg.norm(d))
    x_vals = np.linspace(p1[0], p2[0], length + 1)
    y_vals = np.linspace(p1[1], p2[1], length + 1)
    all_points = [(int(round(x)), int(round(y))) for x, y in zip(x_vals, y_vals)]

    # Compute extrapolated points on both sides
    d_unit = d / np.linalg.norm(d)  # Unit direction vector
    p1_extrapolated = []
    p2_extrapolated = []

    for i in range(1, n + 1):
        p1_ext = p1 - i * d_unit
        p2_ext = p2 + i * d_unit
        p1_ext = (int(round(p1_ext[0])), int(round(p1_ext[1])))
        p2_ext = (int(round(p2_ext[0])), int(round(p2_ext[1])))

        # Stop extrapolation if outside mask
        if 0 <= p1_ext[0] < w and 0 <= p1_ext[1] < h and mask[p1_ext[1], p1_ext[0]] > 0:
            p1_extrapolated.append(p1_ext)
        else:
            break

        if 0 <= p2_ext[0] < w and 0 <= p2_ext[1] < h and mask[p2_ext[1], p2_ext[0]] > 0:
            p2_extrapolated.append(p2_ext)
        else:
            break

    # Keep only points that fall inside the segmentation mask
    valid_points = [pt for pt in all_points if mask[pt[1], pt[0]] > 0]
    valid_points = p1_extrapolated + valid_points + p2_extrapolated  # Order: left extra, middle, right extra

    # Compute contact point as the midpoint of p1 and p2
    contact_point = (int(round((p1[0] + p2[0]) / 2)), int(round((p1[1] + p2[1]) / 2)))

    return valid_points



# def sample_points_on_line(p1, p2, mask):
#     """
#     Sample 'n' points along the line segment between p1 and p2.
    
#     :param p1: Tuple (x1, y1) - Start point
#     :param p2: Tuple (x2, y2) - End point
#     :param n: Number of points to sample (including endpoints)
#     :return: List of (x, y) coordinates of sampled points
#     """
#     # x_vals = np.linspace(p1[0], p2[0], n, dtype=int)
#     # y_vals = np.linspace(p1[1], p2[1], n, dtype=int)
    
#     # return list(zip(x_vals, y_vals))
#     h, w = mask.shape[:2]
#     p1 = np.array(p1, dtype=float)
#     p2 = np.array(p2, dtype=float)
#     d = p2 - p1
#     if np.allclose(d, 0):
#         contact_point = (int(round(p1[0])), int(round(p1[1])))
#         valid_points = [contact_point] if mask[contact_point[1], contact_point[0]] > 0 else []
#         return valid_points, contact_point

#     # Find intersections of the infinite line with the image boundaries.
#     ts = []

#     # Left boundary (x=0)
#     if d[0] != 0:
#         t = (0 - p1[0]) / d[0]
#         y = p1[1] + t * d[1]
#         if 0 <= y <= h - 1:
#             ts.append(t)
#     # Right boundary (x=w-1)
#     if d[0] != 0:
#         t = ((w - 1) - p1[0]) / d[0]
#         y = p1[1] + t * d[1]
#         if 0 <= y <= h - 1:
#             ts.append(t)
#     # Top boundary (y=0)
#     if d[1] != 0:
#         t = (0 - p1[1]) / d[1]
#         x = p1[0] + t * d[0]
#         if 0 <= x <= w - 1:
#             ts.append(t)
#     # Bottom boundary (y=h-1)
#     if d[1] != 0:
#         t = ((h - 1) - p1[1]) / d[1]
#         x = p1[0] + t * d[0]
#         if 0 <= x <= w - 1:
#             ts.append(t)

#     if len(ts) < 2:
#         # If we don't get two intersections, fall back to the original points.
#         t_min, t_max = 0, 1
#     else:
#         t_min, t_max = min(ts), max(ts)

#     # Extended endpoints along the line within the image boundaries.
#     p_start = p1 + t_min * d
#     p_end = p1 + t_max * d

#     # Determine the number of pixels between these two endpoints.
#     length = int(np.linalg.norm(p_end - p_start))
#     if length == 0:
#         point = (int(round(p_start[0])), int(round(p_start[1])))
#         valid_points = [point] if mask[point[1], point[0]] > 0 else []
#         contact_point = (int(round((p1[0] + p2[0]) / 2)), int(round((p1[1] + p2[1]) / 2)))
#         return valid_points, contact_point

#     # Sample all points along the line from p_start to p_end.
#     x_vals = np.linspace(p_start[0], p_end[0], length + 1)
#     y_vals = np.linspace(p_start[1], p_end[1], length + 1)
#     all_points = [(int(round(x)), int(round(y))) for x, y in zip(x_vals, y_vals)]
#     # Remove any duplicate points caused by rounding.
#     all_points = list(dict.fromkeys(all_points))
    
#     # Keep only the points that fall inside the segmentation mask.
#     valid_points = [pt for pt in all_points if mask[pt[1], pt[0]] > 0]

#     # Calculate the contact point as the midpoint between the original two points.
#     contact_point = (int(round((p1[0] + p2[0]) / 2)), int(round((p1[1] + p2[1]) / 2)))

    
#     return valid_points