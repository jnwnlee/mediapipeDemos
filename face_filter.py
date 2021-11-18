import cv2
import mediapipe as mp
import numpy as np

from videosource import WebcamSource

from custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

# uncomment next line to use all points for PnP algorithm
# points_idx = list(range(0,468)); points_idx[0:2] = points_idx[0:2:-1];

frame_height, frame_width, channels = (720, 1280, 3)

# pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)

dist_coeff = np.zeros((4, 1))


def main():
    source = WebcamSource()

    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.1,
    ) as face_mesh:

        for idx, (frame, frame_rgb) in enumerate(source):
            results = face_mesh.process(frame)
            multi_face_landmarks = results.multi_face_landmarks

            if multi_face_landmarks:
                face_landmarks = multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )

                convexhull = cv2.convexHull(np.array(
                    [(lm.x*frame_width, lm.y*frame_height) for lm in face_landmarks.landmark]
                , dtype=np.int32))

                # print(landmarks.shape)
                landmarks = landmarks.T

                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
                )
                model_points = metric_landmarks[0:3, points_idx].T
                image_points = (
                    landmarks[0:2, points_idx].T
                    * np.array([frame_width, frame_height])[None, :]
                )

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    dist_coeff,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                (nose_end_point2D, jacobian) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 25.0)]),
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    dist_coeff,
                )

                # for face_landmarks in multi_face_landmarks:
                #     mp_drawing.draw_landmarks(
                #         image=frame,
                #         landmark_list=face_landmarks,
                #         connections=mp_face_mesh.FACE_CONNECTIONS,
                #         landmark_drawing_spec=drawing_spec,
                #         connection_drawing_spec=drawing_spec,
                #     )

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                frame = cv2.line(frame, p1, p2, (255, 0, 0), 2)

                # 2. Face blurrying
                mask = np.zeros((frame_height, frame_width), np.uint8)
                # cv2.polylines(mask, [convexhull], True, 255, 3)
                cv2.fillConvexPoly(mask, convexhull, 255)
                # Extract the face
                frame_copy = frame.copy()
                #frame_copy = cv2.blur(frame_copy, (27, 27))
                frame_copy = cv2.bitwise_not(frame_copy)
                face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
                # Extract background
                background_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(frame, frame, mask=background_mask)
                # Final result
                result = cv2.add(background, face_extracted)
            #source.show(frame)
            #cv2.imshow('face', face_extracted)
            #cv2.imshow('back', background)
            cv2.imshow('result', result)
            # source.show(frame_copy)


if __name__ == "__main__":
    main()
