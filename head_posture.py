import cv2
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt

from videosource import WebcamSource

from custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

LEFT_EYE = []
RIGHT_EYE = []
LIPS = []

for x, y in mp_face_mesh.FACEMESH_LEFT_EYE:
    LEFT_EYE.append(x)
    LEFT_EYE.append(y)

for x, y in mp_face_mesh.FACEMESH_RIGHT_EYE:
    RIGHT_EYE.append(x)
    RIGHT_EYE.append(y)

for x, y in mp_face_mesh.FACEMESH_LIPS:
    LIPS.append(x)
    LIPS.append(y)

LEFT_EYE = set(LEFT_EYE)
RIGHT_EYE = set(RIGHT_EYE)
LIPS = set(LIPS)


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



def plotting(points, eyes, lips, eye, lip):
    fig = plt.figure(figsize=(5, 12))
    ax = fig.add_subplot(111, projection='3d')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    ax.scatter(eyes[:, 0], eyes[:, 1], eyes[:, 2], c='blue', s=3)
    ax.scatter(lips[:, 0], lips[:, 1], lips[:, 2], c='blue', s=3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=3)

    ax.scatter(eye[0], eye[1], eye[2], c='green', s=10)
    ax.scatter(lip[0], lip[1], lip[2], c='red', s=10)
    ax.set_zlim(0, 1)

    plt.show()
    exit()

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    # source = WebcamSource(width=frame_width, height=frame_height)

    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        distances = []

        # for idx, (frame, frame_rgb) in enumerate(source):
        while cap.isOpened():
            ret, frame = cap.read()
            assert ret, 'result not successfully returned.'
            results = face_mesh.process(frame)
            multi_face_landmarks = results.multi_face_landmarks

            if multi_face_landmarks:
                face_landmarks = multi_face_landmarks[0]

                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                landmarks_iris = landmarks[-10:, :]
                landmarks = landmarks[:-10, :]

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

                for face_landmarks in multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                frame = cv2.line(frame, p1, p2, (255, 0, 0), 2)

                # calculating euler angles
                rmat, jac = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                print('*' * 80)
                # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
                x = np.arctan2(Qx[2][1], Qx[2][2])
                y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
                z = np.arctan2(Qz[0][0], Qz[1][0])
                print("ThetaX: ", x)
                print("ThetaY: ", y)
                print("ThetaZ: ", z)

                landmark = landmarks.T

                left_eye_list = []
                right_eye_list = []
                lips_list = []
                for i in LEFT_EYE:
                    left_eye_list.append([landmark[i][0], landmark[i][1], landmark[i][2]])
                for i in RIGHT_EYE:
                    right_eye_list.append([landmark[i][0], landmark[i][1], landmark[i][2]])
                for i in LIPS:
                    lips_list.append([landmark[i][0], landmark[i][1], landmark[i][2]])

                left_eye_arr = np.array(left_eye_list)
                right_eye_arr = np.array(right_eye_list)
                lips_arr = np.array(lips_list)

                eyes_arr = np.concatenate((left_eye_arr, right_eye_arr), axis=0)
                eyes_mean = np.mean(eyes_arr, axis=0)
                lips_mean = np.mean(lips_arr, axis=0)
                print('mean of eyes : ', eyes_mean, ' // mean of lips : ', lips_mean)

                iris_left_arr, iris_right_arr = landmarks_iris[:5], landmarks_iris[5:]
                iris_left_std_arr = np.std(iris_left_arr[1:, :], axis=0)
                iris_right_std_arr = np.std(iris_right_arr[1:, :], axis=0)
                iris_left_std = np.mean(iris_left_std_arr)
                iris_right_std = np.mean(iris_right_std_arr)
                iris_std = iris_left_std + iris_right_std
                # print(iris_left_arr)
                # print(iris_right_arr)
                distance = 1 / math.sqrt(iris_std)
                distances.append(distance)

                GAZE = "Distance Estimation : " + str(round(distance, 1))
                cv2.putText(frame, GAZE, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                print('*' * 80)
                if angles[1] < -15:
                    GAZE = "Looking: Left"
                elif angles[1] > 15:
                    GAZE = "Looking: Right"
                else:
                    GAZE = "Forward"
                GAZE += ' '+str(round(angles[1], 4))
                cv2.putText(frame, GAZE, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if angles[0] > -180+15 and angles[0] < 0:
                    GAZE = "Looking: Down"
                elif angles[0] < 180-15 and angles[0] > 0:
                    GAZE = "Looking: Up"
                else:
                    GAZE = "Forward"
                GAZE += ' '+str(round(angles[0], 4))
                cv2.putText(frame, GAZE, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if angles[2] < -15:
                    GAZE = "Tilting: Right"
                elif angles[2] > 15:
                    GAZE = "Tilting: Left"
                else:
                    GAZE = "Forward"
                GAZE += ' '+str(round(angles[2], 4))
                cv2.putText(frame, GAZE, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # plt.scatter(range(len(distances)), distances)
                # plt.title('Distance estimation (relative)')
                # plt.pause(0.01)

                if cv2.waitKey(1) == ord('w'):
                    # visualize landmark's scatter plot, W
                    plotting(landmark, eyes_arr, lips_arr, eyes_mean, lips_mean)
            
            # source.show(frame)
            cv2.imshow('head posture', frame)
        # plt.show()


if __name__ == "__main__":
    main()
