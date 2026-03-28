import cv2
import mediapipe as mp
import time
import numpy as np

class VisionModule:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for Left and Right Eyes
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def calculate_ear(self, landmarks, eye_indices):
        """Calculates the Eye Aspect Ratio (EAR)"""
        try:
            coords = []
            for idx in eye_indices:
                lm = landmarks[idx]
                coords.append(np.array([lm.x, lm.y]))

            # Horizontal distance
            dist_h = np.linalg.norm(coords[0] - coords[3])
            # Vertical distances
            dist_v1 = np.linalg.norm(coords[1] - coords[5])
            dist_v2 = np.linalg.norm(coords[2] - coords[4])

            ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
            return ear
        except Exception:
            return 0.0

    def run(self, shared_data=None):
        """Main Vision Loop"""
        cap = cv2.VideoCapture(0)
        # PC Capping: Lower resolution to save CPU on Pi 5
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Vision Module Started...")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # Performance: Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            results = self.face_mesh.process(rgb_frame)
            t1 = time.perf_counter()

            if results.multi_face_landmarks:
                mesh_coords = results.multi_face_landmarks[0].landmark
                
                left_ear = self.calculate_ear(mesh_coords, self.LEFT_EYE)
                right_ear = self.calculate_ear(mesh_coords, self.RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                # Update the Shared Blackboard for the NN
                if shared_data is not None:
                    shared_data['ear'] = float(avg_ear)
                    shared_data['vision_latency'] = (t1 - t0) * 1000

            # Optional: Display for troubleshooting (Disable during final headless run)
            # cv2.imshow('Triage Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

# Boilerplate to allow independent testing
if __name__ == "__main__":
    vision = VisionModule()
    vision.run()