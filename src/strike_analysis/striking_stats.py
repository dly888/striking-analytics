import numpy as np

from strike_analysis import Detections, PersonState, get_joint_speed, StrikeConfig

strike_config = StrikeConfig()


class StrikingStats:
    def __init__(self, person_state: PersonState, detections: Detections):
        self.person_state = person_state
        self.detections = detections
        self.fps = person_state.fps
        self.strike_records = detections.to_records(fps=self.fps)
        self.joint_speeds = self.get_joint_speeds()

    def get_joint_speeds(self):
        speeds = {
            "left_straight": get_joint_speed(self.person_state, "left_wrist", strike_config),
            "right_straight": get_joint_speed(self.person_state, "right_wrist", strike_config),
            "left_hook": get_joint_speed(self.person_state, "left_wrist", strike_config),
            "right_hook": get_joint_speed(self.person_state, "right_wrist", strike_config),
            "left_uppercut": get_joint_speed(self.person_state, "left_wrist", strike_config),
            "right_uppercut": get_joint_speed(self.person_state, "right_wrist", strike_config),
            "left_kick": get_joint_speed(self.person_state, "left_ankle", strike_config),
            "right_kick": get_joint_speed(self.person_state, "right_ankle", strike_config),
        }

        return speeds

    def get_max_speeds(self):
        max_speeds = {
            "left_straight": 0,
            "right_straight": 0,
            "left_hook": 0,
            "right_hook": 0,
            "left_uppercut": 0,
            "right_uppercut": 0,
            "left_kick": 0,
            "right_kick": 0,
        }


        for record in self.strike_records:
            start_frame = record["frame"]
            strike_type = record["strike_type"]

            current_max_speed = np.max(self.joint_speeds[strike_type][start_frame: start_frame + self.fps // 3])
            max_speeds[strike_type] = max(current_max_speed, max_speeds[strike_type])


        return max_speeds

    def get_strike_count(self):
        strike_counts = {
            "left_straight": 0,
            "right_straight": 0,
            "left_hook": 0,
            "right_hook": 0,
            "left_uppercut": 0,
            "right_uppercut": 0,
            "left_kick": 0,
            "right_kick": 0,
        }

        for record in self.strike_records:
            strike_type = record["strike_type"]
            strike_counts[strike_type] += 1

        return strike_counts


    def get_combo_count(self):
        strike_frame_idx = np.ndarray(shape=len(self.strike_records))  # Index of when a strike occurs

        for i, record in enumerate(self.strike_records):
            strike_frame_idx[i] = record["frame"]


        # Convert strike_frame_idx to mask, where if the frame_idx are close enough they are grouped as 1



