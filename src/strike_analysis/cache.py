from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import Config
from .tracking import PoseTracker, Person, PersonState


class TrackCache:
    def __init__(self):
        self.person_tracks = []
        self.pose_tracker = None

    def add_person_track(self, track: PersonState) -> None:
        """
        Add a PersonTrack object to be serialised.

        Args:
            track: PersonTrack object to be added to be serialised.
        """
        self.person_tracks.append(track)

    def add_pose_tracker(self, pose_tracker: PoseTracker) -> None:
        """
        Add pose tracker to be serialised.

        Args:
            pose_tracker: PoseTracker to be serialised
        """
        self.pose_tracker = pose_tracker

    @staticmethod
    def save_pose_tracker(pose_tracker: PoseTracker, new_path: Path) -> None:
        """
        Serialises pose tracker to new file path if it does not exist already.

        Args:
            pose_tracker: Pose tracker to be serialised
            new_path: Path of the file which pose tracker will be serialised into
        """
        tracking_data = {}
        state_meta = {}

        for track_id, state in pose_tracker.person_states.items():
            key = str(track_id)
            tracking_data[f"boxes_{key}"] = state.boxes
            tracking_data[f"keypoints_{key}"] = state.keypoints
            tracking_data[f"conf_{key}"] = state.box_conf

            state_meta[key] = {
                "track_id": state.track_id,
                "fps": state.fps,
                "person": asdict(state.person)
            }

        meta = {
            "model_name": pose_tracker.model_name,
            "config": asdict(pose_tracker.config),
            "person": asdict(pose_tracker.person),
            "states": state_meta,
        }

        new_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            new_path,
            _meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
            **tracking_data,
        )

    @staticmethod
    def load_pose_tracker(path: Path) -> PoseTracker:
        """
        Deserialises pose tracker from file path if it exists.

        Args:
            path: Path of the file which pose tracker will be deserialised from

        Returns:
            Posetracker object which has been deserialised.
        """
        with np.load(path) as data:
            meta = json.loads(bytes(data["_meta"]))

            tracker = PoseTracker(
                person=Person(**meta["person"]),
                model_name=meta["model_name"],
                config=Config(**meta["config"]),
            )

            states = {}
            for key, state_meta in meta["states"].items():
                states[int(key)] = PersonState(
                    track_id=state_meta["track_id"],
                    keypoints=data[f"keypoints_{key}"],
                    boxes=data[f"boxes_{key}"],
                    box_conf=data[f"conf_{key}"],
                    fps=state_meta["fps"],
                    person=Person(**state_meta["person"]),
                )
            tracker.person_states = states

        return tracker

    @staticmethod
    def save_state(person_state: PersonState, new_path: Path) -> None:
        """
        Serialise a single PersonTrack object.

        Args:
            person_state: PersonTrack object to be serialised.
            new_path: Path which the track object will be serialised to.
        """
        meta = {
            "track_id": person_state.track_id,
            "fps": person_state.fps,
            "person": asdict(person_state.person),
        }
        new_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            new_path,
            keypoints=person_state.keypoints,
            boxes=person_state.boxes,
            box_conf=person_state.box_conf,
            _meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        )

    @staticmethod
    def load_state(path: Path) -> PersonState:
        """
        Deserialize a file into PersonTrack object.

        Args:
            path: Path to file which will be deserialized.

        Return:
            Deserialized PersonTrack object.
        """
        with np.load(path) as data:
            meta = json.loads(bytes(data["_meta"]))
            return PersonState(
                track_id=meta["track_id"],
                keypoints=data["keypoints"],
                boxes=data["boxes"],
                box_conf=data["box_conf"],
                fps=meta["fps"],
                person=Person(**meta["person"]),
            )