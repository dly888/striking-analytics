from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tracking import Person, PersonTrack


class TrackCache:
    def __init__(self):
        self.person_tracks = []

    def add_person_track(self, track: PersonTrack):
        """
        Add a PersonTrack object to be serialised.

        Args:
            track: PersonTrack object to be added to be serialised.
        """
        self.person_tracks.append(track)

    def save_track(self, track: PersonTrack, new_path: Path) -> None:
        """
        Serialise a single PersonTrack object.

        Args:
            track: PersonTrack object to be serialised.
            new_path: Path which the track object will be serialised to.
        """
        meta = {
            "track_id": track.track_id,
            "fps": track.fps,
            "person": asdict(track.person),
        }
        new_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            new_path,
            keypoints=track.keypoints,
            boxes=track.boxes,
            box_conf=track.box_conf,
            _meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        )

    def load_track(self, path: Path) -> PersonTrack:
        """
        Deserialize a file into PersonTrack object.

        Args:
            path: Path to file which will be deserialized.

        Return:
            Deserialized PersonTrack object.
        """
        with np.load(path) as data:
            meta = json.loads(bytes(data["_meta"]))
            return PersonTrack(
                track_id=meta["track_id"],
                keypoints=data["keypoints"],
                boxes=data["boxes"],
                box_conf=data["box_conf"],
                fps=meta["fps"],
                person=Person(**meta["person"]),
            )