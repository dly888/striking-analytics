from .config import DefenseConfig
from .defense_detections import GuardDetections
from .defense_detectors import detect_guard_drop
from .tracking import PersonState


class DefenseAnalyser:
    def __init__(
        self,
        person_state: PersonState,
        defense_config: DefenseConfig | None = None,
    ):
        self.person_state = person_state
        self.defense_config = defense_config or DefenseConfig()

    def get_guard_dropped_detections(self):
        mask = detect_guard_drop(self.person_state, self.defense_config)
        return GuardDetections(mask=mask[2])  # Use the both_guard_dropped mask for now

    def get_defense_detections(self):
        """
        To be added
        """
        pass
