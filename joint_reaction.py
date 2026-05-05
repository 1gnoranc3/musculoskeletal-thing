import numpy as np
from dataclasses import dataclass


@dataclass
class JointReactionResult:
    joint_name: str
    contact_force_bw: np.ndarray
    contact_force_n: np.ndarray


class JointReactionAnalysis:
    """F_contact_knee ≈ GRF_vertical + Σ F_muscle (правая нога, пересекающие колено)"""

    def __init__(self, body_mass_kg: float, g: float = 9.81):
        self.body_weight = body_mass_kg * g

    def compute_knee_contact_force(
        self,
        activations: np.ndarray,
        max_forces: np.ndarray,
        knee_muscle_mask: np.ndarray,
        grf_vertical: np.ndarray,
        knee_moment_arm_approx: float = 0.04,
    ) -> JointReactionResult:
        muscle_forces           = activations * max_forces[np.newaxis, :]
        total_knee_muscle_force = np.sum(muscle_forces[:, knee_muscle_mask], axis=1)
        contact_force_n         = grf_vertical + total_knee_muscle_force
        return JointReactionResult(
            joint_name="knee",
            contact_force_n=contact_force_n,
            contact_force_bw=contact_force_n / self.body_weight,
        )

    @staticmethod
    def compute_rmse_bw(predicted_bw: np.ndarray, reference_bw: np.ndarray) -> float:
        return float(np.sqrt(np.mean((predicted_bw - reference_bw) ** 2)))

    @staticmethod
    def compute_peak_rmse_bw(predicted_bw: np.ndarray, reference_bw: np.ndarray) -> float:
        return abs(np.max(predicted_bw) - np.max(reference_bw))
