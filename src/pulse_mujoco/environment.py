from __future__ import annotations

import copy

import numpy as np

from .actions import PulseAction
from .backend import MujocoPulseDiagnosisBackend
from .observation import PulseObservation
from .safety import SafetySupervisor


class PulseDiagnosisEnv:
    """Gymnasium-style, privilege-separated policy boundary."""

    def __init__(
        self,
        backend: MujocoPulseDiagnosisBackend,
        safety: SafetySupervisor | None = None,
        perturbation_scheduler=None,
    ):
        self.backend = backend
        self.safety = safety or SafetySupervisor()
        self.perturbations = perturbation_scheduler
        self.observation: PulseObservation | None = None
        self.backend_info: dict = {}
        self.environment_step = 0

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self.observation, self.backend_info = self.backend.reset(seed)
        self.environment_step = 0
        if self.perturbations is not None:
            self.observation = self.perturbations.apply_observation(
                self.observation, self.environment_step
            )
            self.backend_info = dict(self.backend_info)
            self.backend_info["perturbation_state"] = (
                self.perturbations.state(self.environment_step)
            )
        return self.observation.policy_dict(), copy.deepcopy(
            self.backend_info
        )

    def step(
        self, action: PulseAction | np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        if self.observation is None:
            raise RuntimeError("reset() must be called before step()")
        proposed = (
            action
            if isinstance(action, PulseAction)
            else PulseAction.from_array(action)
        )
        safe, safety_info = self.safety.filter_action(
            self.observation, proposed, self.backend_info
        )
        execution = safe
        execution_safety = None
        perturbation_state = None
        if self.perturbations is not None:
            perturbation_state = self.perturbations.state(
                self.environment_step
            )
            perturbed = self.perturbations.apply_action(
                safe, self.environment_step
            )
            execution, execution_safety = self.safety.filter_action(
                self.observation, perturbed, self.backend_info
            )
        observation, reward, terminated, truncated, info = (
            self.backend.apply_action(execution)
        )
        self.environment_step += 1
        if self.perturbations is not None:
            observation = self.perturbations.apply_observation(
                observation, self.environment_step
            )
        self.observation = observation
        self.backend_info = dict(info)
        result = dict(info)
        if perturbation_state is not None:
            result["perturbation_state"] = perturbation_state
        if perturbation_state and perturbation_state.get("recovery_active"):
            result["expert_recovery"] = {
                "active": True,
                "mode": "+".join(perturbation_state["recovery_modes"]),
                "reason": "execution_disturbance",
            }
        result.update(
            {
                "safety": safety_info,
                "proposed_action": proposed.as_array(),
                "safe_action": safe.as_array(),
                "execution_action": execution.as_array(),
                "executed_action": np.asarray(
                    info["executed_action"], dtype=np.float32
                ),
                "proposed_action_structured": proposed.as_dict(),
                "safe_action_structured": safe.as_dict(),
                "execution_action_structured": execution.as_dict(),
                "execution_safety": execution_safety,
                "perturbation_state": perturbation_state or {
                    "step": self.environment_step - 1,
                    "active": False,
                    "events": [],
                    "recoverable": True,
                },
            }
        )
        return (
            observation.policy_dict(),
            float(reward),
            bool(terminated),
            bool(truncated),
            result,
        )

    def save_state(self) -> dict:
        if self.observation is None:
            raise RuntimeError("reset() must be called before save_state()")
        return {
            "schema_version": 1,
            "backend": self.backend.save_state(),
            "observation": copy.deepcopy(self.observation),
            "backend_info": copy.deepcopy(self.backend_info),
            "environment_step": int(self.environment_step),
        }

    def load_state(self, snapshot: dict) -> dict:
        if snapshot.get("schema_version") != 1:
            raise ValueError("unsupported snapshot schema")
        self.backend.load_state(copy.deepcopy(snapshot["backend"]))
        self.observation = copy.deepcopy(snapshot["observation"])
        self.backend_info = copy.deepcopy(snapshot["backend_info"])
        self.environment_step = int(snapshot.get("environment_step", 0))
        return self.observation.policy_dict()

    def get_privileged_state(self) -> dict:
        return self.backend.get_privileged_state()

    def emergency_retreat(self) -> dict:
        self.observation = self.backend.emergency_retreat()
        self.backend_info = self.backend._info(ik_converged=True)
        return self.observation.policy_dict()

    def close(self) -> None:
        self.backend.close()
