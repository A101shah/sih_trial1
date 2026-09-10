"""
Agent state and execution trace management for SatQuery AI.
"""

from typing import List, Dict, Any, Optional
import time


class AgentState:
    """
    Maintains the execution state, trace logs, and intermediate evidence during an agentic run.
    """

    def __init__(self, task_type: str = "auto"):
        self.task_type = task_type
        self.start_time = time.time()
        self.trace: List[str] = []
        self.models_used: List[str] = []
        self.evidence: Dict[str, Any] = {}
        self.warnings: List[str] = []

    def log_step(self, step_name: str, details: Optional[str] = None):
        """Appends a timestamped step to the agent execution trace."""
        entry = step_name if details is None else f"{step_name}: {details}"
        self.trace.append(entry)

    def register_model(self, model_name: str):
        """Records a specialist model used in the pipeline."""
        if model_name not in self.models_used:
            self.models_used.append(model_name)

    def add_warning(self, warning: str):
        """Adds non-fatal warning to state."""
        self.warnings.append(warning)

    def elapsed_time(self) -> float:
        """Returns elapsed time in seconds."""
        return round(time.time() - self.start_time, 3)
