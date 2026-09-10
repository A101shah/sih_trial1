"""
Report Generation & Export for SatQuery AI.
Generates comprehensive analysis reports in Markdown, HTML, and JSON formats.
Clearly distinguishes real model outputs from unconfigured specialists.
"""

import json
from typing import Dict, Any
import numpy as np


class ReportExporter:
    """
    Exports SatQuery AI execution results into professional remote-sensing intelligence reports.
    """

    @classmethod
    def to_markdown(cls, result: Dict[str, Any], title: str = "SatQuery AI Remote Sensing Report") -> str:
        """
        Generates a clean GitHub-Flavored Markdown report.
        """
        conf_val = result.get("confidence")
        conf_str = f"{conf_val * 100:.2f}% ({result.get('confidence_source', 'Model Logits')})" if conf_val is not None else "N/A (Specialist Not Configured)"

        lines = [
            f"# {title}",
            "",
            f"**Task Classification:** `{result.get('task', 'N/A')}`  ",
            f"**Model Confidence:** `{conf_str}`  ",
            f"**Execution Latency:** `{result.get('execution_time_sec', 'N/A')}s`  ",
            "",
            "## 1. Query & AI Result",
            f"**Query / Question:** {result.get('question', 'N/A')}  ",
            f"**Result / Response:**  \n> {result.get('answer', 'N/A')}",
            "",
            "## 2. Specialist Statuses & Models Executed",
        ]

        if result.get("specialist_statuses"):
            for spec, status in result["specialist_statuses"].items():
                badge = "[EXECUTED]" if status == "EXECUTED" else "[NOT CONFIGURED]"
                lines.append(f"- **{spec}:** `{status}` {badge}")
        else:
            models = result.get("models_used", [])
            lines.append(f"- Models: {', '.join(models) if models else 'None executed'}")

        if "change_percentage" in result:
            lines.extend([
                "",
                "## 3. Spatial Change Evidence (ChangeFormerV6)",
                f"- **Detected Changed Area:** `{result['change_percentage']:.2f}%`",
                f"- **Connected Change Regions:** `{len(result.get('bboxes', []))}`",
                f"- **Confidence Source:** `{result.get('confidence_source', 'ChangeFormerV6 Softmax')}`"
            ])

        if "optical_evidence" in result:
            lines.extend([
                "",
                "## 3. Optical Preprocessing Metadata",
                f"- Metadata: `{result['optical_evidence'].get('metadata', {})}`"
            ])

        if "sar_evidence" in result:
            lines.extend([
                "",
                "## 4. SAR Preprocessing Metadata",
                f"- Backscatter dB Range: `{result['sar_evidence'].get('db_range', 'N/A')}`",
                f"- Metadata: `{result['sar_evidence'].get('metadata', {})}`"
            ])

        lines.extend([
            "",
            "## Execution Trace",
        ])

        for idx, step in enumerate(result.get("trace", []), 1):
            lines.append(f"{idx}. `{step}`")

        if result.get("geospatial_metadata"):
            lines.extend([
                "",
                "## Geospatial Metadata",
                "```json",
                json.dumps(result["geospatial_metadata"], indent=2, default=str),
                "```"
            ])

        return "\n".join(lines)

    @classmethod
    def to_json(cls, result: Dict[str, Any]) -> str:
        """
        Serializes result object cleanly to JSON, stripping large numpy arrays.
        """
        clean_dict = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                clean_dict[k + "_shape"] = list(v.shape)
            else:
                clean_dict[k] = v
        return json.dumps(clean_dict, indent=2, default=str)
