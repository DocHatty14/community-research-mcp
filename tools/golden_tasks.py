"""
Lightweight golden tasks for manual/regression evaluation.

These are NOT full integration tests—they provide a repeatable set of prompts
and expected signals to check whether community_search outputs stay actionable.
"""

from typing import Dict, List

GoldenTask = Dict[str, object]

GOLDEN_TASKS: List[GoldenTask] = [
    {
        "name": "wgpu PipelineCompilationOptions removal",
        "language": "Rust",
        "topic": "wgpu PipelineCompilationOptions removed in latest version",
        "goal": "Fix compile errors after upgrading to wgpu 0.19",
        "must_include": [
            "PipelineCompilationOptions",
            "ShaderModuleDescriptor",
            "ShaderSource::Wgsl",
        ],
        "min_sources": 2,
    },
    {
        "name": "FastAPI background tasks with Celery",
        "language": "Python",
        "topic": "FastAPI background tasks queue with Celery and Redis",
        "goal": "Run slow email sending without blocking API responses",
        "must_include": ["Celery", "Redis", "FastAPI", "background tasks"],
        "min_sources": 2,
    },
    {
        "name": "React form validation hooks",
        "language": "JavaScript",
        "topic": "React custom hooks for form validation with Yup",
        "goal": "Client-side validation with minimal boilerplate",
        "must_include": ["Yup", "React hooks"],
        "min_sources": 2,
    },
]
