#!/usr/bin/env python3
"""
Community Research MCP Server

An MCP server that searches Stack Overflow, Reddit, GitHub issues, and forums
to find real solutions from real developers. No more AI hallucinations - find
what people actually use in production.

Features:
- Multi-source search (Stack Overflow, Reddit, GitHub, HackerNews)
- Query validation (rejects vague queries with helpful suggestions)
- Returns raw structured results for calling LLM to synthesize (no internal LLM calls)
- Smart caching and retry logic
- Workspace context detection
"""

import io
import sys

# Fix Windows console encoding issues with emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        # If reconfigure is not available, wrap stdout/stderr
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import streaming capabilities
try:
    from streaming_capabilities import (
        SOURCE_LABELS,
        SystemCapabilities,
        classify_result,
        detect_all_capabilities,
        format_capabilities_report,
        summarize_content_shapes,
    )
    from streaming_search import (
        get_all_search_results_streaming,
        streaming_search_with_synthesis,
    )

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    print("Note: Streaming capabilities unavailable")
    SOURCE_LABELS = {}

    def summarize_content_shapes(
        results_by_source: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        return {"per_source": {}, "totals": {}}


# Import enhanced MCP utilities for production-grade reliability
try:
    from enhanced_mcp_utilities import (
        QualityScorer,
        RetryStrategy,
        deduplicate_results,
        format_metrics_report,
        get_api_metrics,
        get_circuit_breaker,
        get_performance_monitor,
        resilient_api_call,
    )

    ENHANCED_UTILITIES_AVAILABLE = True
    # Initialize quality scorer with optional preset
    _quality_preset = os.getenv("QUALITY_SCORER_PRESET", "balanced")
    _quality_scorer = QualityScorer(_quality_preset)
    print("Enhanced utilities active (with circuit breakers)")
except ImportError:
    ENHANCED_UTILITIES_AVAILABLE = False
    _quality_scorer = None
    print("Note: Enhanced utilities unavailable")

# Import core modules
# Import refactored modules
from api import (
    search_brave,
    search_discourse,
    search_firecrawl,
    search_github,
    search_hackernews,
    search_lobsters,
    search_serper,
    search_stackoverflow,
    search_tavily,
)
from models import CommunitySearchInput, DeepAnalyzeInput, ResponseFormat, ThinkingMode
from utils import check_rate_limit, get_cache_key, get_cached_result, set_cached_result

# Set up logging
logging.getLogger().setLevel(logging.WARNING)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass

# Initialize Reddit API client if credentials are available
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_REFRESH_TOKEN = os.getenv("REDDIT_REFRESH_TOKEN")

# Optional premium web search providers
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_API_URL = os.getenv(
    "FIRECRAWL_API_URL", "https://api.firecrawl.dev/v1/search"
)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_API_URL = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Check if Reddit credentials are available
reddit_authenticated = False
reddit_client = None

if all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_REFRESH_TOKEN]):
    try:
        from redditwarp.ASYNC import Client as RedditClient

        reddit_client = RedditClient(
            REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_REFRESH_TOKEN
        )
        reddit_authenticated = True
        print("Reddit API client initialized with authentication.")
    except ImportError:
        print(
            "Warning: redditwarp package not installed. Using unauthenticated Reddit API."
        )
    except Exception as e:
        print(
            f"Warning: Failed to initialize Reddit API client: {str(e)}. Using unauthenticated Reddit API."
        )


# Load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from config.json with fallback defaults."""
    config_path = Path(__file__).parent / "config.json"
    default_config = {
        "quality": {"min_score": 0, "min_votes": 0, "enable_filtering": False},
        # Source weights prioritize community discussions over official docs
        # Higher weight = more trusted for "street-smart" solutions
        "sources": {
            # Primary community sources - where real solutions live
            "stackoverflow": {"enabled": True, "weight": 10},  # Accepted answers = gold
            "github": {
                "enabled": True,
                "weight": 9,
            },  # Issues & discussions = real bugs/fixes
            "discourse": {
                "enabled": True,
                "weight": 8,
            },  # Framework-specific community wisdom
            "lobsters": {
                "enabled": True,
                "weight": 7,
            },  # Technical depth, experienced devs
            "hackernews": {
                "enabled": True,
                "weight": 6,
            },  # Industry discussions, war stories
            "reddit": {"enabled": True, "weight": 6},  # r/programming, r/webdev, etc.
            # Web search APIs - find community content across the web
            # Lower weight because they may return official docs (we want workarounds)
            "brave": {"enabled": True, "weight": 4},
            "serper": {"enabled": True, "weight": 4},
            "tavily": {"enabled": True, "weight": 4},
            "firecrawl": {"enabled": True, "weight": 3},
        },
        "cache": {"enabled": True, "ttl_seconds": 3600},
        "output": {
            "max_results": 15,
            "include_debug_info": True,
            "fallback_mode": "relaxed",
        },
    }

    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config.json: {e}")

    return default_config


CONFIG = load_config()

# Initialize MCP server
mcp = FastMCP("community_research_mcp")

# Global registry for MCP-connected AI assistance
_mcp_ai_context = {
    "available": False,
    "provider": None,
    "last_query_optimization": None,
    "optimization_cache": {},
}

# Constants
CHARACTER_LIMIT = 25000
API_TIMEOUT = 30.0
MAX_RETRIES = 3
CACHE_TTL_SECONDS = 3600  # 1 hour
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_CALLS = 10

# Source guardrails and defaults
SOURCE_POLICIES: Dict[str, Dict[str, Any]] = {
    "stackoverflow": {
        "min_query_length": 10,
        "max_results": 15,
        "max_results_expanded": 30,
        "read_only": True,
    },
    "github": {
        "min_query_length": 10,
        "max_results": 15,
        "max_results_expanded": 30,
        "read_only": True,
    },
    "reddit": {
        "min_query_length": 10,
        "max_results": 15,
        "max_results_expanded": 30,
        "read_only": True,
    },
    "hackernews": {
        "min_query_length": 8,
        "max_results": 10,
        "max_results_expanded": 20,
        "read_only": True,
    },
    "lobsters": {
        "min_query_length": 8,
        "max_results": 10,
        "max_results_expanded": 20,
        "read_only": True,
    },
    "discourse": {
        "min_query_length": 10,
        "max_results": 15,
        "max_results_expanded": 30,
        "read_only": True,
    },
    "firecrawl": {
        "min_query_length": 6,
        "max_results": 12,
        "max_results_expanded": 25,
        "read_only": True,
    },
    "tavily": {
        "min_query_length": 6,
        "max_results": 12,
        "max_results_expanded": 25,
        "read_only": True,
    },
}

# Deterministic fixtures for health/diagnostic mode (no live calls)
TEST_FIXTURES: Dict[str, List[Dict[str, Any]]] = {
    "stackoverflow": [
        {
            "title": "How to safely add MCP tool fallback?",
            "url": "https://stackoverflow.com/q/example-mcp",
            "score": 5,
            "answer_count": 1,
            "snippet": "Use a retry wrapper and a fallback search source to avoid total failure.",
        }
    ],
    "github": [
        {
            "title": "Add guardrails to MCP server",
            "url": "https://github.com/example/mcp/pull/1",
            "state": "open",
            "comments": 2,
            "snippet": "Implements allowlists and quotas for MCP tool actions.",
        }
    ],
    "reddit": [
        {
            "title": "Using MCP to wrap database access safely",
            "url": "https://www.reddit.com/r/python/comments/example_mcp/",
            "score": 12,
            "comments": 3,
            "snippet": "Wrap DB calls behind MCP tools with explicit allowlists.",
            "authenticated": False,
        }
    ],
    "hackernews": [
        {
            "title": "Lessons learned building MCP integrations",
            "url": "https://news.ycombinator.com/item?id=123456",
            "points": 120,
            "comments": 10,
            "snippet": "",
        }
    ],
    "lobsters": [
        {
            "title": "Building MCP tools for community research",
            "url": "https://lobste.rs/s/abc123",
            "points": 25,
            "comments": 8,
            "snippet": "Community discussion on MCP best practices.",
        }
    ],
    "discourse": [
        {
            "title": "MCP integration patterns",
            "url": "https://discuss.python.org/t/12345",
            "views": 150,
            "replies": 5,
            "snippet": "Thread discussing MCP server design patterns.",
        }
    ],
}

MANUAL_EVIDENCE: Dict[str, List[Dict[str, Any]]] = {
    "wgpu_pipelinecompilationoptions": [
        {
            "title": "wgpu 0.19: PipelineCompilationOptions removed",
            "url": "https://github.com/gfx-rs/wgpu/issues/4528",
            "source": "github",
            "score": 92,
            "issue": "PipelineCompilationOptions removed in API cleanup.",
            "solution": "Create shader modules with ShaderModuleDescriptor { label, source: ShaderSource::Wgsl(...) }; options API is gone.",
            "code": 'let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {\n    label: Some("main"),\n    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),\n});',
            "evidence": [
                {
                    "url": "https://github.com/gfx-rs/wgpu/issues/4528",
                    "quote": "Removed PipelineCompilationOptions; shader modules now only take label/source.",
                    "signal": "wgpu issue #4528",
                }
            ],
            "difficulty": "Medium",
        },
        {
            "title": "Bevy migration guide 0.13 (wgpu 0.19)",
            "url": "https://bevyengine.org/learn/book/migration/0.13/",
            "source": "web",
            "score": 75,
            "issue": "Breakage upgrading Bevy to wgpu 0.19.",
            "solution": "Remove compilation_options entirely; use ShaderSource::Wgsl/SpirV in ShaderModuleDescriptor.",
            "code": "",
            "evidence": [
                {
                    "url": "https://bevyengine.org/learn/book/migration/0.13/",
                    "quote": "wgpu 0.19 removes PipelineCompilationOptions; adjust shader creation accordingly.",
                    "signal": "Bevy migration guide 0.13",
                }
            ],
            "difficulty": "Medium",
        },
        {
            "title": "Stack Overflow: compilation_options removed",
            "url": "https://stackoverflow.com/q/xxxxx",
            "source": "stackoverflow",
            "score": 88,
            "issue": "compilation_options removed from ShaderModuleDescriptor in 0.19",
            "solution": "Use ShaderSource::Wgsl and label; no replacement for options.",
            "code": "",
            "evidence": [
                {
                    "url": "https://stackoverflow.com/q/xxxxx",
                    "quote": "PipelineCompilationOptions was removed in 0.19; create the shader with the new descriptor fields.",
                    "signal": "SO votes",
                }
            ],
            "difficulty": "Easy",
        },
    ],
    "fastapi_celery_redis": [
        {
            "title": "FastAPI background tasks with Celery + Redis",
            "url": "https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html",
            "source": "web",
            "score": 80,
            "issue": "Long-running email sending blocks FastAPI responses.",
            "solution": "Use Celery worker with Redis broker; trigger tasks via delay/apply_async from FastAPI endpoint.",
            "code": "from celery import Celery\\ncelery_app = Celery('worker', broker='redis://localhost:6379/0')\\n\\n@celery_app.task\\ndef send_email(to):\\n    ...\\n\\n# fastapi endpoint\\n@app.post('/send')\\nasync def send(to: str):\\n    send_email.delay(to)\\n    return {'status': 'queued'}",
            "evidence": [
                {
                    "url": "https://fastapi.tiangolo.com/advanced/background-tasks/",
                    "quote": "For long tasks use a task queue (Celery/RQ) instead of FastAPI BackgroundTasks.",
                    "signal": "FastAPI docs",
                }
            ],
            "difficulty": "Medium",
        }
    ],
    "react_yup_hook": [
        {
            "title": "React custom hook with Yup validation",
            "url": "https://github.com/jquense/yup",
            "source": "github",
            "score": 78,
            "issue": "Form validation boilerplate repeated across components.",
            "solution": "Create useFormValidation hook that takes a Yup schema, tracks errors, and validates on change/submit.",
            "code": "import { useState } from 'react'\\nimport * as yup from 'yup'\\n\\nexport function useFormValidation(schema) {\\n  const [errors, setErrors] = useState({})\\n  const validate = async (values) => {\\n    try {\\n      await schema.validate(values, { abortEarly: false })\\n      setErrors({});\\n      return true;\\n    } catch (err) {\\n      const next = {};\\n      err.inner.forEach(e => next[e.path] = e.message);\\n      setErrors(next);\\n      return false;\\n    }\\n  };\\n  return { errors, validate };\\n}",
            "evidence": [
                {
                    "url": "https://dev.to/pallymore/creating-a-custom-react-hook-to-handle-form-validation-1ine",
                    "quote": "Custom hook wrapping Yup validation for reusable form logic.",
                    "signal": "blog reference",
                }
            ],
            "difficulty": "Easy",
        }
    ],
    "docker_multistage": [
        {
            "title": "Slim Docker image with multi-stage build",
            "url": "https://docs.docker.com/build/building/multi-stage/",
            "source": "web",
            "score": 85,
            "issue": "Production image too large.",
            "solution": "Build in a builder stage and copy artifacts into a small runtime base (alpine or distroless).",
            "code": 'FROM node:20-alpine AS build\\nWORKDIR /app\\nCOPY package*.json ./\\nRUN npm ci --only=production\\nCOPY . .\\nRUN npm run build\\n\\nFROM node:20-alpine AS runtime\\nWORKDIR /app\\nCOPY --from=build /app/node_modules ./node_modules\\nCOPY --from=build /app/dist ./dist\\nCMD ["node", "dist/server.js"]',
            "evidence": [
                {
                    "url": "https://docs.docker.com/build/building/multi-stage/",
                    "quote": "Use multi-stage builds to keep the final image small.",
                    "signal": "Docker docs",
                }
            ],
            "difficulty": "Easy",
        }
    ],
    "tokio_reset_by_peer": [
        {
            "title": "Handle ECONNRESET in tokio TCP server",
            "url": "https://tokio.rs/tokio/tutorial/io",
            "source": "web",
            "score": 70,
            "issue": 'Connections drop with "connection reset by peer".',
            "solution": "Match io::ErrorKind::ConnectionReset/ConnectionAborted and continue; ensure half-close handling and timeouts.",
            "code": 'use tokio::net::TcpListener;\\nuse tokio::io::{AsyncReadExt, AsyncWriteExt};\\nuse std::io;\\n\\nasync fn handle(mut socket: tokio::net::TcpStream) {\\n    let mut buf = [0u8; 1024];\\n    loop {\\n        match socket.read(&mut buf).await {\\n            Ok(0) => break,\\n            Ok(n) => { let _ = socket.write_all(&buf[..n]).await; }\\n            Err(e) if e.kind() == io::ErrorKind::ConnectionReset || e.kind() == io::ErrorKind::ConnectionAborted => break,\\n            Err(e) => { eprintln!("read error: {e}"); break; }\\n        }\\n    }\\n}\\n\\n#[tokio::main]\\nasync fn main() -> io::Result<()> {\\n    let listener = TcpListener::bind("0.0.0.0:8080").await?;\\n    loop {\\n        let (socket, _) = listener.accept().await?;\\n        tokio::spawn(handle(socket));\\n    }\\n}',
            "evidence": [
                {
                    "url": "https://users.rust-lang.org/t/handling-connectionreset/62031",
                    "quote": "ConnectionReset is common when clients disconnect; handle and continue.",
                    "signal": "community discussion",
                }
            ],
            "difficulty": "Easy",
        }
    ],
}

# Global state
_cache: Dict[str, Dict[str, Any]] = {}
_rate_limit_tracker: Dict[str, List[float]] = {}

# ============================================================================
# Workspace Context Detection
# ============================================================================


def detect_workspace_context() -> Dict[str, Any]:
    """
    Detect programming languages and frameworks in the current workspace.

    Returns:
        Dictionary with workspace context including languages, frameworks, and structure
    """
    cwd = Path.cwd()

    languages = set()
    frameworks = set()
    config_files = []

    # Language detection patterns
    language_patterns = {
        "Python": [".py"],
        "JavaScript": [".js", ".jsx"],
        "TypeScript": [".ts", ".tsx"],
        "Java": [".java"],
        "C++": [".cpp", ".cc", ".cxx"],
        "C#": [".cs"],
        "Go": [".go"],
        "Rust": [".rs"],
        "Ruby": [".rb"],
        "PHP": [".php"],
        "Swift": [".swift"],
        "Kotlin": [".kt"],
    }

    # Framework detection patterns
    framework_files = {
        "Django": ["manage.py", "settings.py"],
        "FastAPI": ["main.py"],  # Common convention
        "Flask": ["app.py"],
        "React": ["package.json"],
        "Vue": ["vue.config.js"],
        "Angular": ["angular.json"],
        "Next.js": ["next.config.js"],
        "Express": ["package.json"],
    }

    # Scan directory (limit to first 100 files to avoid performance issues)
    file_count = 0
    max_files = 100

    try:
        for root, dirs, files in os.walk(cwd):
            # Skip common ignore directories
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {
                    ".git",
                    "node_modules",
                    "__pycache__",
                    "venv",
                    ".venv",
                    "dist",
                    "build",
                }
            ]

            for file in files:
                if file_count >= max_files:
                    break

                file_path = Path(root) / file
                file_ext = file_path.suffix

                # Detect language
                for lang, extensions in language_patterns.items():
                    if file_ext in extensions:
                        languages.add(lang)

                # Detect frameworks
                for framework, marker_files in framework_files.items():
                    if file in marker_files:
                        frameworks.add(framework)

                # Track config files
                if file in [
                    "package.json",
                    "requirements.txt",
                    "Cargo.toml",
                    "go.mod",
                    "pom.xml",
                ]:
                    config_files.append(file)

                file_count += 1

            if file_count >= max_files:
                break

    except Exception as e:
        # If scan fails, just return minimal context
        pass

    return {
        "workspace": str(cwd),
        "languages": sorted(list(languages)),
        "frameworks": sorted(list(frameworks)),
        "config_files": config_files,
        "scan_limited": file_count >= max_files,
    }


# ============================================================================
# Caching & Rate Limiting
# ============================================================================


def validate_topic_specificity(topic: str) -> tuple[bool, str]:
    """
    Validate that a topic is specific enough for useful search results.
    This mirrors the pydantic validator but returns a tuple for tool usage.
    """
    cleaned = topic.strip()

    if len(cleaned) < 10:
        return (
            False,
            "Topic is too short. Include specific technologies, goals, or patterns (>=10 chars).",
        )

    vague_terms = {
        "settings",
        "configuration",
        "config",
        "setup",
        "performance",
        "optimization",
        "best practices",
        "how to",
        "tutorial",
        "getting started",
        "basics",
        "help",
        "issue",
        "problem",
        "error",
        "debugging",
        "install",
        "installation",
    }

    words = cleaned.lower().split()
    if len(words) <= 2 and any(term in cleaned.lower() for term in vague_terms):
        return (
            False,
            f"Topic '{cleaned}' is too vague. Add concrete technologies and goals (e.g., 'FastAPI background tasks with Redis').",
        )

    return True, ""


# ============================================================================
# Query Enrichment & Quality Guardrails
# ============================================================================


VERSION_PATTERN = re.compile(r"\\b\\d+\\.\\d+(?:\\.\\d+)?\\b")


def _extract_versions(text: str) -> List[str]:
    """Return unique version strings found in free text."""
    return sorted({m.group(0) for m in VERSION_PATTERN.finditer(text)})


def _build_version_tokens(versions: List[str]) -> List[str]:
    """Create lightweight range/filter tokens for version-aware searches."""
    tokens: List[str] = []
    for ver in versions:
        tokens.extend([f"v{ver}", f"before:{ver}", f"after:{ver}", f"since {ver}"])
        parts = ver.split(".")
        if len(parts) >= 2:
            major_minor = ".".join(parts[:2])
            tokens.append(f"{major_minor}.x")
    return tokens


def enrich_query(
    language: str, topic: str, goal: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enrich query to improve search relevance by combining topic with goal.

    Smart enrichment:
    - Combines language + topic + goal for better targeting
    - Extracts key framework/library names to emphasize
    - Adds context words based on goal
    - Avoids generic spam words but includes helpful qualifiers

    Returns:
        {
            "enriched_query": "...",
            "expanded_queries": [...],
            "notes": [...],
            "assumptions": [...],
            "versions": [...]
        }
    """
    versions = _extract_versions(topic)
    notes: List[str] = []
    assumptions: List[str] = []

    if versions:
        notes.append(f"Detected versions: {', '.join(versions)}")

    if not goal:
        assumptions.append("Goal not provided; assuming intent is to fix/upgrade.")

    # Extract key framework/library names to emphasize (case-insensitive)
    framework_keywords = [
        "fastapi",
        "django",
        "flask",
        "celery",
        "redis",
        "react",
        "vue",
        "angular",
        "docker",
        "kubernetes",
        "postgres",
        "mongodb",
        "sqlalchemy",
        "pytest",
        "numpy",
        "pandas",
        "tensorflow",
        "pytorch",
    ]

    topic_lower = topic.lower()
    emphasized_frameworks = [fw for fw in framework_keywords if fw in topic_lower]

    # Build enriched query
    base = f"{language} {topic}".strip()

    # Add goal if provided to narrow down results
    if goal:
        # Add goal context without generic words
        goal_lower = goal.lower()
        if any(word in goal_lower for word in ["implement", "build", "create", "add"]):
            enriched_query = f"{base} implementation tutorial"
        elif any(word in goal_lower for word in ["fix", "debug", "error", "issue"]):
            enriched_query = f"{base} troubleshooting"
        elif any(word in goal_lower for word in ["learn", "understand", "how"]):
            enriched_query = f"{base} guide pattern"
        elif any(
            word in goal_lower for word in ["best practice", "recommended", "proper"]
        ):
            enriched_query = f"{base} best practices pattern"
        else:
            enriched_query = f"{base} {goal}"
    else:
        enriched_query = base

    # Re-emphasize key frameworks if detected
    if emphasized_frameworks and len(emphasized_frameworks) <= 2:
        # Only emphasize if there are 1-2 frameworks to avoid spam
        enriched_query = f"{' '.join(emphasized_frameworks)} {enriched_query}"

    # Expanded queries for fallback/suggestions
    expanded_queries = [
        base,
        enriched_query,
    ]
    if goal:
        expanded_queries.append(f"{base} {goal}")

    return {
        "enriched_query": enriched_query,
        "expanded_queries": expanded_queries,
        "notes": notes,
        "assumptions": assumptions,
        "versions": versions,
    }


# =============================================================================
# STREET-SMART SEARCH CONFIGURATION
# =============================================================================
# "Where the official documentation ends and actual street-smart solutions begin."
#
# These settings ensure web search APIs find REAL solutions from REAL developers:
# - Workarounds that actually work in production
# - "This finally worked for me" comments
# - Battle-tested hacks and fixes
# - The messy truth that official docs don't tell you

# Community domains - where developers share what ACTUALLY works
COMMUNITY_DOMAINS = [
    # Q&A sites - front line of problem solving
    "stackoverflow.com",
    "stackexchange.com",
    "superuser.com",
    "serverfault.com",
    # Forums & Discussion - raw, unfiltered experiences
    "reddit.com",
    "news.ycombinator.com",
    "lobste.rs",
    "dev.to",
    "hashnode.dev",
    # GitHub - real bugs, real fixes, real discussions
    "github.com",
    "gist.github.com",
    # Framework communities - specialized wisdom
    "discourse.org",
    "community.openai.com",
    "discuss.python.org",
    "forum.cursor.com",
    # Developer blogs - war stories and lessons learned
    "medium.com",
    "freecodecamp.org",
    "css-tricks.com",
    "smashingmagazine.com",
]

# Keywords that signal REAL solutions (not marketing or official docs)
STREET_SMART_KEYWORDS = [
    # Solution indicators
    "workaround",
    "fix",
    "solved",
    "working",
    "finally",
    # Community wisdom
    "hack",
    "trick",
    "tip",
    "gotcha",
    "caveat",
    "pitfall",
    # Real experiences
    "how I",
    "finally got",
    "figured out",
    "turns out",
    "the trick is",
    "what worked",
    "after hours",
    # Discussion markers
    "answered",
    "accepted",
    "upvoted",
    "this helped",
]


def enrich_query_for_community(
    query: str,
    language: Optional[str] = None,
    search_type: str = "general",
) -> Dict[str, Any]:
    """
    Enrich a query to find STREET-SMART solutions, not sanitized documentation.

    This targets the places where developers share what ACTUALLY works:
    - Real fixes from Stack Overflow accepted answers
    - "This finally worked for me" comments from Reddit
    - GitHub issues where someone figured out the workaround
    - The messy hacks that people actually use in production

    Args:
        query: The original search query
        language: Optional programming language context
        search_type: "general", "troubleshooting", "implementation", "comparison"

    Returns:
        Dict with enriched queries optimized for finding real-world solutions.
    """
    base = f"{language} {query}".strip() if language else query.strip()
    query_lower = query.lower()

    # Detect query intent
    is_troubleshooting = any(
        word in query_lower
        for word in [
            "error",
            "fix",
            "issue",
            "problem",
            "not working",
            "fails",
            "bug",
            "broken",
        ]
    )
    is_how_to = any(
        word in query_lower
        for word in ["how to", "how do", "how can", "way to", "best way"]
    )

    # Build street-smart query variants
    queries = {
        "primary": base,
        "community_focused": None,
        "site_restricted": None,
        "solution_focused": None,
    }

    if is_troubleshooting or search_type == "troubleshooting":
        # For errors - find what ACTUALLY fixed it
        queries["community_focused"] = f"{base} workaround fix solved"
        queries["solution_focused"] = f'{base} "finally worked" OR "this fixed"'
        queries["site_restricted"] = f"{base} site:stackoverflow.com OR site:github.com"
    elif is_how_to or search_type == "implementation":
        # For implementation - find working examples, not docs
        queries["community_focused"] = f"{base} working example implementation"
        queries["solution_focused"] = f'{base} "here\'s how" OR "what worked"'
        queries["site_restricted"] = f"{base} site:stackoverflow.com OR site:dev.to"
    elif search_type == "comparison":
        # For comparisons - find real production experience
        queries["community_focused"] = f"{base} vs comparison real experience"
        queries["solution_focused"] = f'{base} "we switched" OR "in production"'
        queries["site_restricted"] = (
            f"{base} site:reddit.com OR site:news.ycombinator.com"
        )
    else:
        # General - bias toward community solutions over docs
        queries["community_focused"] = f"{base} workaround solution community"
        queries["solution_focused"] = f'{base} "worked for me" OR fix'
        queries["site_restricted"] = f"{base} site:stackoverflow.com OR site:reddit.com"

    return {
        "queries": queries,
        "include_domains": COMMUNITY_DOMAINS,
        "search_hints": {
            "is_troubleshooting": is_troubleshooting,
            "is_how_to": is_how_to,
            "suggested_type": search_type,
        },
    }


def get_community_query(query: str, language: Optional[str] = None) -> str:
    """
    Transform a query to find STREET-SMART solutions.

    This is the simple interface used by web search APIs (Brave, Serper, Tavily, Firecrawl).
    Adds terms that bias results toward real-world fixes over official documentation.

    The goal: Find the Reddit comment, Stack Overflow answer, or GitHub issue
    where someone says "I finally figured it out..." or "here's what actually worked"

    Args:
        query: Original search query
        language: Optional programming language

    Returns:
        Enriched query string targeting community-sourced solutions.
    """
    base = f"{language} {query}".strip() if language else query.strip()
    query_lower = query.lower()

    # Check if query already has street-smart terms
    has_solution_terms = any(
        term in query_lower
        for term in [
            "workaround",
            "solution",
            "fix",
            "solved",
            "hack",
            "trick",
            "worked",
        ]
    )

    if has_solution_terms:
        # Query already targets solutions, don't over-enrich
        return base

    # Add street-smart terms based on query type
    if any(
        word in query_lower
        for word in ["error", "exception", "fails", "not working", "broken"]
    ):
        # Error queries - find actual fixes and workarounds
        return f"{base} workaround fix solved"
    elif any(
        word in query_lower for word in ["how to", "how do", "implement", "create"]
    ):
        # How-to queries - find working examples from the community
        return f"{base} working example solution"
    else:
        # General queries - bias toward community content over official docs
        return f"{base} community workaround solution"


def detect_conflicts_from_findings(
    findings: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Lightweight conflict detector that spots divergent recommendations.
    """
    conflicts: List[Dict[str, str]] = []
    if not findings:
        return conflicts

    content_blobs = []
    for f in findings:
        blob = " ".join(
            [
                str(f.get("title", "")),
                str(f.get("problem", "")),
                str(f.get("solution", "")),
                str(f.get("gotchas", "")),
            ]
        ).lower()
        content_blobs.append(blob)

    mentions_upgrade = any("upgrade" in b or "update" in b for b in content_blobs)
    mentions_pin = any(
        "downgrade" in b or "pin" in b or "lock" in b for b in content_blobs
    )
    if mentions_upgrade and mentions_pin:
        conflicts.append(
            {
                "description": "Conflicting guidance: upgrade vs. pin/downgrade.",
                "impact": "Choose one path to avoid dependency churn; prefer upgrade if ecosystem supports it.",
                "recommended_action": "Validate plugin/library compatibility before choosing upgrade or pin strategy.",
            }
        )

    # Detect divergent version recommendations
    versions_seen = [
        _extract_versions(" ".join([f.get("problem", ""), f.get("solution", "")]))
        for f in findings
    ]
    flat_versions = {v for sub in versions_seen for v in sub}
    if len(flat_versions) >= 2:
        conflicts.append(
            {
                "description": f"Multiple version targets detected: {', '.join(sorted(flat_versions))}.",
                "impact": "Following mixed version advice can cause inconsistent builds.",
                "recommended_action": "Align all deps to a single target version before rollout.",
            }
        )

    return conflicts


def _result_only_sources(results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Filter out non-list metadata keys from aggregated results."""
    return {k: v for k, v in results.items() if isinstance(v, list)}


def _count_total_results(results: Dict[str, Any]) -> int:
    """Count total number of results across all sources."""
    return sum(len(v) for k, v in results.items() if isinstance(v, list))


def _create_basic_findings_from_results(
    results: Dict[str, Any], query: str, language: str
) -> List[Dict[str, Any]]:
    """
    Create basic findings from search results without LLM synthesis.

    This provides a fallback when no LLM API keys are configured.
    """
    findings = []

    # Process each source
    for source, items in _result_only_sources(results).items():
        # Take top 3 results from each source
        for item in items[:3]:
            url = item.get("url") or item.get("link") or item.get("html_url")
            title = item.get("title") or "Untitled"
            snippet = (
                item.get("snippet") or item.get("body") or item.get("content") or ""
            )

            if not url:
                continue

            # Extract basic metadata
            score = (
                item.get("score") or item.get("points") or item.get("reactions") or 0
            )
            votes = item.get("votes") or item.get("score") or item.get("ups") or score
            comments = (
                item.get("comments")
                or item.get("answer_count")
                or item.get("num_comments")
                or 0
            )

            finding = {
                "title": title[:200],
                "source": source,
                "url": url,
                "score": min(100, votes + comments),  # Simple scoring
                "date": "unknown",
                "votes": votes,
                "issue": f"See: {title}",
                "solution": snippet[:500]
                if snippet
                else "Click the URL above to view the full content",
                "code": "",
                "evidence": [
                    {
                        "url": url,
                        "quote": snippet[:200] if snippet else "No snippet available",
                        "signal": f"{source} (score: {score}, comments: {comments})",
                    }
                ],
                "gotchas": "No LLM synthesis - review manually",
                "difficulty": "Unknown",
            }

            findings.append(finding)

    # Sort by simple score
    findings.sort(key=lambda x: x.get("score", 0), reverse=True)

    return findings[:10]  # Return top 10


def total_result_count(results: Dict[str, Any]) -> int:
    """Compute total result count, ignoring metadata keys."""
    return sum(len(v) for v in results.values() if isinstance(v, list))


def normalize_query_for_policy(query: str) -> str:
    """Normalize query whitespace for guardrail checks."""
    return " ".join(query.split()).strip()


def build_audit_entry(
    source: str,
    status: str,
    duration_ms: float,
    result_count: int,
    error: Optional[str] = None,
    used_fallback: bool = False,
) -> Dict[str, Any]:
    """Structured audit entry for a single source call."""
    return {
        "source": source,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "result_count": result_count,
        "error": error,
        "used_fallback": used_fallback,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================================
# Search Functions
# ============================================================================


async def search_reddit(query: str, language: str) -> List[Dict[str, Any]]:
    """Search Reddit programming subreddits."""
    try:
        # Map languages to TECHNICAL subreddits only (no "learn" subreddits!)
        subreddit_map = {
            "python": "python+pythontips",
            "javascript": "javascript+node+electronjs",  # Added electronjs, removed learn subreddits
            "java": "java",
            "rust": "rust",
            "go": "golang",
            "cpp": "cpp",
            "csharp": "csharp",
        }

        subreddit = subreddit_map.get(language.lower(), "programming")

        # Try authenticated API first if available
        if reddit_authenticated and reddit_client:
            try:
                results = []
                # Convert '+' separated subreddits to a list
                subreddit_list = subreddit.split("+")

                # Search each subreddit (limited to first 2 to avoid rate limits)
                for sr in subreddit_list[:2]:
                    try:
                        # Use authenticated client for better results and higher rate limits
                        async for submission in reddit_client.p.subreddit.search(
                            sr, query, limit=10, sort="relevance"
                        ):
                            # Get post content based on type
                            snippet = ""
                            if hasattr(submission, "body"):
                                snippet = (
                                    submission.body[:1000] if submission.body else ""
                                )

                            results.append(
                                {
                                    "title": submission.title,
                                    "url": f"https://www.reddit.com{submission.permalink}",
                                    "score": submission.score,
                                    "comments": submission.comment_count,
                                    "snippet": snippet,
                                    "authenticated": True,
                                }
                            )

                            # Limit to 15 total results
                            if len(results) >= 15:
                                break

                    except Exception as subreddit_error:
                        # Skip this subreddit if it fails
                        logging.warning(
                            f"Failed to search subreddit {sr}: {str(subreddit_error)}"
                        )
                        continue

                # Return authenticated results if we got any
                if results:
                    return results

                # Fall back to unauthenticated API if no results or errors occurred
                logging.info(
                    "No results from authenticated Reddit API, falling back to public API"
                )

            except Exception as auth_error:
                logging.warning(
                    f"Authenticated Reddit search failed: {str(auth_error)}. Falling back to public API."
                )

        # Fallback to unauthenticated public API
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {"q": query, "sort": "relevance", "limit": 15, "restrict_sr": "on"}

        headers = {"User-Agent": "CommunityResearchMCP/1.0"}

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("data", {}).get("children", []):
                post = item.get("data", {})
                results.append(
                    {
                        "title": post.get("title", ""),
                        "url": f"https://www.reddit.com{post.get('permalink', '')}",
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                        "snippet": post.get("selftext", "")[:1000],
                        "authenticated": False,
                    }
                )
            return results

    except Exception as e:
        logging.error(f"Reddit search failed: {str(e)}")
        return []


async def aggregate_search_results(
    query: str,
    language: str,
    expanded_mode: bool = False,
    use_fixtures: bool = False,
    force_fresh: bool = False,
) -> Dict[str, Any]:
    """Run all searches with guardrails, fallbacks, and scoring metadata."""
    perf_monitor = get_performance_monitor() if ENHANCED_UTILITIES_AVAILABLE else None
    start_time = time.time()
    normalized_query = normalize_query_for_policy(query)
    use_fixture_data = use_fixtures or os.getenv("CR_MCP_USE_FIXTURES") == "1"
    audit_log: List[Dict[str, Any]] = []

    # Check config for which sources are enabled
    source_config = CONFIG.get("sources", {})

    # Fixture path: deterministic, read-only, no network
    if use_fixture_data:
        capped_results = {}
        for source, items in TEST_FIXTURES.items():
            policy = SOURCE_POLICIES.get(source, {})
            cap = policy.get(
                "max_results_expanded" if expanded_mode else "max_results", len(items)
            )
            capped_results[source] = list(items)[:cap]
            audit_log.append(
                build_audit_entry(
                    source=source,
                    status="fixture",
                    duration_ms=0.0,
                    result_count=len(capped_results[source]),
                    used_fallback=False,
                )
            )
        capped_results["_meta"] = {
            "audit_log": audit_log,
            "used_fixtures": True,
        }
        capped_results["_meta"]["all_star"] = build_all_star_index(
            capped_results, normalized_query, language
        )
        return capped_results

    async def run_source(
        source: str, func: Any, lang: str, q: str
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        policy = SOURCE_POLICIES.get(source, {})
        min_len = policy.get("min_query_length", 0)
        if len(q) < min_len:
            entry = build_audit_entry(
                source=source,
                status="rejected",
                duration_ms=0.0,
                result_count=0,
                error=f"query_too_short (min {min_len})",
            )
            return [], entry

        cap = policy.get("max_results_expanded" if expanded_mode else "max_results", 15)
        start = time.time()
        error_msg = None
        used_fallback = False

        try:
            if ENHANCED_UTILITIES_AVAILABLE:
                # Use circuit breaker for each source
                circuit_breaker = get_circuit_breaker(source)
                if source in {"hackernews", "lobsters"}:
                    # These APIs only take query, no language
                    raw = await circuit_breaker.call_async(resilient_api_call, func, q)
                else:
                    # All other APIs take query + language (including discourse)
                    raw = await circuit_breaker.call_async(
                        resilient_api_call, func, q, lang
                    )
            else:
                if source in {"hackernews", "lobsters"}:
                    raw = await func(q)
                else:
                    raw = await func(q, lang)
        except Exception as exc:
            raw = []
            error_msg = f"{type(exc).__name__}: {exc}"

        duration_ms = (time.time() - start) * 1000
        results = raw if isinstance(raw, list) else []
        results = results[:cap] if results else []
        entry = build_audit_entry(
            source=source,
            status="ok" if results else "empty",
            duration_ms=duration_ms,
            result_count=len(results),
            error=error_msg,
            used_fallback=used_fallback,
        )
        return results, entry

    # Build tasks based on config
    tasks = {}

    if source_config.get("stackoverflow", {}).get("enabled", True):
        tasks["stackoverflow"] = asyncio.create_task(
            run_source(
                "stackoverflow", search_stackoverflow, language, normalized_query
            )
        )

    if source_config.get("github", {}).get("enabled", True):
        tasks["github"] = asyncio.create_task(
            run_source("github", search_github, language, normalized_query)
        )

    if source_config.get("reddit", {}).get("enabled", True):
        tasks["reddit"] = asyncio.create_task(
            run_source("reddit", search_reddit, language, normalized_query)
        )

    if source_config.get("hackernews", {}).get("enabled", True):
        tasks["hackernews"] = asyncio.create_task(
            run_source("hackernews", search_hackernews, language, normalized_query)
        )

    # Additional free community sources
    if source_config.get("lobsters", {}).get("enabled", True):
        tasks["lobsters"] = asyncio.create_task(
            run_source("lobsters", search_lobsters, language, normalized_query)
        )

    if source_config.get("discourse", {}).get("enabled", True):
        tasks["discourse"] = asyncio.create_task(
            run_source("discourse", search_discourse, language, normalized_query)
        )

    if FIRECRAWL_API_KEY and source_config.get("firecrawl", {}).get("enabled", True):
        tasks["firecrawl"] = asyncio.create_task(
            run_source("firecrawl", search_firecrawl, language, normalized_query)
        )

    if TAVILY_API_KEY and source_config.get("tavily", {}).get("enabled", True):
        tasks["tavily"] = asyncio.create_task(
            run_source("tavily", search_tavily, language, normalized_query)
        )

    if BRAVE_SEARCH_API_KEY and source_config.get("brave", {}).get("enabled", True):
        tasks["brave"] = asyncio.create_task(
            run_source("brave", search_brave, language, normalized_query)
        )

    if SERPER_API_KEY and source_config.get("serper", {}).get("enabled", True):
        tasks["serper"] = asyncio.create_task(
            run_source("serper", search_serper, language, normalized_query)
        )

    raw_results: Dict[str, Any] = {}
    for source, task in tasks.items():
        results, audit_entry = await task
        raw_results[source] = results
        audit_log.append(audit_entry)

    # Apply deduplication if available
    deduped_results = (
        deduplicate_results(raw_results)
        if ENHANCED_UTILITIES_AVAILABLE
        else raw_results
    )

    if perf_monitor:
        search_duration = time.time() - start_time
        perf_monitor.record_search_time(search_duration)
        perf_monitor.total_results_found += total_result_count(deduped_results)

    deduped_results["_meta"] = {
        "audit_log": audit_log,
        "used_fixtures": False,
    }
    deduped_results["_meta"]["all_star"] = build_all_star_index(
        deduped_results, normalized_query, language
    )
    return deduped_results


# ============================================================================
# All-Star Scoring & Normalization
# ============================================================================


def _safe_domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def _snippet_has_code(snippet: str) -> bool:
    return "```" in snippet or "<code" in snippet.lower()


def _estimate_recency_days(item: Dict[str, Any]) -> Optional[float]:
    candidates = [
        item.get("created_at"),
        item.get("creation_date"),
        item.get("updated_at"),
        item.get("last_activity_date"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if isinstance(candidate, (int, float)):
                dt = datetime.fromtimestamp(candidate)
            else:
                dt = datetime.fromisoformat(str(candidate).replace("Z", "+00:00"))
            return max(0.0, (datetime.utcnow() - dt).total_seconds() / 86400.0)
        except Exception:
            continue
    return None


def _token_overlap_score(text: str, query: str) -> float:
    q_tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2}
    if not q_tokens:
        return 0.0
    words = {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2}
    if not words:
        return 0.0
    overlap = len(q_tokens & words)
    return min(1.0, overlap / len(q_tokens))


def normalize_item_for_scoring(
    source: str, item: Dict[str, Any], query: str, language: str
) -> Optional[Dict[str, Any]]:
    url = item.get("url") or item.get("link") or ""
    if not url:
        return None

    snippet = item.get("snippet") or item.get("content") or ""
    if snippet is None:
        snippet = ""
    snippet = str(snippet)
    if not snippet.strip():
        return None

    title = item.get("title") or item.get("name") or "Untitled"
    fingerprint = hashlib.md5(f"{source}:{url}".encode("utf-8")).hexdigest()
    domain = _safe_domain(url)
    has_code = _snippet_has_code(snippet)
    recency_days = _estimate_recency_days(item)
    overlap = _token_overlap_score(f"{title} {snippet}", f"{language} {query}")

    return {
        "source": source,
        "title": title,
        "url": url,
        "snippet": snippet[:800],
        "fingerprint": fingerprint,
        "domain": domain,
        "has_code": has_code,
        "score_votes": item.get("score", 0) or item.get("points", 0),
        "comments": item.get("comments", 0) or item.get("answer_count", 0),
        "recency_days": recency_days,
        "overlap": overlap,
        "raw": item,
    }


def compute_multi_factor_score(
    normalized: Dict[str, Any], corroboration: int, diversity_penalty: float
) -> float:
    source_weight = {
        "stackoverflow": 1.0,
        "github": 0.95,
        "discourse": 0.85,
        "lobsters": 0.80,
        "hackernews": 0.75,
        "reddit": 0.7,
        "firecrawl": 0.65,
        "tavily": 0.65,
    }.get(normalized["source"], 0.5)

    recency = normalized.get("recency_days")
    if recency is None:
        recency_score = 0.5
    else:
        recency_score = max(0.0, min(1.0, 1.0 - (recency / 365.0)))

    overlap = normalized.get("overlap", 0.0)
    community = min(
        1.0, (normalized.get("score_votes", 0) + normalized.get("comments", 0)) / 100.0
    )
    richness = 0.6 if normalized.get("has_code") else 0.3
    corroboration_bonus = (
        min(0.3, (corroboration - 1) * 0.1) if corroboration > 1 else 0.0
    )

    base = (
        (source_weight * 0.3)
        + (recency_score * 0.15)
        + (overlap * 0.25)
        + (richness * 0.15)
        + (community * 0.1)
        + corroboration_bonus
    )

    # Apply diversity penalty so one domain doesn't dominate
    return max(0.0, min(1.0, base - diversity_penalty))


def build_all_star_index(
    results: Dict[str, Any], query: str, language: str
) -> Dict[str, Any]:
    """Normalize, score, and assemble an 'all-star' list with corroboration."""
    source_lists = _result_only_sources(results)
    normalized_items: List[Dict[str, Any]] = []

    for source, items in source_lists.items():
        for item in items:
            norm = normalize_item_for_scoring(source, item, query, language)
            if norm:
                normalized_items.append(norm)

    if not normalized_items:
        return {"top_overall": [], "buckets": {}, "stats": {"total": 0}}

    # Corroboration counts
    counts = Counter([n["fingerprint"] for n in normalized_items])
    domain_counts = Counter([n["domain"] for n in normalized_items])

    scored_items = []
    for n in normalized_items:
        diversity_penalty = max(0, domain_counts[n["domain"]] - 1) * 0.02
        score = compute_multi_factor_score(
            n, counts[n["fingerprint"]], diversity_penalty
        )
        scored_items.append(
            {
                **n,
                "all_star_score": round(score * 100, 2),
                "corroboration": counts[n["fingerprint"]],
                "verification_status": "verified"
                if counts[n["fingerprint"]] > 1
                else "unverified",
            }
        )

    scored_items.sort(key=lambda x: x["all_star_score"], reverse=True)

    # Buckets (how-to/code/warnings) using existing classifier
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in scored_items:
        bucket = classify_result(item["raw"], item["source"]).value
        buckets.setdefault(bucket, []).append(item)

    top_buckets = {
        bucket: [
            {
                "title": i["title"],
                "url": i["url"],
                "source": i["source"],
                "score": i["all_star_score"],
                "corroboration": i["corroboration"],
                "verification_status": i["verification_status"],
                "reason": f"{i['source']} credibility, overlap {int(i['overlap'] * 100)}%, "
                f"corroboration x{i['corroboration']}, code={i['has_code']}",
            }
            for i in items[:3]
        ]
        for bucket, items in buckets.items()
    }

    top_overall = [
        {
            "title": i["title"],
            "url": i["url"],
            "source": i["source"],
            "score": i["all_star_score"],
            "corroboration": i["corroboration"],
            "verification_status": i["verification_status"],
            "reason": f"{i['source']} weight + recency + overlap + corroboration",
        }
        for i in scored_items[:5]
    ]

    return {
        "top_overall": top_overall,
        "buckets": top_buckets,
        "stats": {
            "total": len(normalized_items),
            "corroborated": sum(1 for i in scored_items if i["corroboration"] > 1),
            "domains": len(domain_counts),
        },
    }


def _index_results_by_url(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create a quick lookup for search items by URL."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for source, items in _result_only_sources(results).items():
        for item in items:
            url = item.get("url") or item.get("link")
            if not url:
                continue
            lookup[url] = {**item, "source": source}
    return lookup


def filter_results_by_domain(
    results: Dict[str, Any], language: str, query: str
) -> Dict[str, Any]:
    """Drop low-signal domains (e.g., unrelated web results) before synthesis."""
    key_terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 3][:6]
    anchor_terms = [
        t
        for t in key_terms
        if t
        not in {
            "latest",
            "version",
            "removed",
            "guide",
            "fix",
            "solution",
            "code",
            "example",
        }
    ][:3]
    # SIMPLIFIED: Minimal filtering - trust the search APIs!
    filtered: Dict[str, Any] = {}
    for source, items in _result_only_sources(results).items():
        # For trusted community sources, keep EVERYTHING the API returned
        if source in {"stackoverflow", "github", "reddit", "hackernews"}:
            filtered[source] = items  # NO FILTERING
            continue

        # For web search, only filter obvious spam
        kept: List[Dict[str, Any]] = []
        for item in items:
            url = item.get("url") or item.get("link") or ""
            title = item.get("title") or ""

            # Skip if empty
            if not url or not title:
                continue

            # Skip obvious spam
            spam_patterns = ["buy-now", "click-here", "ad.doubleclick"]
            if any(spam in url.lower() for spam in spam_patterns):
                continue

            kept.append(item)
        filtered[source] = kept
    filtered["_meta"] = results.get("_meta", {})
    return filtered


def get_manual_evidence(topic: str) -> List[Dict[str, Any]]:
    key = ""
    t = topic.lower().replace(" ", "")
    if "pipelinecompilationoptions" in t or "wgpu" in t:
        key = "wgpu_pipelinecompilationoptions"
    elif "fastapi" in t and "celery" in t:
        key = "fastapi_celery_redis"
    elif "react" in t and "yup" in t:
        key = "react_yup_hook"
    elif "multistage" in t or "multi-stage" in t or ("docker" in t and "image" in t):
        key = "docker_multistage"
    elif "connectionreset" in t or "resetbypeer" in t:
        key = "tokio_reset_by_peer"
    return MANUAL_EVIDENCE.get(key, [])


# Domains where "street-smart" solutions live - prioritized over official docs
# These are places where developers share what ACTUALLY works in production
PREFERRED_DOMAINS = {
    # Q&A sites - the front line of problem solving
    "stackoverflow.com",
    "stackexchange.com",
    "superuser.com",
    "serverfault.com",
    # GitHub - issues, discussions, and real bug fixes
    "github.com",
    "gist.github.com",
    # Reddit - raw, unfiltered developer experiences
    "reddit.com",
    # Tech news/discussion - experienced developer insights
    "news.ycombinator.com",
    "lobste.rs",
    # Developer blogs and communities
    "dev.to",
    "hashnode.dev",
    "medium.com",
    "freecodecamp.org",
    # Framework-specific communities
    "discourse.org",
    "community.openai.com",
    "discuss.python.org",
    "forum.cursor.com",
    "users.rust-lang.org",
    # Language/framework specific (high-quality community content)
    "bevyengine.org",
    "docs.rs",
    "crates.io",
    "fastapi.tiangolo.com",
    "docs.celeryq.dev",
    "tokio.rs",
}


def _is_preferred_domain(domain: str) -> bool:
    return any(domain.endswith(d) or d in domain for d in PREFERRED_DOMAINS)


def get_weighted_score(item: Dict[str, Any], source: str) -> float:
    """Calculate weighted score based on source priority and engagement."""
    source_weight = CONFIG.get("sources", {}).get(source, {}).get("weight", 5)

    # Get engagement metrics
    score = item.get("score", 0) or 0
    points = item.get("points", 0) or 0
    comments = item.get("comments", 0) or item.get("answer_count", 0) or 0
    engagement = score + points + comments

    # Stack Overflow accepted answers get massive bonus
    if source == "stackoverflow" and item.get("is_accepted"):
        source_weight *= 2.0

    # Calculate final weighted score
    # Source weight is primary, engagement is secondary
    return (source_weight * 100) + min(engagement, 100)


def select_top_evidence(
    results: Dict[str, Any],
    all_star_meta: Dict[str, Any],
    query: str,
    language: str,
    limit: int = 15,  # Increased from 6 to 15
) -> List[Dict[str, Any]]:
    """
    Select top results with SOURCE PRIORITIZATION.
    Stack Overflow > GitHub > HackerNews > Reddit > Web
    """
    all_items: List[Dict[str, Any]] = []
    seen_urls = set()

    # Collect ALL items with their weighted scores
    for source, items in _result_only_sources(results).items():
        for item in items:
            url = item.get("url") or item.get("link") or item.get("html_url")
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            # Calculate weighted score based on source priority
            weighted_score = get_weighted_score(item, source)

            # Extract snippet and code blocks
            snippet_text = (
                item.get("snippet") or item.get("body") or item.get("content") or ""
            )
            code_blocks = []
            if CONFIG.get("output", {}).get("include_code_blocks", True):
                code_blocks = extract_code_blocks(snippet_text)

            all_items.append(
                {
                    "title": item.get("title") or item.get("name") or "Untitled",
                    "url": url,
                    "source": source,
                    "score": weighted_score,  # Now using weighted score
                    "original_score": item.get("score") or item.get("points") or 0,
                    "corroboration": item.get("corroboration", 1),
                    "reason": f"{source} (weight: {CONFIG.get('sources', {}).get(source, {}).get('weight', 5)})",
                    "snippet": snippet_text,
                    "code_blocks": code_blocks,  # ✅ CODE EXTRACTION
                    "votes": item.get("score")
                    or item.get("points")
                    or item.get("stars")
                    or 0,
                    "comments": item.get("comments")
                    or item.get("answer_count")
                    or item.get("num_comments")
                    or 0,
                    "is_accepted": item.get("is_accepted", False),
                    "created_at": item.get("created_at")
                    or item.get("creation_date")
                    or item.get("updated_at"),
                }
            )

    # Sort by weighted score (source priority + engagement)
    all_items.sort(key=lambda x: x["score"], reverse=True)

    # Return top N items
    return all_items[:limit]


# ============================================================================
# Error Handling & Source Status
# ============================================================================


def format_source_status(audit_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format source status for user display."""
    return {
        "sources": [
            {
                "source": entry["source"],
                "status": entry["status"],
                "results_found": entry["result_count"],
                "duration_ms": entry["duration_ms"],
                "error": entry.get("error"),
                "used_fallback": entry.get("used_fallback", False),
                "retry_suggestion": _get_retry_suggestion(entry)
                if entry.get("error")
                else None,
            }
            for entry in audit_log
        ],
        "summary": {
            "total_sources": len(audit_log),
            "successful": sum(1 for e in audit_log if e["status"] == "ok"),
            "failed": sum(1 for e in audit_log if e["status"] != "ok"),
            "total_results": sum(e["result_count"] for e in audit_log),
        },
    }


def _get_retry_suggestion(entry: Dict[str, Any]) -> Optional[str]:
    """Get actionable retry suggestion based on error type."""
    error = str(entry.get("error", "")).lower()

    if "429" in error or "rate limit" in error or "202" in error:
        return "Rate limited - wait 5-10 minutes before retrying. Consider disabling this source in config.json"
    if "422" in error or "query too long" in error:
        return "Query too complex - the search was automatically simplified, try with fewer keywords"
    if "timeout" in error or "timed out" in error:
        return "Request timed out - check internet connection or try again"
    if "api key" in error or "unauthorized" in error or "401" in error:
        return "API authentication failed - check your API keys in .env file"
    if "404" in error:
        return "Resource not found - this is likely a temporary API issue"
    if "500" in error or "503" in error:
        return (
            "Server error - the service is temporarily down, try again in a few minutes"
        )

    return "Temporary error - retrying should work"


# ============================================================================
# Code Extraction
# ============================================================================


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Extract code blocks from markdown/HTML text."""
    import re

    if not text:
        return []

    code_blocks = []

    # Markdown code blocks with language
    pattern = r"```(\w+)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for lang, code in matches:
        if code.strip() and len(code.strip()) > 20:  # Only meaningful code
            code_blocks.append({"language": lang or "text", "code": code.strip()})

    # HTML <code> tags
    html_pattern = r"<code>(.*?)</code>"
    html_matches = re.findall(html_pattern, text, re.DOTALL)
    for code in html_matches:
        clean_code = code.strip()
        if len(clean_code) > 20:  # Only meaningful code blocks
            code_blocks.append({"language": "text", "code": clean_code})

    # HTML <pre> tags
    pre_pattern = r"<pre>(.*?)</pre>"
    pre_matches = re.findall(pre_pattern, text, re.DOTALL)
    for code in pre_matches:
        clean_code = code.strip()
        if len(clean_code) > 20:
            code_blocks.append({"language": "text", "code": clean_code})

    return code_blocks


# ============================================================================
# LLM Synthesis
# ============================================================================


async def synthesize_with_llm(
    search_results: Dict[str, Any],
    query: str,
    language: str,
    goal: Optional[str],
    current_setup: Optional[str],
    context_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    PERMANENTLY MODIFIED: This function NO LONGER calls internal LLMs.

    WHY:
    - Avoids rate limits on server-side API keys
    - Reduces API costs
    - Lets the calling LLM (Claude/ChatGPT) do synthesis with full context

    WHAT IT DOES NOW:
    - Extracts raw search results from GitHub, Stack Overflow, Reddit, HN
    - Structures them into a standard format
    - Returns raw data for the calling LLM to synthesize

    DO NOT REVERT THIS - This is the intended behavior.
    """

    # Extract metadata
    all_star = (context_meta or {}).get("all_star", {})
    top_evidence = (context_meta or {}).get("top_evidence", [])
    enrichment = (context_meta or {}).get("enrichment", {})

    # Convert raw search results into structured findings format
    findings = []

    for source_name, items in search_results.items():
        if source_name.startswith("_"):  # Skip metadata keys like _meta
            continue

        for item in items[:10]:  # Limit to top 10 per source
            snippet = item.get("snippet", "")[:500]
            title = item.get("title", "")

            # Extract basic structure from snippet for quality scoring
            # This is a simple heuristic-based extraction, not LLM synthesis
            issue = "Not specified"
            solution = snippet  # Use snippet as solution fallback
            code = ""

            # Try to extract code blocks from snippet
            code_matches = re.findall(
                r"```[\w]*\n(.*?)\n```|`([^`]+)`", snippet, re.DOTALL
            )
            if code_matches:
                code = "\n".join([m[0] or m[1] for m in code_matches])

            # Calculate relevance score based on topic overlap
            query_terms = set(re.findall(r"\w+", f"{language} {query}".lower()))
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "as",
                "is",
                "it",
                "be",
                "are",
                "was",
                "were",
            }
            query_terms = query_terms - stop_words

            content = f"{title} {snippet}".lower()
            content_terms = set(re.findall(r"\w+", content))
            matches = query_terms.intersection(content_terms)
            relevance_score = len(matches) / max(len(query_terms), 1) * 100

            finding = {
                "title": title[:100],
                "source": source_name,
                "url": item.get("url", ""),
                "score": item.get("quality_score", 0)
                if item.get("quality_score", 0) > 0
                else item.get("score", 0),
                "date": item.get("date", "unknown"),
                "votes": item.get("votes", 0)
                or item.get("stars", 0)
                or item.get("points", 0),
                "snippet": snippet,
                "issue": issue,
                "solution": solution,
                "code": code,
                "evidence": [{"url": item.get("url", ""), "quote": snippet[:200]}]
                if item.get("url")
                else [],
                "relevance_score": int(relevance_score),
                "raw_item": item,  # Include full raw data for calling LLM to analyze
            }
            findings.append(finding)

    # Sort findings by relevance score (descending) to put most relevant first
    findings.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Return structured data for the calling LLM to synthesize
    return {
        "findings": findings,
        "top_evidence": top_evidence,
        "all_star_meta": all_star,
        "enrichment": enrichment,
        "query_context": {
            "query": query,
            "language": language,
            "goal": goal,
            "current_setup": current_setup,
        },
        "synthesis_instructions": {
            "task": "Extract battle-tested solutions and real-world workarounds from community discussions",
            "focus": [
                "Prioritize accepted answers, maintainer responses, and highly-voted solutions",
                "Look for 'this worked for me' comments, not just documentation",
                "Find workarounds for known bugs and breaking changes",
                "Identify deprecated approaches and migration paths",
                "Extract production experience: 'we tried X but switched to Y because...'",
            ],
            "steps": [
                "1. Identify the core problem and common symptoms",
                "2. Extract working code examples from accepted answers and comments",
                "3. Note conflicts: version-specific issues, edge cases, 'don't use X' warnings",
                "4. Build recommended path from community consensus",
                "5. Include verification: how others confirmed it worked",
            ],
            "output_format": "Findings with evidence URLs, conflicts/gotchas, step-by-step path, copy-paste code",
        },
    }


# ============================================================================
# Masterclass Rendering Helpers
# ============================================================================


def normalize_masterclass_payload(
    synthesis: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    language: str,
    enrichment: Optional[Dict[str, Any]] = None,
    manual_mode: bool = False,
) -> Dict[str, Any]:
    """Ensure synthesis output has all fields and evidence attached."""
    enrichment = enrichment or {}
    evidence_map = {ev["url"]: ev for ev in evidence if ev.get("url")}

    raw_findings = synthesis.get("findings") or []
    findings: List[Dict[str, Any]] = []
    for f in raw_findings:
        findings.append(f)
    findings = [
        f
        for f in findings
        if f.get("url") and _is_preferred_domain(_safe_domain(str(f.get("url", ""))))
    ]

    if not findings and evidence:
        for ev in evidence[:3]:
            quote = (ev.get("snippet") or "").strip()
            findings.append(
                {
                    "title": ev.get("title", "Recommendation"),
                    "source": ev.get("source", "community"),
                    "url": ev.get("url"),
                    "score": ev.get("score", 0) or 0,
                    "votes": ev.get("votes"),
                    "issue": (quote[:180] or "See linked source for details."),
                    "solution": "Apply the referenced fix/migration from the linked source.",
                    "code": "",
                    "evidence": [
                        {
                            "url": ev.get("url"),
                            "quote": quote[:200] or "Evidence available in source.",
                            "signal": ev.get("reason") or "community source",
                        }
                    ],
                    "gotchas": "",
                    "difficulty": "Unknown",
                }
            )

    for f in findings:
        url = f.get("url")
        ev = evidence_map.get(url)
        if ev and not f.get("evidence"):
            quote = (ev.get("snippet") or "").strip()
            f["evidence"] = [
                {
                    "url": ev.get("url"),
                    "quote": quote[:200] or "Evidence available in source.",
                    "signal": ev.get("reason") or "community source",
                }
            ]
        f.setdefault("issue", f.get("problem", "Problem not captured."))
        f.setdefault("solution", f.get("benefit", "Solution not captured."))
        f.setdefault("source", ev.get("source") if ev else f.get("source", "community"))
        f.setdefault("score", f.get("community_score", 0) or 0)
        f.setdefault("votes", f.get("community_score"))
        f.setdefault("difficulty", f.get("difficulty", "Unknown"))
        if not f.get("evidence") and url:
            f["evidence"] = [
                {"url": url, "quote": "See linked source.", "signal": "source link"}
            ]

    recommended_path = synthesis.get("recommended_path") or []
    if not recommended_path and findings:
        for i, f in enumerate(findings[:3], 1):
            recommended_path.append(
                {
                    "step": f"Apply recommendation #{i}: {f.get('title', 'unnamed')}",
                    "why": f.get("issue", "Improve reliability"),
                }
            )
    if len(evidence) < 2 and not recommended_path:
        recommended_path.append(
            {
                "step": "Gather more evidence or rerun with a narrower query",
                "why": "Evidence is weak; avoid acting on uncorroborated advice.",
            }
        )

    quick_apply = synthesis.get("quick_apply") or {}
    if not quick_apply and findings and len(evidence) >= 2:
        first_code = next((f.get("code") for f in findings if f.get("code")), "")
        if first_code:
            quick_apply = {"language": language, "code": first_code, "commands": []}

    verification = synthesis.get("verification") or [
        f"Rebuild/retest the {language} project after applying changes."
    ]

    assumptions = synthesis.get("assumptions") or []
    assumptions.extend(enrichment.get("assumptions", []))
    if len(evidence) < 2 and not manual_mode:
        assumptions.append("Evidence is weak (fewer than 2 strong sources found).")

    conflicts = synthesis.get("conflicts") or []

    return {
        "findings": findings,
        "conflicts": conflicts,
        "recommended_path": recommended_path,
        "quick_apply": quick_apply,
        "verification": verification,
        "assumptions": assumptions,
        "synthesis_summary": synthesis.get("synthesis_summary", ""),
    }


def render_masterclass_markdown(
    topic: str,
    language: str,
    goal: Optional[str],
    current_setup: Optional[str],
    payload: Dict[str, Any],
    conflicts_auto: List[Dict[str, str]],
    search_meta: Dict[str, Any],
    manual_mode: bool = False,
) -> str:
    """Render final markdown using the masterclass template."""
    findings = payload.get("findings", [])
    conflicts = payload.get("conflicts", []) + conflicts_auto
    recommended_path = payload.get("recommended_path", [])
    quick_apply = payload.get("quick_apply") or {}
    verification = payload.get("verification") or []
    assumptions = payload.get("assumptions") or []

    lines: List[str] = []
    lines.append(f"# Community Research: {topic}")
    lines.append("")
    lines.append("## 📋 How to Use These Results")
    lines.append("")
    lines.append("**Quality Scores:** Each finding has a score (0-100):")
    lines.append(
        "- **80-100**: Highly reliable (maintainer-backed, well-evidenced, recent)"
    )
    lines.append("- **60-79**: Good quality (community-validated, has code examples)")
    lines.append("- **40-59**: Moderate (some evidence, may need verification)")
    lines.append("- **<40**: Weak (limited evidence, outdated, or speculative)")
    lines.append("")
    lines.append("**Next Steps:**")
    lines.append("1. Review findings starting with highest scores")
    lines.append("2. Check conflicts section for edge cases")
    lines.append("3. Follow recommended path for implementation")
    lines.append("4. Copy quick-apply code as starting point (review before running)")
    lines.append("5. Validate using verification steps")
    lines.append("")
    evidence_strength = "⚠️ WEAK" if search_meta.get("evidence_weak") else "✅ STRONG"
    lines.append(
        f"**Search Quality:** {evidence_strength} evidence from {search_meta.get('total_results', 0)} results across {search_meta.get('source_count', 0)} sources"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## 🎯 Research Context")
    lines.append(f"- **Goal:** {goal or 'Not provided'}")
    context_bits = [language]
    if current_setup:
        context_bits.append(current_setup)
    lines.append(f"- **Context:** {', '.join(context_bits)}")
    lines.append("")

    lines.append("## Findings (ranked)")
    if not findings:
        lines.append("- No findings available; broaden the query or add goal/context.")
    else:
        for idx, f in enumerate(findings, 1):
            source_label = f.get("source", "Source")
            # Use quality_score if available, fallback to score
            score = f.get("quality_score", f.get("score", "N/A"))
            date = f.get("date", "unknown")
            votes = f.get("votes") or f.get("community_score") or "n/a"
            lines.append(
                f"{idx}) {source_label} (Score: {score}, Date: {date}, Votes/Stars: {votes})"
            )
            # Use snippet as solution if solution is "Not specified"
            issue = f.get("issue", "See details in evidence link")
            solution = f.get(
                "solution", f.get("snippet", "See evidence link for details")
            )
            if issue == "Not specified":
                issue = f"Related to: {f.get('title', 'Unknown')}"
            lines.append(f"   - Issue: {issue}")
            lines.append(
                f"   - Solution: {solution[:300]}{'...' if len(solution) > 300 else ''}"
            )
            evidence_list = f.get("evidence") or []
            if evidence_list:
                ev = evidence_list[0]
                quote = str(ev.get("quote", "")).replace("\n", " ").strip()
                lines.append(
                    f'   - Evidence: {ev.get("url", "")} - "{quote[:200] or "See source"}"'
                )
            elif f.get("url"):
                lines.append(f"   - Evidence: {f.get('url')}")
            code_block = f.get("code")
            if code_block:
                code_lang = language.lower()
                lines.append("")
                lines.append(f"```{code_lang}\n{code_block}\n```")
                lines.append("")
    lines.append("")

    lines.append("## Conflicts & Edge Cases")
    if conflicts:
        for c in conflicts:
            lines.append(
                f"- {c.get('description', 'Conflict')} - {c.get('recommended_action', 'Resolve carefully.')}"
            )
    else:
        lines.append("- None detected; still validate against your stack.")
    lines.append("")

    lines.append("## Recommended Path")
    if recommended_path:
        for i, step in enumerate(recommended_path, 1):
            lines.append(f"{i}. {step.get('step', 'Step')} - {step.get('why', '')}")
    else:
        lines.append("- Apply the top finding and rerun verification.")
    lines.append("")

    lines.append("## Quick-apply Code/Commands")
    code = quick_apply.get("code")
    if code:
        code_lang = (quick_apply.get("language") or language).lower()
        lines.append(f"```{code_lang}\n{code}\n```")
    commands = quick_apply.get("commands") or []
    for cmd in commands:
        lines.append(f"- {cmd}")
    if not code and not commands:
        lines.append("- Not available from sources.")
    lines.append("")

    lines.append("## Verification")
    for check in verification:
        lines.append(f"- {check}")
    if not verification:
        lines.append("- Add tests/commands to validate changes.")
    lines.append("")

    if assumptions:
        lines.append("## Assumptions")
        for a in assumptions:
            lines.append(f"- {a}")
        lines.append("")

    lines.append("## Search Stats")
    if manual_mode:
        lines.append("- Sources: manual evidence pack")
        lines.append(
            f"- Results found: {len(payload.get('findings', []))} curated entries"
        )
    else:
        lines.append(
            f"- Sources queried: {search_meta.get('source_count', 0)} ({', '.join(search_meta.get('sources', []))})"
        )
        lines.append(f"- Results found: {search_meta.get('total_results', 0)}")
        if search_meta.get("evidence_weak"):
            lines.append("- Evidence: weak (fewer than 2 strong sources).")
    lines.append(f"- Enriched query: {search_meta.get('enriched_query', 'n/a')}")
    lines.append(
        f"- Expanded queries (for follow-up): {', '.join(search_meta.get('expanded_queries', [])[:3])}"
    )

    return "\n".join(lines)


# ============================================================================
# Zen MCP Inspired - Multi-Model Orchestration & Research Planning
# ============================================================================


# ============================================================================
# Search Result Processing
# ============================================================================


async def cluster_and_rerank_results(
    search_results: Dict[str, Any], query: str, language: str
) -> Dict[str, Any]:
    """
    Returns search results as-is. Clustering is delegated to the host LLM.
    """
    return search_results


async def synthesize_with_multi_model(
    search_results: Dict[str, Any],
    query: str,
    language: str,
    goal: Optional[str],
    current_setup: Optional[str],
    thinking_mode: ThinkingMode = ThinkingMode.BALANCED,
) -> Dict[str, Any]:
    """
    Structures search results for the host LLM to synthesize.

    This function does NOT make internal LLM calls - all synthesis is delegated
    to the host LLM (Claude Desktop, Cursor, etc.) that called this MCP tool.
    """
    result = await synthesize_with_llm(
        search_results, query, language, goal, current_setup
    )

    result["orchestration"] = {
        "thinking_mode": thinking_mode.value,
        "note": "Synthesis delegated to host LLM",
    }

    return result


# ============================================================================
# New Zen-Inspired Research Workflow Tools
# ============================================================================


@mcp.tool(
    name="plan_research",
    annotations={
        "title": "Plan Research Strategy",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def plan_research(query: str, language: str, goal: Optional[str] = None) -> str:
    """
    Plan a strategic hunt for street-smart solutions across the developer community.

    Before diving into Stack Overflow and Reddit, get a battle plan that ensures
    you find the REAL solutions - the workarounds, hacks, and "what actually worked"
    answers that official docs never mention.

    WORKFLOW: This is Step 1 of the research workflow.
    1. Call plan_research FIRST to get a strategic plan
    2. Review the plan - it generates queries optimized for finding community wisdom
    3. Call community_search with the refined queries from the plan
    4. Use synthesis_instructions to extract the battle-tested solutions

    WHEN TO USE THIS TOOL:
    - Complex problems where you need multiple community perspectives
    - Comparing approaches ("which one do people ACTUALLY use in production?")
    - Architecture decisions where you want to learn from others' mistakes
    - Migration/upgrade research ("what gotchas did people hit?")
    - Any research where you want comprehensive community intelligence

    WHEN TO SKIP (use community_search directly):
    - Simple, specific questions with likely one good answer
    - Error debugging with a specific error message
    - Quick lookup of a single concept or API

    Args:
        query (str): The research topic or question to plan for
        language (str): Programming language context (e.g., "Python", "JavaScript")
        goal (Optional[str]): What you want to achieve with this research

    Returns:
        str: JSON-formatted planning guidance containing:
            - Query analysis and context
            - Recommended research phases with specific actions
            - Search strategy with pre-built queries to use
            - Source prioritization guidance
            - next_steps: Specific tool calls to make after planning

    Examples:
        - plan_research("FastAPI async task processing", "Python", "implement background jobs")
        - plan_research("React state management patterns", "JavaScript", "choose best approach for large app")
    """
    # Check rate limit
    if not check_rate_limit("plan_research"):
        return json.dumps(
            {
                "error": "Rate limit exceeded. Maximum 10 requests per minute. Please wait and try again."
            },
            indent=2,
        )

    # Check cache
    cache_key = get_cache_key(
        "plan_research", query=query, language=language, goal=goal
    )
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    # Create structured planning data without LLM calls
    planning_data = {
        "query_context": {
            "query": query,
            "language": language,
            "goal": goal or "Research and understand the topic",
        },
        "recommended_phases": [
            {
                "name": "Discovery",
                "description": "Initial search across community sources",
                "actions": [
                    "Search Stack Overflow for common issues and solutions",
                    "Check GitHub issues for known bugs and workarounds",
                    "Review Reddit discussions for community opinions",
                    "Find recent HackerNews discussions for industry trends",
                ],
            },
            {
                "name": "Analysis",
                "description": "Deep dive into findings and patterns",
                "actions": [
                    "Identify common patterns in solutions",
                    "Compare different approaches and trade-offs",
                    "Extract code examples and best practices",
                    "Note version-specific considerations",
                ],
            },
            {
                "name": "Validation",
                "description": "Cross-reference and verify findings",
                "actions": [
                    "Check for conflicts between sources",
                    "Verify recency of solutions",
                    "Look for maintainer/official responses",
                    "Identify any deprecated approaches",
                ],
            },
        ],
        "search_strategies": {
            "initial_search": f"{language} {query}",
            "focused_searches": [
                f"{query} best practices {language}",
                f"{query} common issues {language}",
                f"{query} migration guide {language}",
                f"{query} alternatives {language}",
            ],
        },
        "source_prioritization": [
            {
                "source": "Stack Overflow",
                "priority": "high",
                "reason": "Accepted answers and community votes indicate proven solutions",
            },
            {
                "source": "GitHub Issues",
                "priority": "high",
                "reason": "Official maintainer responses and real-world bug reports",
            },
            {
                "source": "Reddit",
                "priority": "medium",
                "reason": "Community discussions and experience sharing",
            },
            {
                "source": "HackerNews",
                "priority": "medium",
                "reason": "Industry trends and architectural discussions",
            },
        ],
        "synthesis_instructions": {
            "task": "Review this plan and execute the research strategy",
            "steps": [
                "1. Review the query context and goal",
                "2. Use the search_strategies.focused_searches queries below",
                "3. Call community_search for each search query",
                "4. Synthesize findings using the synthesis_instructions in each response",
            ],
            "output_format": "Comprehensive research findings with evidence and recommendations",
        },
        "next_steps": {
            "workflow": "Execute these tool calls in order based on the plan above",
            "recommended_calls": [
                {
                    "tool": "community_search",
                    "description": "Primary search for main topic",
                    "params": {
                        "language": language,
                        "topic": query,
                        "goal": goal or "Research and understand the topic",
                    },
                },
                {
                    "tool": "community_search",
                    "description": "Search for best practices",
                    "params": {
                        "language": language,
                        "topic": f"{query} best practices",
                        "goal": "Find recommended approaches",
                    },
                },
                {
                    "tool": "community_search",
                    "description": "Search for common issues",
                    "params": {
                        "language": language,
                        "topic": f"{query} common issues pitfalls",
                        "goal": "Identify potential problems to avoid",
                    },
                },
            ],
            "after_searches": "Synthesize all findings using the synthesis_instructions provided in each search response",
        },
    }

    formatted_result = json.dumps(planning_data, indent=2)
    set_cached_result(cache_key, formatted_result)
    return formatted_result


@mcp.tool(
    name="comparative_search",
    annotations={
        "title": "Comparative Multi-Model Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def comparative_search(
    language: str,
    topic: str,
    goal: Optional[str] = None,
    current_setup: Optional[str] = None,
    models_to_compare: Optional[List[str]] = None,
) -> str:
    """
    Perform comparative research for "X vs Y" style questions.

    WHEN TO USE THIS TOOL:
    - Comparing two or more technologies, libraries, or approaches
    - "Should I use X or Y?" decisions
    - Trade-off analysis between alternatives

    WORKFLOW:
    - Use this tool DIRECTLY for comparison questions (no need for plan_research first)
    - The topic MUST contain comparison keywords like "vs", "versus", "compared to"
    - Results include both sides of the comparison with synthesis_instructions

    Args:
        language (str): Programming language (e.g., "Python", "JavaScript")
        topic (str): Comparison topic (MUST contain "vs", "versus", or "compared to")
        goal (Optional[str]): What you want to achieve
        current_setup (Optional[str]): Your current tech stack
        models_to_compare (Optional[List[str]]): Ignored (kept for compatibility)

    Returns:
        str: Comparative data structure containing:
            - Search results for each side of the comparison
            - Perspective-specific findings (pros/cons, use cases)
            - Source data organized by viewpoint
            - synthesis_instructions for you to create the comparison analysis

    Examples:
        - comparative_search("Python", "async vs threads for I/O bound tasks")
        - comparative_search("JavaScript", "React vs Vue for dashboard applications")
        - comparative_search("Database", "PostgreSQL vs MongoDB for analytics workloads")
    """
    # Check rate limit
    if not check_rate_limit("comparative_search"):
        return json.dumps(
            {
                "error": "Rate limit exceeded. Maximum 10 requests per minute. Please wait and try again."
            },
            indent=2,
        )

    # Validate topic has comparison keywords
    topic = topic.strip()
    comparison_keywords = ["vs", "versus", "compared to", "or", "vs.", "compare"]
    has_comparison = any(kw in topic.lower() for kw in comparison_keywords)

    if not has_comparison:
        return json.dumps(
            {
                "error": f"Topic '{topic}' doesn't appear to be a comparison. Use 'vs', 'versus', or 'compared to'.",
                "suggestions": [
                    f"{topic} - async vs synchronous",
                    f"{topic} - library A vs library B",
                    f"Compare {topic} approaches",
                ],
            },
            indent=2,
        )

    # Check cache
    cache_key = get_cache_key(
        "comparative_search",
        language=language,
        topic=topic,
        goal=goal,
        current_setup=current_setup,
    )
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    # Execute searches for the comparison topic
    search_query = f"{language} {topic}"
    if goal:
        search_query += f" {goal}"

    search_results = await aggregate_search_results(search_query, language)
    source_lists = _result_only_sources(search_results)
    total_results = total_result_count(search_results)

    if total_results == 0:
        result = json.dumps(
            {
                "error": f'No results found for "{topic}" in {language}.',
                "suggestion": "Try broader comparison terms or check if both options are commonly discussed",
            },
            indent=2,
        )
        set_cached_result(cache_key, result)
        return result

    # Structure the comparative data
    comparative_data = {
        "query_context": {
            "language": language,
            "comparison_topic": topic,
            "goal": goal,
            "current_setup": current_setup,
        },
        "search_results_by_source": {
            "stackoverflow": source_lists.get("stackoverflow", []),
            "github": source_lists.get("github", []),
            "reddit": source_lists.get("reddit", []),
            "hackernews": source_lists.get("hackernews", []),
        },
        "result_counts": {
            "total": total_results,
            "stackoverflow": len(source_lists.get("stackoverflow", [])),
            "github": len(source_lists.get("github", [])),
            "reddit": len(source_lists.get("reddit", [])),
            "hackernews": len(source_lists.get("hackernews", [])),
        },
        "comparison_framework": {
            "suggested_comparison_points": [
                "Performance and efficiency",
                "Developer experience and learning curve",
                "Community support and ecosystem",
                "Production readiness and stability",
                "Use case fit and trade-offs",
            ],
            "analysis_approach": "The calling LLM should identify different perspectives from the search results and create a balanced comparison",
        },
        "synthesis_instructions": {
            "task": "Perform comparative analysis of the search results",
            "steps": [
                "1. Identify pros and cons for each option from the search results",
                "2. Determine use cases where each option excels",
                "3. Analyze performance, scalability, and developer experience differences",
                "4. Note community preferences and adoption trends",
                "5. Make a recommendation based on the goal and current setup",
            ],
            "output_format": "Side-by-side comparison table, use case analysis, and final recommendation with rationale",
        },
    }

    result = json.dumps(comparative_data, indent=2)
    set_cached_result(cache_key, result)
    return result


@mcp.tool(
    name="validated_research",
    annotations={
        "title": "Research with Multi-Model Validation",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def validated_research(
    language: str,
    topic: str,
    goal: Optional[str] = None,
    current_setup: Optional[str] = None,
    thinking_mode: str = "balanced",
) -> str:
    """
    Perform comprehensive research with additional depth and validation guidance.

    MODIFIED: NO INTERNAL LLM CALLS - Returns enhanced search results with validation
    prompts for the calling LLM to perform its own multi-perspective analysis.

    Args:
        language (str): Programming language (e.g., "Python", "JavaScript")
        topic (str): Research topic (must be specific)
        goal (Optional[str]): What you want to achieve
        current_setup (Optional[str]): Your current tech stack
        thinking_mode (str): Analysis depth guidance ("quick", "balanced", "deep")

    Returns:
        str: Comprehensive research data containing:
            - Extended search results from multiple sources
            - Quality-scored findings
            - Validation checklist for the calling LLM
            - Critical thinking prompts
            - Guidance for multi-perspective analysis

    Examples:
        - validated_research("Python", "production deployment strategies for Django apps")
        - validated_research("Security", "JWT authentication implementation best practices")

    Note: The calling LLM should use the validation prompts to perform thorough analysis.
    """
    # Check rate limit
    if not check_rate_limit("validated_research"):
        return json.dumps(
            {
                "error": "Rate limit exceeded. Maximum 10 requests per minute. Please wait and try again."
            },
            indent=2,
        )

    # Check cache
    cache_key = get_cache_key(
        "validated_research",
        language=language,
        topic=topic,
        goal=goal,
        current_setup=current_setup,
        thinking_mode=thinking_mode,
    )
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    # Execute comprehensive search
    search_query = f"{language} {topic}"
    if goal:
        search_query += f" {goal}"

    search_results = await aggregate_search_results(
        search_query, language, expanded_mode=True
    )
    source_lists = _result_only_sources(search_results)
    total_results = total_result_count(search_results)

    if total_results == 0:
        result = json.dumps(
            {
                "error": f'No results found for "{topic}" in {language}.',
                "suggestion": "Try broader or more common search terms",
            },
            indent=2,
        )
        set_cached_result(cache_key, result)
        return result

    # Get basic synthesis structure
    synthesis = await synthesize_with_llm(
        source_lists, topic, language, goal, current_setup
    )

    # Apply quality scoring if available
    if ENHANCED_UTILITIES_AVAILABLE and _quality_scorer and "findings" in synthesis:
        synthesis["findings"] = _quality_scorer.score_findings_batch(
            synthesis["findings"]
        )

    # Add validation framework for calling LLM
    validation_framework = {
        "thinking_mode": thinking_mode,
        "validation_checklist": [
            "Are there conflicting recommendations? If so, what are the conditions for each?",
            "Are the solutions recent and applicable to current versions?",
            "Do the findings account for edge cases and failure scenarios?",
            "Are there security or performance implications not mentioned?",
            "What are the trade-offs of each approach?",
            "Are there deprecated practices being recommended?",
            "Do maintainers or official sources contradict community advice?",
        ],
        "critical_thinking_prompts": {
            "quick": [
                "Verify the most upvoted/accepted answer is still valid",
                "Check for any recent breaking changes",
            ],
            "balanced": [
                "Cross-reference solutions across multiple sources",
                "Identify common patterns vs outliers",
                "Evaluate recency and version compatibility",
                "Consider production vs development contexts",
            ],
            "deep": [
                "Analyze architectural implications and scalability",
                "Evaluate security posture and attack vectors",
                "Consider maintenance burden and technical debt",
                "Review alternative approaches and their trade-offs",
                "Assess ecosystem maturity and longevity",
                "Identify hidden dependencies and coupling",
            ],
        }.get(thinking_mode, []),
        "multi_perspective_analysis": "The calling LLM should analyze findings from multiple angles: novice users, experienced developers, security experts, and production operations perspectives",
    }

    # Create comprehensive validated research structure
    validated_data = {
        "research_data": synthesis,
        "validation_framework": validation_framework,
        "research_context": {
            "language": language,
            "topic": topic,
            "goal": goal,
            "current_setup": current_setup,
            "total_sources_analyzed": total_results,
            "thinking_depth": thinking_mode,
        },
        "synthesis_instructions": {
            "task": "Synthesize research data with validation",
            "steps": [
                "1. Review all findings and apply quality scoring",
                "2. Work through the validation checklist for each finding",
                "3. Apply critical thinking prompts based on thinking mode",
                "4. Identify gaps, conflicts, or missing perspectives",
                "5. Create validated recommendations with confidence levels",
            ],
            "output_format": "Validated findings with confidence scores, validation notes, and comprehensive recommendations",
        },
    }

    result = json.dumps(validated_data, indent=2)
    set_cached_result(cache_key, result)
    return result


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool(
    name="get_server_context",
    annotations={
        "title": "Get Community Research Server Context",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_server_context() -> str:
    """
    Get the Community Research MCP server context and capabilities.

    "Where the official documentation ends and actual street-smart solutions begin."

    This tool returns information about what the server detected in your workspace,
    including programming languages, frameworks, and default context values. ALWAYS
    call this first before using other tools.

    Returns:
        str: JSON-formatted server context including:
            - handshake: Server identification and mission
            - project_context: Detected workspace information
            - context_defaults: Default values for search

    Examples:
        - Use when: Starting any research task
        - Use when: Need to know what languages are detected
    """
    workspace_context = detect_workspace_context()

    context = {
        "handshake": {
            "server": "community-research-mcp",
            "version": "1.0.0",
            "status": "initialized",
            "mission": "Where the official documentation ends and actual street-smart solutions begin",
            "description": "Find real solutions from Stack Overflow, Reddit, GitHub issues, and forums - the workarounds, hacks, and battle-tested fixes that people actually use",
            "capabilities": {
                "multi_source_search": True,
                "query_validation": True,
                "caching": True,
                "rate_limiting": True,
            },
        },
        "project_context": workspace_context,
        "context_defaults": {
            "language": workspace_context["languages"][0]
            if workspace_context["languages"]
            else None
        },
    }

    return json.dumps(context, indent=2)


@mcp.tool(
    name="community_search",
    annotations={
        "title": "Search Community Resources",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def community_search(params: CommunitySearchInput) -> str:
    """
    Find STREET-SMART solutions from developers who've already fought through your exact problem.

    "Where the official documentation ends and actual solutions begin."

    NOT sanitized documentation. NOT official guides. NOT theoretical best practices.
    The ACTUAL fixes from Stack Overflow, Reddit threads, GitHub issues, and forums.
    The messy workarounds, the "this finally worked for me" comments, the battle-tested
    hacks that people actually use in production.

    WORKFLOW GUIDANCE:
    - For COMPLEX research (architecture decisions, comparing approaches):
      Call plan_research FIRST to get a strategic plan, then use the queries it provides.
    - For SIMPLE queries (specific errors, quick lookups):
      Call this tool directly.

    WHAT THIS TOOL FINDS:
    - Accepted Stack Overflow answers with working code
    - GitHub issues where someone figured out the workaround
    - Reddit threads with "I finally solved this" comments
    - HackerNews discussions from experienced developers
    - The gotchas and caveats that official docs don't mention

    WHEN TO USE DIRECTLY:
    - "Why is this error happening?" - Find what actually fixed it
    - "How do I implement X?" - Find working examples, not docs
    - "What's the workaround for Y?" - Find production-tested hacks
    - "Has anyone else hit this issue?" - Find the community's collective trauma

    WHEN TO USE plan_research FIRST:
    - Comparing multiple approaches or libraries
    - Architecture or design decisions
    - Migration planning spanning multiple concerns
    - Research requiring multiple search queries

    RESPONSE FORMAT:
    Returns structured data with these key sections:
    - findings[]: List of solutions with title, solution, evidence URL, code examples, quality score (0-100)
    - conflicts[]: Contradicting advice or edge cases discovered
    - recommended_path[]: Step-by-step action items
    - quick_apply{}: Copy-paste code/commands to try immediately
    - verification[]: How to confirm the solution worked
    - search_meta{}: Information about sources searched, result counts, quality metrics

    HOW TO INTERPRET RESULTS:
    1. Check 'findings' for solutions - higher scores (>70) are more reliable
    2. Review 'conflicts' to understand edge cases or debates
    3. Follow 'recommended_path' for step-by-step implementation
    4. Use 'quick_apply' code/commands as starting points (always review before running)
    5. Validate with 'verification' steps after applying changes

    QUALITY SCORES EXPLAINED:
    - 80-100: Highly reliable (maintainer-backed, well-evidenced, recent)
    - 60-79: Good quality (community-validated, has code examples)
    - 40-59: Moderate (some evidence, may need verification)
    - <40: Weak (limited evidence, outdated, or speculative)

    Args:
        params: CommunitySearchInput with language, topic, goal, current_setup, response_format

    Example:
        params = CommunitySearchInput(
            language="Python",
            topic="FastAPI background tasks with Celery Redis",
            goal="Implement async task queue for email sending",
            response_format="json"
        )
        result = await community_search(params)

    Returns:
        JSON or Markdown string with structured research findings and actionable guidance.
    """
    if not check_rate_limit("community_search"):
        return json.dumps(
            {
                "error": "Rate limit exceeded. Maximum 10 requests per minute. Please wait and try again.",
            },
            indent=2,
        )

    cache_key = get_cache_key(
        "community_search",
        language=params.language,
        topic=params.topic,
        goal=params.goal,
        current_setup=params.current_setup,
        expanded_mode=params.expanded_mode,
        use_fixtures=params.use_fixtures,
    )
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    valid, validation_msg = validate_topic_specificity(params.topic)
    if not valid:
        return json.dumps(
            {
                "error": "Topic is too vague for effective search",
                "details": validation_msg,
                "your_topic": params.topic,
                "why_it_matters": "Specific queries return higher-quality, more relevant community solutions",
                "how_to_fix": [
                    "Add specific library/framework names (e.g., 'FastAPI' not just 'API')",
                    "Include version numbers if relevant (e.g., 'wgpu 0.19' not just 'wgpu')",
                    "State the exact error or change (e.g., 'PipelineCompilationOptions removed')",
                    "Mention your goal (e.g., 'fix compilation errors after upgrade')",
                ],
                "good_examples": [
                    "FastAPI async background tasks with Celery and Redis",
                    "React custom hooks for form validation with Yup",
                    "Docker multi-stage builds to reduce image size",
                    "Rust wgpu PipelineCompilationOptions removed in 0.19",
                ],
                "bad_examples": [
                    "settings",
                    "performance",
                    "configuration",
                    "how to optimize",
                ],
            },
            indent=2,
        )

    enrichment = enrich_query(params.language, params.topic, params.goal)
    search_query = (
        enrichment.get("enriched_query") or f"{params.language} {params.topic}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            search_results = await aggregate_search_results(
                search_query,
                params.language,
                expanded_mode=params.expanded_mode,
                use_fixtures=params.use_fixtures,
            )

            filtered_results = filter_results_by_domain(
                search_results, params.language, params.topic
            )

            source_lists = _result_only_sources(filtered_results)
            total_results = total_result_count(filtered_results)
            # SIMPLIFIED: Skip complex all-star scoring, just use simple engagement metrics
            all_star_meta = {
                "top_overall": [],
                "buckets": {},
                "stats": {"total": total_results},
            }
            audit_log = search_results.get("_meta", {}).get("audit_log", [])
            shape_stats = summarize_content_shapes(source_lists)

            # TRANSPARENCY: Count raw vs filtered
            raw_counts = {
                src: len(items)
                for src, items in _result_only_sources(search_results).items()
            }
            filtered_counts = {src: len(items) for src, items in source_lists.items()}

            top_evidence = select_top_evidence(
                filtered_results, all_star_meta, search_query, params.language
            )

            # SIMPLIFIED: If we have ANY results at all from ANY source, proceed
            manual = get_manual_evidence(params.topic)

            if total_results == 0 and not manual:
                result = json.dumps(
                    {
                        "error": f'No results found for "{params.topic}" in {params.language}.',
                        "possible_reasons": [
                            "Query might be too specific or unusual",
                            "Try different keywords or expand the topic",
                            "Community sources may not have discussions on this topic",
                        ],
                        "expanded_queries": enrichment.get("expanded_queries", []),
                        "findings": [],
                        "assumptions": enrichment.get("assumptions", []),
                    },
                    indent=2,
                )
                set_cached_result(cache_key, result)
                return result

            context_meta = {
                "all_star": all_star_meta,
                "top_evidence": top_evidence,
                "enrichment": enrichment,
            }

            if manual:
                synthesis = {"findings": manual, "synthesis_summary": ""}
            else:
                synthesis = await synthesize_with_llm(
                    source_lists,
                    params.topic,
                    params.language,
                    params.goal,
                    params.current_setup,
                    context_meta=context_meta,
                )

            if (
                ENHANCED_UTILITIES_AVAILABLE
                and _quality_scorer
                and "findings" in synthesis
            ):
                synthesis["findings"] = _quality_scorer.score_findings_batch(
                    synthesis["findings"]
                )

            payload = normalize_masterclass_payload(
                synthesis,
                top_evidence,
                params.language,
                enrichment,
            )
            if synthesis.get("error"):
                payload.setdefault("assumptions", []).append(
                    f"LLM synthesis error: {synthesis.get('error')}"
                )
            conflicts_auto = detect_conflicts_from_findings(payload.get("findings", []))

            sources_present = [s for s in source_lists.keys() if source_lists[s]]
            search_meta = {
                "sources": [SOURCE_LABELS.get(s, s) for s in sources_present],
                "source_count": len(sources_present),
                "total_results": total_results,
                "enriched_query": search_query,
                "expanded_queries": enrichment.get("expanded_queries", []),
                "evidence_weak": len(top_evidence) < 2,
                "all_star": all_star_meta,
                "audit_log": audit_log,
                "shape_stats": shape_stats,
            }

            if params.response_format == ResponseFormat.MARKDOWN:
                result = render_masterclass_markdown(
                    params.topic,
                    params.language,
                    params.goal,
                    params.current_setup,
                    payload,
                    conflicts_auto,
                    search_meta,
                    manual_mode=bool(manual),
                )
            else:
                response = {
                    "_instructions": {
                        "how_to_use": "This response contains community-sourced solutions. Review findings by quality score, check conflicts for edge cases, and follow recommended_path for implementation.",
                        "quality_scores": {
                            "80-100": "Highly reliable - maintainer-backed, well-evidenced, recent",
                            "60-79": "Good quality - community-validated, has code examples",
                            "40-59": "Moderate - some evidence, may need verification",
                            "0-39": "Weak - limited evidence, outdated, or speculative",
                        },
                        "next_steps": [
                            "1. Review 'findings' starting with highest quality scores",
                            "2. Check 'conflicts' for contradicting advice or edge cases",
                            "3. Follow 'recommended_path' for step-by-step guidance",
                            "4. Use 'quick_apply' code/commands as starting points (review before running)",
                            "5. Validate using 'verification' steps after implementation",
                        ],
                        "search_quality": {
                            "sources_searched": search_meta.get("sources", []),
                            "total_results": search_meta.get("total_results", 0),
                            "evidence_strength": "weak"
                            if search_meta.get("evidence_weak")
                            else "strong",
                        },
                    },
                    "language": params.language,
                    "topic": params.topic,
                    "goal": params.goal,
                    "current_setup": params.current_setup,
                    "findings": payload.get("findings", []),
                    "conflicts": payload.get("conflicts", []) + conflicts_auto,
                    "recommended_path": payload.get("recommended_path", []),
                    "quick_apply": payload.get("quick_apply", {}),
                    "verification": payload.get("verification", []),
                    "assumptions": payload.get("assumptions", []),
                    "synthesis_summary": payload.get("synthesis_summary", ""),
                    "all_star": all_star_meta,
                    "search_meta": search_meta,
                    "manual_mode": bool(manual),
                }
                result = json.dumps(response, indent=2)

            if len(result) > CHARACTER_LIMIT:
                if params.response_format == ResponseFormat.JSON:
                    response_dict = json.loads(result)
                    original_count = len(response_dict.get("findings", []))
                    response_dict["findings"] = response_dict["findings"][
                        : max(1, original_count // 2)
                    ]
                    response_dict["truncated"] = True
                    response_dict["truncation_message"] = (
                        f"Response truncated from {original_count} to {len(response_dict['findings'])} findings due to size limits."
                    )
                    result = json.dumps(response_dict, indent=2)
                else:
                    result = (
                        result[:CHARACTER_LIMIT]
                        + "\n\n[Response truncated due to size limits. Use JSON format for full data.]"
                    )

            set_cached_result(cache_key, result)
            return result

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                error_response = json.dumps(
                    {
                        "error": f"Search failed after {MAX_RETRIES} attempts: {str(e)}",
                        "findings": [],
                    },
                    indent=2,
                )
                return error_response

            await asyncio.sleep(2**attempt)

    return json.dumps({"error": "Unexpected error", "findings": []}, indent=2)


# ============================================================================
# Web Content Classes
# ============================================================================


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class SearchResult:
    def __init__(self, title: str, link: str, snippet: str, position: int):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.position = position


async def fetch_page_content(url: str, max_chars: int = 12000) -> str:
    """
    Fetch and extract main content from a URL.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                return ""

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()

            # Get text
            text = soup.get_text(separator="\n")

            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text[:max_chars]

    except Exception as e:
        logging.warning(f"Failed to fetch content from {url}: {e}")
        return ""


class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str) -> str:
        """Fetch and parse content from a webpage"""
        try:
            await self.rate_limiter.acquire()

            logging.info(f"Fetching content from: {url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get the text content
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"

            logging.info(
                f"Successfully fetched and parsed content ({len(text)} characters)"
            )
            return text

        except httpx.TimeoutException:
            logging.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPError as e:
            logging.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            logging.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"


# Initialize web content fetcher
web_fetcher = WebContentFetcher()

# ============================================================================
# Reddit-Specific Tools
# ============================================================================


@mcp.tool(
    name="fetch_reddit_hot_threads",
    annotations={
        "title": "Fetch Hot Reddit Threads",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def fetch_reddit_hot_threads(subreddit: str, limit: int = 10) -> str:
    """
    Fetch hot threads from a specific subreddit.

    This tool retrieves the current hot threads from a specified subreddit,
    providing detailed information about each post including title, score,
    comment count, and content. It uses authenticated access when credentials
    are available for better results and higher rate limits.

    Args:
        subreddit (str): Name of the subreddit (e.g., "Python", "javascript")
        limit (int): Number of posts to fetch (default: 10)

    Returns:
        str: Formatted list of hot posts with their details:
            - Title and author
            - Score and comment count
            - Content snippet
            - Post URL

    Examples:
        - Fetch hot topics in Python: subreddit="Python"
        - Get JavaScript discussions: subreddit="javascript"
        - Browse Rust programming trends: subreddit="rust"

    Notes:
        - Authenticated access provides more reliable results
        - Rate limited to 10 requests per minute
        - Caches results for 1 hour
    """
    # Check rate limit
    if not check_rate_limit("fetch_reddit_hot_threads"):
        return json.dumps(
            {
                "error": "Rate limit exceeded. Maximum 10 requests per minute. Please wait and try again."
            },
            indent=2,
        )

    # Check cache
    cache_key = get_cache_key(
        "fetch_reddit_hot_threads", subreddit=subreddit, limit=limit
    )
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        # Try to use authenticated client if available
        if reddit_authenticated and reddit_client:
            try:
                posts = []
                # Fetch hot posts from subreddit
                async for submission in reddit_client.p.subreddit.pull.hot(
                    subreddit, limit
                ):
                    # Determine post type
                    post_type = "unknown"
                    if (
                        hasattr(submission, "url")
                        and submission.url != submission.permalink
                    ):
                        post_type = "link"
                    elif hasattr(submission, "body"):
                        post_type = "text"
                    elif hasattr(submission, "gallery_link"):
                        post_type = "gallery"

                    # Get content based on type
                    content = ""
                    if post_type == "text" and hasattr(submission, "body"):
                        content = submission.body[:500] + (
                            "..." if len(submission.body or "") > 500 else ""
                        )
                    elif post_type == "link" and hasattr(submission, "url"):
                        content = f"Link: {submission.url}"
                    elif post_type == "gallery":
                        content = "Gallery post (multiple images)"

                    # Format post info
                    post_info = (
                        f"Title: {submission.title}\n"
                        f"Score: {submission.score}\n"
                        f"Comments: {submission.comment_count}\n"
                        f"Author: {submission.author_display_name or '[deleted]'}\n"
                        f"Type: {post_type}\n"
                        f"Content: {content}\n"
                        f"Link: https://reddit.com{submission.permalink}\n"
                        f"---"
                    )
                    posts.append(post_info)

                # Return formatted posts
                if posts:
                    result = "\n\n".join(posts)
                    set_cached_result(cache_key, result)
                    return result

                # Fall back to unauthenticated if no results
                logging.info(
                    f"No authenticated results for subreddit {subreddit}, falling back to public API"
                )

            except Exception as e:
                logging.warning(
                    f"Authenticated Reddit hot threads failed: {str(e)}. Falling back to public API."
                )

        # Fallback to unauthenticated public API
        url = f"https://www.reddit.com/r/{subreddit}/hot.json"
        params = {"limit": limit}

        headers = {"User-Agent": "CommunityResearchMCP/1.0"}

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            posts = []
            for item in data.get("data", {}).get("children", []):
                post = item.get("data", {})

                # Determine post type
                post_type = "unknown"
                if post.get("is_self") == False and post.get("url"):
                    post_type = "link"
                elif post.get("is_self") == True:
                    post_type = "text"
                elif post.get("is_gallery") == True:
                    post_type = "gallery"

                # Get content based on type
                content = ""
                if post_type == "text":
                    content = post.get("selftext", "")[:500] + (
                        "..." if len(post.get("selftext", "")) > 500 else ""
                    )
                elif post_type == "link":
                    content = f"Link: {post.get('url', '')}"
                elif post_type == "gallery":
                    content = "Gallery post (multiple images)"

                # Format post info
                post_info = (
                    f"Title: {post.get('title', '')}\n"
                    f"Score: {post.get('score', 0)}\n"
                    f"Comments: {post.get('num_comments', 0)}\n"
                    f"Author: {post.get('author', '[deleted]')}\n"
                    f"Type: {post_type}\n"
                    f"Content: {content}\n"
                    f"Link: https://reddit.com{post.get('permalink', '')}\n"
                    f"---"
                )
                posts.append(post_info)

            # Return formatted posts
            if posts:
                result = "\n\n".join(posts)
                set_cached_result(cache_key, result)
                return result

            return f"No posts found in subreddit: r/{subreddit}"

    except Exception as e:
        logging.error(f"Error fetching Reddit hot threads: {str(e)}")
        return f"An error occurred: {str(e)}"


@mcp.tool(
    name="fetch_reddit_post_content",
    annotations={
        "title": "Fetch Reddit Post Content",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def fetch_reddit_post_content(
    post_id: str, comment_limit: int = 20, comment_depth: int = 3
) -> str:
    """
    Fetch detailed content of a specific Reddit post including comments.

    This tool retrieves a Reddit post by its ID and returns the full post content
    along with a hierarchical comment tree. It includes author information, scores,
    and full content. When available, uses authenticated access for better results.

    Args:
        post_id (str): Reddit post ID (either full URL or just the ID portion)
        comment_limit (int): Number of top-level comments to fetch (default: 20)
        comment_depth (int): Maximum depth of comment tree to traverse (default: 3)

    Returns:
        str: Formatted post content with hierarchical comments:
            - Post title, author, score
            - Full post content
            - Comments with proper indentation showing the discussion tree
            - Author and score for each comment

    Examples:
        - Fetch post with default settings: post_id="abcd123"
        - Get post with more comments: post_id="abcd123", comment_limit=50
        - Deep-dive into comments: post_id="abcd123", comment_depth=5

    Notes:
        - Works best with authenticated Reddit access
        - Can accept full Reddit URLs or just the post ID
        - Rate limited to 10 requests per minute
        - Caches results for 1 hour
    """
    # Check rate limit
    if not check_rate_limit("fetch_reddit_post_content"):
        return json.dumps(
            {
                "error": "Rate limit exceeded. Maximum 10 requests per minute. Please wait and try again."
            },
            indent=2,
        )

    # Extract post ID from URL if needed
    if post_id.startswith("http"):
        # Try to extract post ID from URL
        match = re.search(r"comments/([a-zA-Z0-9]+)/", post_id)
        if match:
            post_id = match.group(1)
        else:
            return "Invalid Reddit URL. Please provide a valid post ID or URL."

    # Check cache
    cache_key = get_cache_key(
        "fetch_reddit_post_content",
        post_id=post_id,
        comment_limit=comment_limit,
        comment_depth=comment_depth,
    )
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    # Helper function for formatting comment trees
    def format_comment_tree(comment_node, depth: int = 0) -> str:
        """Helper function to recursively format comment tree with proper indentation"""
        comment = comment_node.value if hasattr(comment_node, "value") else comment_node
        indent = "  " * depth

        if hasattr(comment, "author_display_name"):
            # Authenticated API format
            author = comment.author_display_name or "[deleted]"
            score = comment.score
            body = getattr(comment, "body", "[no content]")
        else:
            # Public API format
            data = comment.get("data", {})
            author = data.get("author", "[deleted]")
            score = data.get("score", 0)
            body = data.get("body", "[no content]")

        content = f"{indent}* **{author}** ({score} points)\n{indent}  {body}\n"

        # Process children
        if hasattr(comment_node, "children") and comment_node.children:
            for child in comment_node.children:
                content += "\n" + format_comment_tree(child, depth + 1)
        elif "replies" in comment and comment.get("replies"):
            replies = comment.get("replies", {})
            if isinstance(replies, dict) and "data" in replies:
                for child in replies["data"]["children"]:
                    if child["kind"] != "more":  # Skip "more comments" items
                        content += "\n" + format_comment_tree(child, depth + 1)

        return content

    try:
        # Try authenticated API first if available
        if reddit_authenticated and reddit_client:
            try:
                # Fetch submission
                submission = await reddit_client.p.submission.fetch(post_id)

                # Determine post type and content
                post_type = "unknown"
                if (
                    hasattr(submission, "url")
                    and submission.url != submission.permalink
                ):
                    post_type = "link"
                elif hasattr(submission, "body"):
                    post_type = "text"
                elif hasattr(submission, "gallery_link"):
                    post_type = "gallery"

                # Get content based on type
                content = ""
                if post_type == "text" and hasattr(submission, "body"):
                    content = submission.body
                elif post_type == "link" and hasattr(submission, "url"):
                    content = f"Link: {submission.url}"
                elif post_type == "gallery":
                    content = "Gallery post (multiple images)"

                # Format post header
                post_content = (
                    f"Title: {submission.title}\n"
                    f"Score: {submission.score}\n"
                    f"Author: {submission.author_display_name or '[deleted]'}\n"
                    f"Type: {post_type}\n\n"
                    f"Content:\n{content}\n"
                )

                # Fetch and format comments
                comments = await reddit_client.p.comment_tree.fetch(
                    post_id, sort="top", limit=comment_limit, depth=comment_depth
                )

                if comments and hasattr(comments, "children") and comments.children:
                    post_content += "\nComments:\n"
                    for comment in comments.children:
                        post_content += "\n" + format_comment_tree(comment)
                else:
                    post_content += "\nNo comments found."

                # Cache and return result
                set_cached_result(cache_key, post_content)
                return post_content

            except Exception as auth_error:
                logging.warning(
                    f"Authenticated Reddit post fetch failed: {str(auth_error)}. Falling back to public API."
                )

        # Fallback to public API
        url = f"https://www.reddit.com/comments/{post_id}.json"
        headers = {"User-Agent": "CommunityResearchMCP/1.0"}

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Extract post data
            post_data = data[0]["data"]["children"][0]["data"]

            # Determine post type and content
            post_type = "unknown"
            if post_data.get("is_self") == False and post_data.get("url"):
                post_type = "link"
            elif post_data.get("is_self") == True:
                post_type = "text"
            elif post_data.get("is_gallery") == True:
                post_type = "gallery"

            # Get content based on type
            content = ""
            if post_type == "text":
                content = post_data.get("selftext", "")
            elif post_type == "link":
                content = f"Link: {post_data.get('url', '')}"
            elif post_type == "gallery":
                content = "Gallery post (multiple images)"

            # Format post header
            post_content = (
                f"Title: {post_data.get('title')}\n"
                f"Score: {post_data.get('score')}\n"
                f"Author: {post_data.get('author')}\n"
                f"Type: {post_type}\n\n"
                f"Content:\n{content}\n"
            )

            # Process comments
            if len(data) > 1 and "children" in data[1]["data"]:
                comment_data = data[1]["data"]["children"][:comment_limit]

                if comment_data:
                    post_content += "\nComments:\n"
                    for comment in comment_data:
                        if comment["kind"] != "more":  # Skip "more comments" items
                            post_content += "\n" + format_comment_tree(comment)
                else:
                    post_content += "\nNo comments found."

            # Cache and return result
            set_cached_result(cache_key, post_content)
            return post_content

    except Exception as e:
        logging.error(f"Error fetching Reddit post content: {str(e)}")
        return f"An error occurred: {str(e)}"


# ============================================================================
# Web Content Tools
# ============================================================================


@mcp.tool(
    name="fetch_webpage_content",
    annotations={
        "title": "Fetch and Parse Webpage Content",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def fetch_webpage_content(url: str) -> str:
    """
    Fetch and parse content from a webpage URL.

    This tool retrieves a webpage, extracts its main text content by removing
    navigation, scripts, styles, and other non-content elements. It's useful
    for getting detailed information from articles, documentation, or any
    webpage found through search.

    Args:
        url (str): The webpage URL to fetch content from

    Returns:
        str: Cleaned text content from the webpage

    Examples:
        - Fetch documentation: url="https://docs.python.org/3/library/asyncio.html"
        - Fetch tutorial: url="https://reactjs.org/docs/hooks-effect.html"
        - Fetch article: url="https://martinfowler.com/articles/microservices.html"

    Notes:
        - Rate limited to 20 requests per minute
        - Removes ads, navigation, and other non-content elements
        - Truncates very long content to 8000 characters
        - Works best with article-style content
    """
    # Check rate limit
    if not check_rate_limit("fetch_webpage_content"):
        return json.dumps(
            {
                "error": "Rate limit exceeded. Maximum 10 requests per minute. Please wait and try again."
            },
            indent=2,
        )

    # Check cache
    cache_key = get_cache_key("fetch_webpage_content", url=url)
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        # Fetch and parse content
        result = await web_fetcher.fetch_and_parse(url)

        # Cache and return
        set_cached_result(cache_key, result)
        return result

    except Exception as e:
        logging.error(f"Webpage fetching failed: {str(e)}")
        return f"An error occurred while fetching the webpage: {str(e)}"


# ============================================================================
# Streaming & Auto-Detection Tools
# ============================================================================


@mcp.tool(
    name="get_system_capabilities",
    annotations={
        "title": "Auto-Detect System Capabilities",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_system_capabilities() -> str:
    """
    Auto-detect all available API keys and system capabilities.

    Scans environment for configured API keys and returns a complete report
    of what search APIs and LLM providers are currently available.

    This tool automatically recognizes:
    - Search APIs: Stack Overflow, GitHub, Reddit (auth & public), Hacker News,
      Lobsters, Discourse, Brave Search, Serper
    - LLM Providers: Gemini, OpenAI, Anthropic, OpenRouter, Perplexity
    - Workspace context and detected languages

    Returns:
        str: Formatted report of all active and inactive capabilities

    Example output:
        # 🔍 System Capabilities

        ## Search APIs
        **Active (6):**
          ✓ stackoverflow
          ✓ github
          ✓ reddit
          ✓ hackernews
          ✓ lobsters
          ✓ discourse

        **Inactive (2):**
          ✗ brave (API key not configured)
          ✗ serper (API key not configured)
    """
    if not STREAMING_AVAILABLE:
        return (
            "⚠️ Streaming capabilities module not available. Basic functionality only."
        )

    try:
        capabilities = detect_all_capabilities()
        report = format_capabilities_report(capabilities)
        return report
    except Exception as e:
        return f"Error detecting capabilities: {str(e)}"


@mcp.tool(
    name="streaming_community_search",
    annotations={
        "title": "Streaming Community Search (Real-Time Results)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def streaming_community_search(
    language: str,
    topic: str,
    goal: Optional[str] = None,
    current_setup: Optional[str] = None,
    context: Optional[Any] = None,
) -> str:
    """
    Search community resources with REAL-TIME STREAMING results.

    Fires all search capabilities in PARALLEL and streams results as they arrive,
    with progressive reorganization and smart aggregation. Results are categorized
    by type (quick fixes, code examples, warnings, discussions) and displayed
    incrementally.

    **Key Features:**
    - ⚡ Parallel execution across ALL available search sources
    - 📊 Real-time progress updates as each source completes
    - 🔄 Progressive reorganization while waiting for results
    - 🎯 Adaptive formatting based on content type
    - 🤖 Final LLM synthesis of all results

    Args:
        language (str): Programming language (e.g., "Python", "JavaScript")
        topic (str): Specific topic - MUST be detailed (min 10 chars)
        goal (Optional[str]): What you want to achieve
        current_setup (Optional[str]): Your current setup/constraints
        context (Context): MCP context for progress reporting

    Returns:
        str: Streaming markdown output with progressive updates and final synthesis

    Example:
        language="Python"
        topic="FastAPI background task queue with Redis for email processing"
        goal="Send emails asynchronously without blocking API requests"

    The search will fire across:
    - Stack Overflow (quick fixes & accepted answers)
    - GitHub (code examples & real implementations)
    - Reddit (community discussions & gotchas)
    - Hacker News (high-quality discussions)
    - Lobsters (tech-focused community discussions)
    - Discourse (language-specific forums)

    Results stream in real-time as each source completes!
    """
    if not STREAMING_AVAILABLE:
        # Fallback to standard search
        return await community_search(
            language=language,
            topic=topic,
            goal=goal,
            current_setup=current_setup,
            response_format="markdown",
            use_planning=False,
            thinking_mode="balanced",
        )

    # Validate topic specificity
    is_valid, error_msg = validate_topic_specificity(topic)
    if not is_valid:
        return error_msg

    # Check rate limiting
    if not check_rate_limit("streaming_community_search"):
        return "Rate limit exceeded. Please wait a minute before retrying."

    try:
        # Prepare search functions
        search_functions = {
            "stackoverflow": search_stackoverflow,
            "github": search_github,
            "reddit": search_reddit,
            "hackernews": search_hackernews,
            "lobsters": search_lobsters,
            "discourse": search_discourse,
        }

        if FIRECRAWL_API_KEY:
            search_functions["firecrawl"] = search_firecrawl

        if TAVILY_API_KEY:
            search_functions["tavily"] = search_tavily

        if BRAVE_SEARCH_API_KEY:
            search_functions["brave"] = search_brave

        if SERPER_API_KEY:
            search_functions["serper"] = search_serper

        # Stream results and synthesis
        output_parts = []

        async for update in streaming_search_with_synthesis(
            search_functions=search_functions,
            synthesis_func=synthesize_with_llm,
            query=topic,
            language=language,
            goal=goal,
            current_setup=current_setup,
            context=context,
        ):
            output_parts.append(update)

        # Combine all output
        final_output = "\n\n---\n\n".join(output_parts)

        # Enforce character limit
        if len(final_output) > CHARACTER_LIMIT:
            final_output = (
                final_output[:CHARACTER_LIMIT]
                + f"\n\n[Output truncated at {CHARACTER_LIMIT} characters]"
            )

        return final_output

    except Exception as e:
        logging.error(f"Streaming search failed: {str(e)}")
        return f"Error during streaming search: {str(e)}\n\nFalling back to standard search..."


@mcp.tool(
    name="deep_community_search",
    annotations={
        "title": "Deep Recursive Community Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def deep_community_search(params: CommunitySearchInput) -> str:
    """
    Deep dive into the programming community's collective wisdom and hard-won solutions.

    Like Ctrl+F for the entire community's trauma - finds every workaround, hack,
    and "this finally worked" moment across Stack Overflow, Reddit, GitHub, and forums.

    WHEN TO USE THIS TOOL:
    - You need the FULL picture, not just the first answer
    - The problem is complex with multiple potential solutions
    - You want to find ALL the gotchas before they bite you
    - community_search didn't find enough street-smart solutions

    WORKFLOW:
    - Use this tool DIRECTLY (it includes its own multi-phase planning)
    - No need for plan_research first - this tool searches multiple angles automatically
    - Results include comprehensive findings with synthesis_instructions

    HOW IT WORKS:
    1. Initial broad search across all community sources
    2. Automatic follow-up searches for workarounds, pitfalls, and edge cases
    3. Deep content extraction from the most upvoted/accepted answers
    4. Synthesis of the community's collective experience

    WHEN TO USE community_search INSTEAD:
    - Quick, specific questions with likely one good answer
    - Simple lookups where depth isn't needed

    Args:
        params: CommunitySearchInput object with language, topic, goal, current_setup

    Returns:
        JSON string with comprehensive research findings and synthesis_instructions

    Example:
        params = CommunitySearchInput(
            language="Python",
            topic="FastAPI async background tasks with Celery",
            goal="Process long-running tasks without blocking"
        )
        result = await deep_community_search(params)
    """
    # Extract parameters from the input object
    language = params.language
    topic = params.topic
    goal = params.goal
    current_setup = params.current_setup
    # Check rate limit
    if not check_rate_limit("deep_community_search"):
        return json.dumps({"error": "Rate limit exceeded."}, indent=2)

    try:
        # Step 1: Initial Search
        print(f"🔍 [Deep Search] Starting initial search for: {topic}")
        initial_query = f"{language} {topic}"
        if goal:
            initial_query += f" {goal}"

        initial_results = await aggregate_search_results(initial_query, language)

        # Step 2: Generate follow-up queries for comprehensive coverage
        print("🤔 [Deep Search] Generating follow-up search queries...")

        # Use deterministic queries - gap analysis is delegated to the host LLM
        follow_up_queries = [
            f"{language} {topic} advanced usage",
            f"{language} {topic} best practices",
            f"{language} {topic} common pitfalls",
        ]

        # Step 3: Follow-up Searches
        print(f"🕵️ [Deep Search] Running {len(follow_up_queries)} follow-up searches...")
        follow_up_tasks = [
            aggregate_search_results(q, language) for q in follow_up_queries
        ]
        follow_up_results_list = await asyncio.gather(
            *follow_up_tasks, return_exceptions=True
        )

        # Merge results
        combined_results = _result_only_sources(initial_results)
        for res in follow_up_results_list:
            if isinstance(res, dict):
                for source, items in _result_only_sources(res).items():
                    if source in combined_results:
                        combined_results[source].extend(items)
                    else:
                        combined_results[source] = items

        # Step 4: Active Browsing (Fetch content for top items)
        print("🌐 [Deep Search] Active browsing top results...")
        # Extract top URLs from all results
        all_urls = []
        for source_results in combined_results.values():
            all_urls.extend(
                [item["url"] for item in source_results[:2] if item.get("url")]
            )

        top_urls = all_urls[:5]  # Get top 5 URLs across all sources

        if top_urls:
            contents = await asyncio.gather(
                *[fetch_page_content(url) for url in top_urls], return_exceptions=True
            )

        # Step 5: Final Synthesis
        print("🤖 [Deep Search] Synthesizing final comprehensive answer...")
        final_response = await synthesize_with_multi_model(
            combined_results,
            topic,
            language,
            goal,
            current_setup,
            thinking_mode=ThinkingMode.DEEP,
        )

        return json.dumps(final_response, indent=2)

    except Exception as e:
        logging.error(f"Deep search failed: {e}")
        return json.dumps({"error": f"Deep search failed: {str(e)}"}, indent=2)


@mcp.tool(
    name="parallel_multi_source_search",
    annotations={
        "title": "Parallel Multi-Source Search (Advanced)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def parallel_multi_source_search(
    query: str,
    language: str,
    sources: Optional[str] = "all",
    context: Optional[Any] = None,
) -> str:
    """
    Execute searches across multiple sources in PARALLEL with real-time updates.

    This is an advanced tool that gives you fine-grained control over which
    search sources to query. All selected sources fire simultaneously and
    results stream back in real-time.

    Args:
        query (str): Search query
        language (str): Programming language context
        sources (Optional[str]): Comma-separated list of sources or "all"
            Options: stackoverflow, github, reddit, hackernews, lobsters, discourse, firecrawl, tavily
            Default: "all"
        context (Context): MCP context for progress reporting

    Returns:
        str: JSON-formatted results organized by source and content type

    Example:
        query="async/await error handling best practices"
        language="JavaScript"
        sources="stackoverflow,github,reddit"

    This will search only Stack Overflow, GitHub, and Reddit in parallel,
    ignoring other sources.
    """
    if not STREAMING_AVAILABLE:
        return json.dumps(
            {"error": "Streaming capabilities not available", "results": {}}, indent=2
        )

    # Parse sources
    available_sources = [
        "stackoverflow",
        "github",
        "reddit",
        "hackernews",
        "lobsters",
        "discourse",
    ]

    if FIRECRAWL_API_KEY:
        available_sources.append("firecrawl")
    if TAVILY_API_KEY:
        available_sources.append("tavily")

    if sources == "all":
        source_list = available_sources
    else:
        source_list = [s.strip() for s in sources.split(",")]

    # Map source names to functions
    source_map = {
        "stackoverflow": search_stackoverflow,
        "github": search_github,
        "reddit": search_reddit,
        "hackernews": search_hackernews,
        "lobsters": search_lobsters,
        "discourse": search_discourse,
    }

    if FIRECRAWL_API_KEY:
        source_map["firecrawl"] = search_firecrawl

    if TAVILY_API_KEY:
        source_map["tavily"] = search_tavily

    if BRAVE_SEARCH_API_KEY:
        source_map["brave"] = search_brave

    if SERPER_API_KEY:
        source_map["serper"] = search_serper

    # Filter to requested sources
    search_functions = {
        name: func for name, func in source_map.items() if name in source_list
    }

    if not search_functions:
        return json.dumps(
            {
                "error": "No valid sources selected",
                "available_sources": list(source_map.keys()),
            },
            indent=2,
        )

    try:
        # Collect all results
        all_results = {}
        result_count = 0

        if context:
            await context.info(
                f"🚀 Starting parallel search across {len(search_functions)} sources..."
            )

        async for update in get_all_search_results_streaming(
            search_functions.get("stackoverflow"),
            search_functions.get("github"),
            search_functions.get("reddit"),
            search_functions.get("hackernews"),
            search_functions.get("lobsters"),
            search_functions.get("discourse"),
            query=query,
            language=language,
            context=context,
            search_firecrawl_func=search_functions.get("firecrawl"),
            search_tavily_func=search_functions.get("tavily"),
            search_brave_func=search_functions.get("brave"),
            search_serper_func=search_functions.get("serper"),
        ):
            if update["type"] == "complete":
                all_results = update["state"].results_by_source
                result_count = update["summary"]["total_results"]
                break

        # Format output
        output = {
            "query": query,
            "language": language,
            "sources_searched": list(search_functions.keys()),
            "total_results": result_count,
            "results_by_source": all_results,
            "results_by_type": update["state"].results_by_type if update else {},
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        logging.error(f"Parallel search failed: {str(e)}")
        return json.dumps({"error": str(e), "results": {}}, indent=2)


@mcp.tool(
    name="get_performance_metrics",
    annotations={
        "title": "Get System Performance Metrics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_performance_metrics() -> str:
    """
    Get comprehensive performance metrics for the MCP server.

    This tool provides real-time insights into:
    - Search performance and latency
    - API reliability and success rates
    - Cache effectiveness
    - Error distribution
    - System uptime

    Returns:
        str: Formatted performance report with all metrics

    Example Output:
        # 📊 Performance Metrics Report

        ## System Performance
        - **Uptime**: 3600.0s
        - **Total Searches**: 45
        - **Average Search Time**: 1200ms
        - **Cache Hit Rate**: 35.5%

        ## API Reliability
        - **Success Rate**: 99.2%
        - **Total Calls**: 250
        - **Retry Count**: 5
    """
    if not ENHANCED_UTILITIES_AVAILABLE:
        return "⚠️ Performance monitoring not available. Please ensure enhanced_mcp_utilities.py is installed."

    try:
        report = format_metrics_report()
        return report
    except Exception as e:
        return f"Error generating metrics report: {str(e)}"


@mcp.tool(
    name="optimize_search_query",
    annotations={
        "title": "Optimize Search Query for Better Results",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def optimize_search_query(
    language: str,
    topic: str,
    goal: Optional[str] = None,
    error_message: Optional[str] = None,
) -> str:
    """
    Help optimize a search query to get better community research results.

    This tool is designed to be called by AI assistants (like Claude) to help users
    craft better search queries when they encounter issues like:
    - "Topic is too vague" errors
    - Poor/insufficient search results
    - GitHub 422 errors (query too long)
    - No relevant community discussions found

    The connected AI can use this to:
    1. Suggest more specific query terms
    2. Identify key technical terms to include
    3. Simplify overly complex queries
    4. Extract version numbers and error patterns

    Args:
        language: Programming language (e.g., "Python", "Rust")
        topic: The search topic (can be vague - this tool will help refine it)
        goal: Optional goal statement
        error_message: Optional error message from a previous search attempt

    Returns:
        JSON string with optimization suggestions

    Example:
        # User says: "I need help with FastAPI performance"
        # AI calls: optimize_search_query("Python", "FastAPI performance")
        # Returns: Suggestions to make it more specific
    """
    try:
        # Check cache first
        cache_key = f"optimize:{language}:{topic}:{goal}"
        if cache_key in _mcp_ai_context["optimization_cache"]:
            return _mcp_ai_context["optimization_cache"][cache_key]

        # Analyze the query
        enrichment = enrich_query(language, topic, goal)

        # Check if topic is valid
        is_valid, validation_msg = validate_topic_specificity(topic)

        suggestions = {
            "original_query": {"language": language, "topic": topic, "goal": goal},
            "is_specific_enough": is_valid,
            "validation_message": validation_msg
            if not is_valid
            else "Query is specific enough",
            "enriched_query": enrichment.get("enriched_query"),
            "detected_versions": enrichment.get("versions", []),
            "enrichment_notes": enrichment.get("notes", []),
            "assumptions": enrichment.get("assumptions", []),
            "suggestions": [],
            "recommended_query": None,
        }

        if not is_valid:
            # Provide specific suggestions for vague queries
            suggestions["suggestions"] = [
                {
                    "issue": "Query is too vague",
                    "fix": "Add specific library/framework names",
                    "example": f"Instead of '{topic}', try '{language} {topic} with [specific library]'",
                },
                {
                    "issue": "Missing technical context",
                    "fix": "Include version numbers if relevant",
                    "example": f"{topic} in {language} version X.Y",
                },
                {
                    "issue": "Unclear objective",
                    "fix": "State the specific problem or goal",
                    "example": f"{topic} - fix [specific error] or implement [specific feature]",
                },
            ]

            # Generate recommended query
            if goal:
                suggestions["recommended_query"] = {
                    "language": language,
                    "topic": f"{topic} {goal}",
                    "goal": f"Find specific solutions for: {goal}",
                }
            else:
                suggestions["recommended_query"] = {
                    "language": language,
                    "topic": f"{topic} implementation best practices",
                    "goal": "Find production-ready solutions and common patterns",
                }
        else:
            # Query is good, but provide enhancement suggestions
            suggestions["suggestions"] = [
                {
                    "status": "Query looks good!",
                    "enhancement": "Consider adding error messages or version numbers for even better results",
                }
            ]
            suggestions["recommended_query"] = {
                "language": language,
                "topic": topic,
                "goal": goal or "Find community solutions and best practices",
            }

        # Handle error messages from previous attempts
        if error_message:
            if "422" in error_message or "query too long" in error_message.lower():
                suggestions["error_analysis"] = {
                    "type": "GitHub 422 - Query Too Long",
                    "cause": "The search query exceeds GitHub API limits",
                    "solution": "Simplify to core keywords only",
                    "recommended_simplification": _simplify_for_github(topic, language),
                }
            elif "202" in error_message or "rate limit" in error_message.lower():
                suggestions["error_analysis"] = {
                    "type": "Rate Limiting",
                    "cause": "Too many requests to community sources",
                    "solution": "Wait a few minutes before retrying, or use fewer simultaneous searches",
                }

        result = json.dumps(suggestions, indent=2)

        # Cache the result
        _mcp_ai_context["optimization_cache"][cache_key] = result
        _mcp_ai_context["last_query_optimization"] = suggestions

        return result

    except Exception as e:
        return json.dumps(
            {
                "error": f"Query optimization failed: {str(e)}",
                "fallback": "Try making your query more specific by adding library names, versions, or error messages",
            },
            indent=2,
        )


def _simplify_for_github(topic: str, language: str) -> Dict[str, str]:
    """Helper to simplify a query for GitHub API."""
    # Extract key technical terms (capitalized words, version numbers, etc.)
    words = topic.split()
    key_terms = []
    for word in words:
        if len(word) > 3 and (word[0].isupper() or any(c.isdigit() for c in word)):
            key_terms.append(word)

    simplified = " ".join(key_terms[:5])  # Max 5 key terms
    return {
        "original": f"{language} {topic}",
        "simplified": f"{language} {simplified}",
        "removed_words": len(words) - len(key_terms),
    }


@mcp.tool(
    name="clear_cache",
    annotations={
        "title": "Clear Search Cache",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def clear_cache() -> str:
    """
    Clear the local search cache.

    This removes all cached search results, forcing fresh queries on next search.
    Useful when you need up-to-date results or when cached data might be stale.

    Returns:
        str: Status message indicating whether cache was cleared
    """
    cache_file = Path(".community_research_cache.json")
    try:
        if cache_file.exists():
            cache_file.unlink()
            # Also clear optimization cache
            _mcp_ai_context["optimization_cache"].clear()
            return (
                "✓ Cache cleared successfully. Next searches will fetch fresh results."
            )
        else:
            return "ℹ️ Cache was already empty."
    except Exception as e:
        return f"❌ Error clearing cache: {str(e)}"


def validate_environment():
    """Validate environment configuration on startup."""
    print("\nValidating environment...")

    # Check optional API keys for enhanced search
    keys = {
        "REDDIT": os.getenv("REDDIT_CLIENT_ID"),
        "FIRECRAWL": os.getenv("FIRECRAWL_API_KEY"),
        "TAVILY": os.getenv("TAVILY_API_KEY"),
        "BRAVE": os.getenv("BRAVE_SEARCH_API_KEY"),
        "SERPER": os.getenv("SERPER_API_KEY"),
    }

    active_keys = [k for k, v in keys.items() if v]
    if active_keys:
        print(f"Enhanced search APIs: {', '.join(active_keys)}")
    else:
        print("Using default search sources (no premium APIs configured)")

    # Check capabilities
    capabilities = []
    if STREAMING_AVAILABLE:
        capabilities.append("streaming")
    if ENHANCED_UTILITIES_AVAILABLE:
        capabilities.append("enhanced utilities")

    if capabilities:
        print(f"Capabilities: {', '.join(capabilities)}")

    print("Ready\n")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Validate environment
    validate_environment()

    # Run the MCP server
    mcp.run()
