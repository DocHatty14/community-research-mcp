![download](https://github.com/user-attachments/assets/45451d4b-170f-459c-a74e-998668af4555)

<div align="center">

![Community Research MCP](https://img.shields.io/badge/Community_Wisdom-Activated-7289DA?style=for-the-badge&logo=discord&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)
![AI Powered](https://img.shields.io/badge/AI-Gemini_|_Claude_|_OpenAI-orange?style=for-the-badge)

**Stop Hallucinating. Start Validating.**
*The only AI research tool that prioritizes "Street Smarts" over textbook theory.*

[Getting Started](docs/getting-started.md) • [Features](docs/features.md) • [API Reference](docs/api-reference.md)

</div>

---

## 🧠 The "Street Smarts" Engine

Most AI tools give you the "textbook" answer. The one that works in theory but fails in production. **Community Research MCP** is different. It bypasses generic training data to tap directly into the **living wisdom of the developer community**.

We scrape, aggregate, and synthesize the **battle-tested workarounds**, **undocumented hacks**, and **hard-earned lessons** from:

| Source | What You Get |
|--------|--------------|
| **Stack Overflow** | Accepted solutions *and* the "real" answer in the comments. |
| **GitHub Issues** | Bug fixes, patch notes, and "closed as won't fix" workarounds. |
| **Reddit** | Honest debates, tool comparisons, and "don't use X, use Y" advice. |
| **Hacker News** | High-level architectural critiques and industry trends. |
| **DuckDuckGo** | The latest docs and blog posts (fetched & parsed). |

## ✨ Why It's "A+" Software

### 1. 🔄 Deep Research Loop
It doesn't just search once. It thinks like a senior engineer:
1.  **Scouts**: Runs a broad parallel search.
2.  **Identifies Gaps**: "I found the library, but not how to handle auth."
3.  **Iterates**: Launches targeted follow-up searches to fill those gaps.
4.  **Synthesizes**: Delivers a cohesive, verified implementation plan.

### 2. 🌐 Active Browsing (No More Snippets)
We don't trust 2-line search snippets. This tool **visits the actual webpages**, scrapes the full documentation, and reads the entire GitHub issue thread to ensure it understands the *context* before giving you an answer.

### 3. 🛡️ Multi-Model Validation
**Trust, but verify.**
When you run a `validated_research` task, we use a second, independent AI model to critique the findings. It checks for security flaws, deprecated methods, and logical inconsistencies.

### 4. ⚡ Parallel Streaming
Don't wait. Results stream in **real-time** as they are found.
- **Stack Overflow**: 0.8s
- **GitHub**: 1.2s
- **Full Synthesis**: ~4s

## 🚀 Quick Start

### Installation
```bash
# 1. Clone & Setup
git clone https://github.com/your-repo/community-research-mcp.git
cd community-research-mcp
initialize.bat

# 2. Configure Keys
# Edit .env and add your GEMINI_API_KEY (or OpenAI/Anthropic)
```

### Usage
**"How do I fix the `wgpu` pipeline error in Rust?"**

```python
# The "Street Smart" Search
result = await deep_community_search(
    language="Rust",
    topic="wgpu PipelineCompilationOptions removed in latest version",
    goal="Fix compilation errors after upgrade"
)
```

**The Result?**
Instead of a generic "check the docs," you get:
> *"The `compilation_options` field was removed in wgpu 0.19. Community discussions on GitHub Issue #452 suggest using `wgpu::ShaderModuleDescriptor` directly. Here is the working migration code used by the Bevy engine team..."*

## 📦 Documentation

- [**📚 Getting Started**](docs/getting-started.md) - Setup guide.
- [**⚡ Features Deep Dive**](docs/features.md) - Learn about Smart Clustering & Active Browsing.
- [**🏗️ Architecture**](docs/architecture.md) - How the parallel engine works.
- [**🔌 API Reference**](docs/api-reference.md) - Full tool documentation.

---
<div align="center">
Built with ❤️ for developers who ship code.
</div>
