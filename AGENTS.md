# AGENTS.md

- Use **Jujutsu VCS** for version control and commits.
- Include agent and model details in the commit-message trailers.
  Example:

  ```text
  Generated-by: Codex (GPT-5.2-Codex, reasoning: high)
  ```

- Sandbox/approval note: in restricted environments, these typically require permission escalation:
  - `jj commit` (needs to write to `.git/objects` to create commit objects)
  - Integration tests that start local servers / bind ports (e.g. Vite) or launch browsers (e.g. Playwright)
- Windows note: PowerShell CP932 can replace IDS/ext-CJK chars with `?`; set UTF-8 before printing.
