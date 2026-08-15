# User Instruction Memory

This file records user instructions, preferences, and teachings for reference in future interactions.

## Format

### User Instruction Entry
User instruction entries should follow this format:

[User Instruction Summary]
- Date: [YYYY-MM-DD]
- Context: [Mentioned scenario or time]
- Instructions:
  - [Content of user teaching or instruction, described line by line]

### Project Knowledge Entry
Entries discovered by the Agent during task execution should follow this format:

[Project Knowledge Summary]
- Date: [YYYY-MM-DD]
- Context: Discovered by Agent while performing [specific task description]
- Category: [Operations & Deployment|Build Methods|Testing Methods|Troubleshooting & Debugging|Workflow & Collaboration|Environment Configuration]
- Instructions:
  - [Specific knowledge points, described line by line]

## Deduplication Strategy
- Before adding a new entry, check for similar or identical instructions.
- If a duplicate is found, skip the new entry or merge it with the existing one.
- When merging, update the context or date information.
- This helps avoid redundant entries and keeps the memory file tidy.

## Entries

[Project Knowledge Summary]
- Date: 2026-08-15
- Context: Discovered by Agent while implementing and verifying the dashboard and benchmark suite
- Category: Testing Methods
- Instructions:
  - Use `python3 -m pytest` to run tests in this environment; the plain `pytest` command is not available by default.
  - Install missing Python tooling with `python3 -m pip install --break-system-packages <package>` when the workspace lacks runtime/test dependencies.
  - The project test suite passes with the focused command set `python3 -m pytest tests/test_evolution_tracker.py tests/test_dashboard_api.py tests/test_benchmark_suite.py -q` and `python3 -m pytest tests/test_core.py tests/test_langgraph.py tests/test_desktop_bridge.py tests/test_turning_consciousness.py tests/test_alaya_persistent.py -q`.
