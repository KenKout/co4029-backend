"""Capability-specific eval runners.

Each runner takes a scenario's inputs, drives the corresponding production
AI pipeline (under `abridgeai.features.<capability>.ai`) for the requested
backend(s), and returns raw outputs + cost breakdown for the judge stage.

T8.1 ships scaffold only. T8.2 will register concrete runners
(quiz_generation, interview_generation, knowledge_graph_extraction, ...).
"""
