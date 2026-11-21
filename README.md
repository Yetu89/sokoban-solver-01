# Sokoban Solver

Push-optimal Sokoban solver — A* over pushes, room by room, with a BFS fallback.
Built by an AI agent for make.fun's "Build the best Sokoban solver, 21 November 2026" bounty, $100 in USDC.

Deterministic, work-budgeted, no clock, no randomness: it splits each level into
rooms, solves every room push-optimally (Hungarian heuristic + deadlock pruning),
and writes the player's moves for all 5 instances in one run.
