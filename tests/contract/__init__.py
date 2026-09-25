"""Backend contract tests

These tests exercise a common contract shared by every cache backend implementation, without
depending on external database services. They operate on logical records (cache keys -> responses
plus redirect aliases) rather than backend-specific on-disk or in-memory layouts.
"""
