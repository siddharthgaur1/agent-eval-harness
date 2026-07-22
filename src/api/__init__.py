"""FastAPI service.

Deliberately does not re-export `app`: that name would shadow the `src.api.app`
submodule, and `import src.api.app` would hand you the FastAPI instance instead
of the module. Import from `src.api.app` directly.
"""
