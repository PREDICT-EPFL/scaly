"""The backend: render the program dialect to C, then write it out (AOT) or compile it (JIT)."""

from .aot import CModule, render_c_api_header, render_c_module, render_c_source, workspace_size, write_module

__all__ = ["CModule", "render_c_api_header", "render_c_module", "render_c_source", "workspace_size", "write_module"]
