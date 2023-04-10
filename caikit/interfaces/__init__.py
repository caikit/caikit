"""This library defines the taxonomy of Data Model objects and Tasks for the
entire CAIKit project. Data objects and tasks are grouped domain, making for a
three-level hierarchy for models:

problem domain -> task -> implementation

This library intentionally does NOT define any implementations, as those are
left to the domain implementation libraries.
"""

# First Party
import import_tracker

# Import each domain with lazy import errors
with import_tracker.lazy_import_errors():
    # Local
    from . import common, runtime
