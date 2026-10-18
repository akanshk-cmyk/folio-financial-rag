"""Context: module invocation works without installing the project as a package."""

from .cli import main

raise SystemExit(main())
