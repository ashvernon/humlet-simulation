import os
import sys
from pathlib import Path

# Ensure pygame can run headlessly during tests
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pytest

# Ensure the repository root is importable when tests are invoked from arbitrary
# working directories (e.g., running a single file from within ``tests/``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pygame
except ModuleNotFoundError:  # pragma: no cover - handled via skip below
    pygame = None


@pytest.fixture()
def headless_pygame():
    """Provide a headless pygame instance for tests that need it."""
    if pygame is None:
        pytest.skip("pygame is required for this test")

    if not pygame.get_init():
        pygame.init()
    pygame.display.init()
    yield pygame
    pygame.quit()
