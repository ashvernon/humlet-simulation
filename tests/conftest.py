import os

# Ensure pygame can run headlessly during tests
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pytest

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
