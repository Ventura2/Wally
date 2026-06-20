from __future__ import annotations

import logging
import signal
from typing import Any, Dict, List

from wally.collector.buffer import TrajectoryBuffer
from wally.collector.config import CollectorConfig
from wally.collector.env import MineStudioEnv
from wally.collector.raw_shard_writer import RawShardWriter
from wally.collector.recorder import TransitionRecorder

logger = logging.getLogger(__name__)


class TrajectoryCollector:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self.env = MineStudioEnv(config)
        self.recorder = TransitionRecorder(config)
        self.buffer = TrajectoryBuffer(
            max_size=config.buffer_size,
            flush_callback=self._on_flush,
        )
        self._writer = RawShardWriter(
            output_dir=config.output_dir,
            shard_size=config.buffer_size,
            jpeg_quality=config.jpeg_quality,
        )
        self._writer.__enter__()
        self._collected: List[Dict[str, Any]] = []
        self._shutdown_requested = False
        self._original_sigint: Any = None

    def _on_flush(self, transitions: List[Dict[str, Any]]) -> None:
        logger.info("Flushing %d transitions", len(transitions))
        self._collected.extend(transitions)
        for t in transitions:
            self._writer.add(t)

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        logger.info("Shutdown requested, finishing current episode...")
        self._shutdown_requested = True
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

    def _sample_action(self) -> Dict[str, Any]:
        return self.env.action_space.sample()

    def run(self, num_episodes: int = 1) -> List[Dict[str, Any]]:
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        self._shutdown_requested = False
        episodes_completed = 0

        try:
            for ep in range(num_episodes):
                if self._shutdown_requested:
                    logger.info(
                    "Shutdown requested, stopping after %d episodes",
                    episodes_completed,
                )
                    break

                logger.info("Starting episode %d/%d", ep + 1, num_episodes)
                self.env.reset()
                self.recorder.start_episode()
                done = False

                step_in_ep = 0
                while not done and not self._shutdown_requested:
                    action = self._sample_action()
                    transition = self.recorder.record_step(self.env, action)
                    self.buffer.add(transition)
                    done = transition["done"]
                    step_in_ep += 1
                    if self.config.max_steps > 0 and step_in_ep >= self.config.max_steps:
                        logger.info(
                            "Reached max_steps (%d), ending episode",
                            self.config.max_steps,
                        )
                        break

                episodes_completed += 1
                logger.info(
                    "Episode %d completed (%d transitions so far)",
                    ep + 1,
                    len(self._collected),
                )
        finally:
            signal.signal(signal.SIGINT, self._original_sigint)

        logger.info(
            "Collection finished: %d episodes, %d transitions",
            episodes_completed,
            len(self._collected),
        )
        return self._collected

    def close(self) -> None:
        self.buffer.shutdown()
        self._writer.close()
        try:
            self.env.close()
        except Exception:
            logger.warning("Error closing environment", exc_info=True)
