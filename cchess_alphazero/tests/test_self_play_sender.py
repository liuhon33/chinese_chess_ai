import unittest
from threading import Lock
from unittest.mock import Mock, patch

from cchess_alphazero.agent.player import CChessPlayer


class SelfPlaySenderTest(unittest.TestCase):
    def make_sender(self):
        player = CChessPlayer.__new__(CChessPlayer)
        player.job_done = False
        player.run_lock = Lock()
        player.q_lock = Lock()
        player.buffer_history = []
        player.buffer_planes = []
        player.pipe = Mock()
        return player

    def test_idle_sender_allows_producers_to_enqueue_during_sleep(self):
        player = self.make_sender()

        def idle_sleep(_seconds):
            # A producer must be able to enqueue while the sender is idle.
            acquired = player.q_lock.acquire(blocking=False)
            try:
                self.assertTrue(acquired, 'Idle sender holds the prediction queue lock')
                self.assertFalse(player.run_lock.locked())
                player.buffer_history.append(['state'])
                player.buffer_planes.append('planes')
            finally:
                if acquired:
                    player.q_lock.release()

        def send(batch):
            self.assertEqual(batch, ['planes'])
            # Keep one batch in flight until the receiver releases run_lock.
            self.assertTrue(player.run_lock.locked())
            player.job_done = True

        player.pipe.send.side_effect = send
        with patch('cchess_alphazero.agent.player.sleep', side_effect=idle_sleep):
            player.sender()
        player.pipe.send.assert_called_once_with(['planes'])

    def test_sender_preserves_batch_limit_and_order(self):
        player = self.make_sender()
        player.buffer_history = list(range(300))
        player.buffer_planes = list(range(300))
        player.pipe.send.side_effect = lambda _batch: setattr(player, 'job_done', True)
        with patch('cchess_alphazero.agent.player.sleep') as idle_sleep:
            player.sender()
        player.pipe.send.assert_called_once_with(list(range(256)))
        self.assertEqual(player.buffer_history, list(range(300)))
        self.assertEqual(player.buffer_planes, list(range(300)))
        idle_sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
