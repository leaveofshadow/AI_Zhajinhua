"""游戏服务器单元测试。"""

import pytest
from server.schemas import RoomCreate, SeatConfig
from server.services.room_manager import RoomManager
from server.services.replay_store import ReplayStore
from server.services.game_runner import GameRunner


class TestRoomManager:
    """房间管理测试。"""

    def test_create_room(self):
        mgr = RoomManager()
        room = mgr.create_room(RoomCreate(num_players=4))
        assert room.id
        assert room.num_players == 4
        assert room.phase == "waiting"

    def test_get_room(self):
        mgr = RoomManager()
        room = mgr.create_room(RoomCreate())
        found = mgr.get_room(room.id)
        assert found is room

    def test_get_nonexistent(self):
        mgr = RoomManager()
        assert mgr.get_room("xxx") is None

    def test_close_room(self):
        mgr = RoomManager()
        room = mgr.create_room(RoomCreate())
        assert mgr.close_room(room.id) is True
        assert mgr.get_room(room.id) is None

    def test_list_rooms(self):
        mgr = RoomManager()
        mgr.create_room(RoomCreate())
        mgr.create_room(RoomCreate())
        assert len(mgr.list_rooms()) == 2

    def test_update_seats(self):
        mgr = RoomManager()
        room = mgr.create_room(RoomCreate(num_players=3))
        new_seats = [
            SeatConfig(player_type="human", display_name="P1"),
            SeatConfig(player_type="ai", display_name="AI-1"),
            SeatConfig(player_type="ai", display_name="AI-2"),
        ]
        updated, error = mgr.update_seats(room.id, new_seats)
        assert updated is not None
        assert error is None
        assert updated.seats[0].player_type == "human"

    def test_update_seats_wrong_count(self):
        mgr = RoomManager()
        room = mgr.create_room(RoomCreate(num_players=3))
        new_seats = [SeatConfig(), SeatConfig()]  # 只有2个
        updated, error = mgr.update_seats(room.id, new_seats)
        assert updated is None
        assert error is not None


class TestReplayStore:
    """回放存储测试。"""

    def test_save_and_get(self, tmp_path):
        store = ReplayStore(save_dir=str(tmp_path / "replays"))
        data = {"players": [], "actions": [], "result": {}}
        rid = store.save(data)
        assert rid

        loaded = store.get(rid)
        assert loaded is not None
        assert loaded["id"] == rid

    def test_get_nonexistent(self, tmp_path):
        store = ReplayStore(save_dir=str(tmp_path / "replays2"))
        assert store.get("xxx") is None

    def test_list_replays(self, tmp_path):
        store = ReplayStore(save_dir=str(tmp_path / "replays3"))
        store.save({"players": [{"a": 1}], "actions": []})
        store.save({"players": [{"b": 2}], "actions": []})
        replays = store.list_replays()
        assert len(replays) == 2


class TestGameRunner:
    """对局调度器测试。"""

    def test_start_game(self):
        store = ReplayStore(save_dir="/tmp/test_replays")
        runner = GameRunner(store)
        seats = [SeatConfig() for _ in range(4)]
        game = runner.start_game("test", 4, 1000, 10, seats)
        assert game is not None
        assert not game.is_finished()

    @pytest.mark.asyncio
    async def test_ai_turn(self):
        store = ReplayStore(save_dir="/tmp/test_replays2")
        runner = GameRunner(store)
        seats = [SeatConfig() for _ in range(4)]
        game = runner.start_game("test2", 4, 1000, 10, seats)
        pid = game.state.current_player_idx
        action = await runner.handle_ai_turn(game, pid, seats)
        assert action is not None

    def test_save_replay(self):
        store = ReplayStore(save_dir="/tmp/test_replays3")
        runner = GameRunner(store)
        seats = [SeatConfig() for _ in range(4)]
        game = runner.start_game("test3", 4, 1000, 10, seats)
        # 模拟对局到结束
        import random
        rng = random.Random(42)
        for _ in range(500):
            if game.is_finished():
                break
            pid = game.state.current_player_idx
            valid = game.get_valid_actions(pid)
            if not valid:
                break
            game.step(rng.choice(valid))

        replay_id = runner.save_replay("test3", game, seats, 1000, 10)
        assert replay_id
        loaded = store.get(replay_id)
        assert loaded is not None
        assert "result" in loaded
