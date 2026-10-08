"""Config read-modify-write cycles must not lose concurrent updates."""

from __future__ import annotations

import threading

from anomx.agent.store import AnomxHome


def test_update_config_writes_only_given_keys_on_fresh_config(tmp_path):
    home = AnomxHome(tmp_path / "home")
    stale = home.load_config()
    home.update_config({"user_name": "Ada"})

    # An older snapshot must not undo the change when another key is updated.
    home.update_config({"debug_mode": True})

    assert stale["user_name"] == ""
    loaded = home.load_config()
    assert loaded["user_name"] == "Ada"
    assert loaded["debug_mode"] is True


def test_concurrent_updates_of_different_keys_are_all_kept(tmp_path):
    home = AnomxHome(tmp_path / "home")
    rounds = 40

    def worker(prefix: str) -> None:
        for index in range(rounds):
            if prefix == "a":
                home.add_global_allowed_command(f"cmd:a{index}")
            else:
                home.add_global_rejected_command(f"cmd:r{index}")

    threads = [threading.Thread(target=worker, args=(prefix,)) for prefix in ("a", "r")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    assert len(home.load_global_allowed_commands()) == rounds
    assert len(home.load_global_rejected_commands()) == rounds


def test_config_lock_is_reentrant_and_save_inside_lock_works(tmp_path):
    home = AnomxHome(tmp_path / "home")

    with home._config_lock():
        config = home.load_config()
        config["user_name"] = "Grace"
        home.save_config(config)
        with home._config_lock():
            home.update_config({"debug_mode": True})

    loaded = home.load_config()
    assert loaded["user_name"] == "Grace"
    assert loaded["debug_mode"] is True
    assert home._config_lock_depth == 0


def test_config_write_leaves_no_temporary_files(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.update_config({"user_name": "Ada"})

    leftovers = [path.name for path in home.root.glob("config.toml*") if path.suffix == ".tmp"]
    assert leftovers == []
