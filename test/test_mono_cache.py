"""Cache and launch-flow regression tests; ROS actions are lightweight doubles."""

import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class Action:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Description:
    def __init__(self, actions):
        self.actions = actions


def load_mono():
    exports = {
        "launch": {"LaunchDescription": Description},
        "launch_ros": {},
        "launch_ros.actions": {"Node": Action},
        "launch.logging": {"get_logger": lambda name: Mock()},
        "launch.actions": {"Shutdown": Action, "RegisterEventHandler": Action},
        "launch.event_handlers": {"OnProcessExit": Action},
        "ament_index_python": {},
        "ament_index_python.packages": {"get_package_prefix": Mock()},
    }
    modules = {}
    for name, attributes in exports.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    path = pathlib.Path(__file__).resolve().parents[1] / "launch/mono.py"
    spec = importlib.util.spec_from_file_location("mono_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


mono = load_mono()
TRAJECTORY = "#timestamp,x,y,z,qw,qx,qy,qz\n0,0,0,0,1,0,0,0\n"


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.source = self.root / "source.hpp"
        self.source.write_text("original", encoding="utf-8")
        self.output = self.root / "RK4Estimator.csv"
        self.manifest = self.root / "cache.json"
        self.paths = [self.source]
        self.params = {"estimators": ["RK4Estimator"], "confidence": 1.0}

    def cache(self):
        return mono.TrajectoryCache(
            self.manifest, self.params, lambda: list(self.paths), [self.output])

    def completed(self):
        cache = self.cache()
        cache.begin()
        self.output.write_text(TRAJECTORY, encoding="utf-8")
        cache.finish()
        return cache

    def test_first_run_then_reuse(self):
        self.assertFalse(self.cache().is_current())
        self.completed()
        self.assertTrue(self.cache().is_current())

    def test_timestamp_only_change_invalidates(self):
        self.completed()
        stat = self.source.stat()
        os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertFalse(self.cache().is_current())

    def test_md5_detects_change_with_same_timestamp_and_size(self):
        self.completed()
        stat = self.source.stat()
        self.source.write_text("modified", encoding="utf-8")
        os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(self.cache().is_current())

    def test_parameter_change_invalidates(self):
        self.completed()
        self.params = {**self.params, "confidence": 2.0}
        self.assertFalse(self.cache().is_current())

    def test_dependency_added_or_removed_invalidates(self):
        self.completed()
        extra = self.root / "added.hpp"
        extra.write_text("new header", encoding="utf-8")
        self.paths.append(extra)
        self.assertFalse(self.cache().is_current())
        self.paths = []
        self.assertFalse(self.cache().is_current())

    def test_missing_input_does_not_reuse_cache(self):
        self.completed()
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.cache()

    def test_output_changed_or_deleted_invalidates(self):
        self.completed()
        stat = self.output.stat()
        self.output.write_text(TRAJECTORY.replace("0,0,0,0,1", "0,9,0,0,1"),
                               encoding="utf-8")
        os.utime(self.output, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(self.cache().is_current())
        self.output.unlink()
        self.assertFalse(self.cache().is_current())

    def test_corrupt_manifest_invalidates(self):
        self.completed()
        for content in ("{", "null", "[]", '{"version": 999}'):
            with self.subTest(content=content):
                self.manifest.write_text(content, encoding="utf-8")
                self.assertFalse(self.cache().is_current())

    def test_failed_run_invalidates_previous_cache(self):
        self.completed()
        self.cache().begin()
        self.assertFalse(self.manifest.exists())
        self.assertFalse(self.cache().is_current())

    def test_unchanged_output_is_not_saved_as_new_result(self):
        cache = self.completed()
        cache.begin()
        with self.assertRaises(RuntimeError):
            cache.finish()
        self.assertFalse(self.manifest.exists())

    def test_empty_output_is_not_cached(self):
        cache = self.cache()
        cache.begin()
        self.output.write_text("#header\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            cache.finish()
        self.assertFalse(self.manifest.exists())

    def test_every_enabled_estimator_must_produce_output(self):
        other = self.root / "FuseEstimator.csv"
        cache = mono.TrajectoryCache(
            self.manifest, self.params, lambda: self.paths, [self.output, other])
        cache.begin()
        self.output.write_text(TRAJECTORY, encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            cache.finish()
        self.assertFalse(self.manifest.exists())

    def test_dependency_change_during_run_is_not_cached(self):
        cache = self.cache()
        cache.begin()
        self.output.write_text(TRAJECTORY, encoding="utf-8")
        self.source.write_text("edited during run", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            cache.finish()
        self.assertFalse(self.manifest.exists())

    def test_source_override_and_dependency_discovery(self):
        root = self.root / "repo"
        source = root / "src/VisualSim/VisualInertial.cpp"
        source.parent.mkdir(parents=True)
        source.write_text("// source", encoding="utf-8")
        with patch.dict(os.environ, {"EUROC_VIO_SOURCE_DIR": str(root)}):
            self.assertEqual(mono.find_source_root(self.root), root.resolve())
        header = root / "include/euroc_vio/Estimator.hpp"
        header.parent.mkdir(parents=True)
        header.write_text("// header", encoding="utf-8")
        paths = mono.estimation_dependencies(
            {"path_imu_csv": str(self.source)}, self.root / "install", root)
        self.assertIn(source, paths)
        self.assertIn(header, paths)
        self.assertIn(self.source, paths)
        self.assertNotIn(self.output, paths)

    def test_normal_install_discovers_sources_from_cmake_or_workspace(self):
        root = self.root / "src/nested/repo"
        source = root / "src/VisualSim/VisualInertial.cpp"
        source.parent.mkdir(parents=True)
        source.write_text("// source", encoding="utf-8")
        cmake = self.root / "build/euroc_vio/CMakeCache.txt"
        cmake.parent.mkdir(parents=True)
        cmake.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={root}\n", encoding="utf-8")
        installed = self.root / "install/euroc_vio/share/euroc_vio/launch/mono.py"
        with patch.dict(os.environ, {}, clear=True), patch.object(mono, "__file__", str(installed)):
            self.assertEqual(mono.find_source_root(self.root), root.resolve())
            cmake.unlink()
            self.assertEqual(mono.find_source_root(self.root), root.resolve())
            source.unlink()
            self.assertIsNone(mono.find_source_root(self.root))


class LaunchFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = pathlib.Path(self.temporary.name)
        self.workdir = self.home / "vio_ws"
        self.workdir.mkdir()
        self.dependency = self.workdir / "input.csv"
        self.dependency.write_text("input", encoding="utf-8")
        self.output = self.workdir / "RK4Estimator.csv"
        self.manifest = self.workdir / ".mono_trajectory_cache.json"
        for patcher in (
            patch.object(mono.pathlib.Path, "home", return_value=self.home),
            patch.object(mono, "get_package_prefix", return_value=str(self.home)),
            patch.object(mono, "find_source_root", return_value=self.home),
            patch.object(mono, "estimation_dependencies", return_value=[self.dependency]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def start(self):
        description = mono.generate_launch_description()
        handler, factory = description.actions
        self.assertEqual(factory.kwargs["executable"], "VisualInertial")
        event_handler = handler.kwargs["event_handler"]
        self.assertIs(event_handler.kwargs["target_action"], factory)
        return event_handler.kwargs["on_exit"]

    def test_success_then_launch_only_loaders_and_rviz(self):
        callback = self.start()
        self.output.write_text(TRAJECTORY, encoding="utf-8")
        nodes = callback(types.SimpleNamespace(returncode=0), None)
        self.assertTrue(self.manifest.exists())
        self.assertEqual(len(nodes), 3)
        cached = mono.generate_launch_description()
        self.assertEqual([node.kwargs["executable"] for node in cached.actions],
                         ["SimpleDataLoader", "SimpleDataLoader", "rviz2"])

    def test_failure_stops_without_loading_old_trajectory(self):
        self.output.write_text(TRAJECTORY, encoding="utf-8")
        callback = self.start()
        actions = callback(types.SimpleNamespace(returncode=1), None)
        self.assertIn("reason", actions[0].kwargs)
        self.assertFalse(self.manifest.exists())

    def test_zero_exit_without_output_stops(self):
        callback = self.start()
        actions = callback(types.SimpleNamespace(returncode=0), None)
        self.assertIn("reason", actions[0].kwargs)
        self.assertFalse(self.manifest.exists())

    def test_debug_does_not_save_cache(self):
        with patch.object(mono, "debug", True):
            callback = self.start()
            self.output.write_text(TRAJECTORY, encoding="utf-8")
            callback(types.SimpleNamespace(returncode=0), None)
        self.assertFalse(self.manifest.exists())


if __name__ == "__main__":
    unittest.main()
