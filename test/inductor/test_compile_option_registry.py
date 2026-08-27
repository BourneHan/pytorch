# Owner(s): ["module: inductor"]
import contextlib
import unittest
from unittest import mock

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import compile_fx_ext, compile_option_registry, config
from torch._inductor.codecache import FxGraphCache
from torch._inductor.compile_fx import (
    compile_fx,
    FxCompileMode,
    get_patched_config_dict,
)
from torch._inductor.compile_option_registry import (
    CompileOptionRoute,
    patch_compile_options,
    patch_routed_configs,
    register_compile_option,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal import fake_config_module
from torch.utils._import_utils import import_dill


dill = import_dill()
HAS_DILL = dill is not None


FAKE_MODULE = "torch.testing._internal.fake_config_module"
FAKE_MODULE2 = "torch.testing._internal.fake_config_module2"


def dummy_fn(x):
    return torch.sigmoid(x + 1.0) / 10.0


@contextlib.contextmanager
def fake_backend_routes():
    with mock.patch.dict(compile_option_registry._routes):
        register_compile_option("fake_backend_bool", module=FAKE_MODULE, key="e_bool")
        register_compile_option(
            "fake_backend2_bool", module=FAKE_MODULE2, key="e_aliasing_bool"
        )
        yield


class TestCompileOptionRegistry(TestCase):
    def test_register_and_lookup(self):
        with fake_backend_routes():
            route = CompileOptionRoute(module=FAKE_MODULE, key="e_bool")
            self.assertEqual(
                compile_option_registry.get_compile_option_route("fake_backend_bool"),
                route,
            )
            # dashed spelling resolves to the same route
            self.assertEqual(
                compile_option_registry.get_compile_option_route("fake-backend-bool"),
                route,
            )
        self.assertIsNone(
            compile_option_registry.get_compile_option_route("fake_backend_bool")
        )

    def test_register_with_module_object(self):
        with mock.patch.dict(compile_option_registry._routes):
            register_compile_option(
                "fake_backend_bool", module=fake_config_module, key="e_bool"
            )
            self.assertEqual(
                compile_option_registry.get_compile_option_route("fake_backend_bool"),
                CompileOptionRoute(module=FAKE_MODULE, key="e_bool"),
            )
            # targets are validated eagerly when the module object is passed
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                register_compile_option(
                    "bad_key", module=fake_config_module, key="missing_key"
                )

    def test_register_duplicate(self):
        with mock.patch.dict(compile_option_registry._routes):
            register_compile_option(
                "fake_backend_bool", module=FAKE_MODULE, key="e_bool"
            )
            # re-registering the same target is a no-op (backends may reload)
            register_compile_option(
                "fake-backend-bool", module=FAKE_MODULE, key="e_bool"
            )
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                register_compile_option(
                    "fake_backend_bool", module=FAKE_MODULE, key="e_string"
                )

    def test_register_invalid_name(self):
        with mock.patch.dict(compile_option_registry._routes):
            for name in ("not an identifier", "1abc", "class", "a.b", "\u212a"):
                with self.assertRaises(AssertionError):
                    register_compile_option(name, module=FAKE_MODULE, key="e_bool")

    def test_register_rejects_shadowing_inductor_config(self):
        with mock.patch.dict(compile_option_registry._routes):
            with self.assertRaisesRegex(AssertionError, "shadows"):
                register_compile_option(
                    "max_fusion_size", module=FAKE_MODULE, key="e_bool"
                )

    def test_apply_options_routed(self):
        with fake_backend_routes():
            wrapper = torch._TorchCompileInductorWrapper(
                None, {"fake_backend_bool": False, "max_fusion_size": 33}, None
            )
            self.assertEqual(wrapper.config["fake_backend_bool"], False)
            self.assertEqual(wrapper.config["max_fusion_size"], 33)
            # dashed spelling is normalized like inductor's own options
            wrapper = torch._TorchCompileInductorWrapper(
                None, {"fake-backend-bool": False}, None
            )
            self.assertEqual(wrapper.config, {"fake_backend_bool": False})
            # type is checked against the owning module
            with self.assertRaisesRegex(RuntimeError, "Unexpected type of attr"):
                torch._TorchCompileInductorWrapper(
                    None, {"fake_backend_bool": "not a bool"}, None
                )

    def test_apply_options_unknown_name_rejected(self):
        with fake_backend_routes():
            with self.assertRaisesRegex(RuntimeError, "Unexpected optimization option"):
                torch._TorchCompileInductorWrapper(
                    None, {"unregistered_option": True}, None
                )

    def test_apply_options_rejects_bad_route(self):
        with mock.patch.dict(compile_option_registry._routes):
            register_compile_option(
                "fake_backend_bool", module=FAKE_MODULE, key="missing_key"
            )
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                torch._TorchCompileInductorWrapper(
                    None, {"fake_backend_bool": True}, None
                )

            register_compile_option(
                "fake_backend_bool2", module="torch._inductor.compile_fx", key="e_bool"
            )
            with self.assertRaisesRegex(RuntimeError, "not a ConfigModule"):
                torch._TorchCompileInductorWrapper(
                    None, {"fake_backend_bool2": True}, None
                )

    def test_patch_compile_options(self):
        default_fusion_size = config.max_fusion_size
        with fake_backend_routes():
            with patch_compile_options(
                {"fake_backend_bool": False, "max_fusion_size": 64}
            ):
                self.assertFalse(fake_config_module.e_bool)
                self.assertEqual(config.max_fusion_size, 64)
            self.assertTrue(fake_config_module.e_bool)
            self.assertEqual(config.max_fusion_size, default_fusion_size)

    def test_patch_compile_options_partial_enter_rolls_back(self):
        # a later owner patch failing to enter must roll back the earlier ones
        default_fusion_size = config.max_fusion_size
        with mock.patch.dict(compile_option_registry._routes):
            register_compile_option(
                "fake_backend_bool", module=FAKE_MODULE, key="e_bool"
            )
            register_compile_option(
                "bad_backend_bool", module=FAKE_MODULE2, key="missing_key"
            )
            with self.assertRaisesRegex(AttributeError, "does not exist"):
                with patch_compile_options(
                    {
                        "max_fusion_size": 64,
                        "fake_backend_bool": False,
                        "bad_backend_bool": True,
                    }
                ):
                    pass
            self.assertEqual(config.max_fusion_size, default_fusion_size)
            self.assertTrue(fake_config_module.e_bool)

    def test_patch_compile_options_decorator_reentry(self):
        # backwards is compiled out of scope of the forward patch context; the
        # decorator form must re-enter every owner patch on each call
        with fake_backend_routes():
            observed = []

            def inner(x):
                observed.append((fake_config_module.e_bool, config.max_fusion_size))
                return x

            decorated = patch_compile_options(
                {"fake_backend_bool": False, "max_fusion_size": 64}
            )(inner)
        decorated(torch.zeros(1))
        self.assertEqual(observed, [(False, 64)])

    def test_patch_compile_options_reentrant(self):
        # the same instance must be safe to enter while already active, like
        # config.patch (nested graph compilation, or backward compiling on
        # another thread)
        with fake_backend_routes():
            patcher = patch_compile_options({"fake_backend_bool": False})
            with patcher:
                self.assertFalse(fake_config_module.e_bool)
                with patcher:
                    self.assertFalse(fake_config_module.e_bool)
                self.assertFalse(fake_config_module.e_bool)
            self.assertTrue(fake_config_module.e_bool)

    def test_compile_fx_patches_owner_module(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x * 2)

        gm = fx.symbolic_trace(M())
        x = torch.randn(4)
        observed = []

        def inner_compile(gm_, *args, **kwargs):
            observed.append((fake_config_module.e_bool, config.max_fusion_size))
            return gm_

        with fake_backend_routes():
            compiled = compile_fx(
                gm,
                [x],
                inner_compile=inner_compile,
                config_patches={"fake_backend_bool": False, "max_fusion_size": 64},
            )
        compiled(x)
        self.assertEqual(observed, [(False, 64)])
        # patches are undone once compilation is over
        self.assertTrue(fake_config_module.e_bool)

    def test_compile_fx_backward_sees_routed_patches(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x * 2) * x.cos()

        gm = fx.symbolic_trace(M())
        x = torch.randn(4, requires_grad=True)
        observed = []

        def inner_compile(gm_, *args, **kwargs):
            observed.append(fake_config_module.e_bool)
            return gm_

        with fake_backend_routes():
            compiled = compile_fx(
                gm,
                [x],
                inner_compile=inner_compile,
                config_patches={"fake_backend_bool": False},
            )
            out = compiled(x)
        out.sum().backward()
        # forward and the out-of-scope backward compilation both ran patched
        self.assertEqual(observed, [False, False])
        self.assertTrue(fake_config_module.e_bool)

    def test_patch_routed_configs(self):
        with patch_routed_configs({FAKE_MODULE: {"e_bool": False}}):
            self.assertFalse(fake_config_module.e_bool)
        self.assertTrue(fake_config_module.e_bool)

    @unittest.skipUnless(HAS_DILL, "dill not available")
    def test_compile_fx_ext_replays_routed_configs(self):
        # subprocess compile workers must see the routed values, not the
        # backend's defaults; SERIALIZE mode runs the same serialize ->
        # _run_in_child -> deserialize path in-process
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x * 2)

        gm = fx.symbolic_trace(M())
        x = torch.randn(4)
        snapshots_seen = []
        child_values = []
        real_patch_routed = compile_option_registry.patch_routed_configs

        def spy(snapshots):
            snapshots_seen.append(snapshots)
            patcher = real_patch_routed(snapshots)

            @contextlib.contextmanager
            def recording():
                with patcher:
                    child_values.append(fake_config_module.e_bool)
                    yield

            return recording()

        with fake_backend_routes():
            with (
                mock.patch(
                    "torch._inductor.compile_fx.fx_compile_mode",
                    FxCompileMode.SERIALIZE,
                ),
                mock.patch.object(compile_fx_ext, "patch_routed_configs", spy),
            ):
                compiled = compile_fx(
                    gm,
                    [x],
                    config_patches={"fake_backend_bool": False},
                )
                compiled(x)

        # only the modules patched by this compile are serialized; the second
        # registered owner was not used
        self.assertEqual(set(snapshots_seen[0]), {FAKE_MODULE})
        self.assertEqual(snapshots_seen[0][FAKE_MODULE]["e_bool"], False)
        self.assertEqual(child_values, [False])
        self.assertTrue(fake_config_module.e_bool)

    def test_get_patched_config_dict_routed(self):
        with fake_backend_routes():
            result = get_patched_config_dict(
                {
                    "fake_backend_bool": False,
                    "fake_backend2_bool": True,
                    "max_fusion_size": 64,
                }
            )
        self.assertEqual(result["fake_backend_bool"], False)
        self.assertEqual(result["fake_backend2_bool"], True)
        self.assertEqual(result["max_fusion_size"], 64)

    def test_torch_compile_routed_option(self):
        with fake_backend_routes():
            optimized = torch.compile(dummy_fn, options={"fake_backend_bool": False})
            # get_compiler_config runs at torch.compile() time, before any
            # device is known; routed keys must not crash it and must surface
            # their effective value
            compiler_config = optimized.get_compiler_config()
            self.assertEqual(compiler_config["fake_backend_bool"], False)
            x = torch.randn(10)
            torch.testing.assert_close(optimized(x), dummy_fn(x))
        # the owner module is restored once compilation is over
        self.assertTrue(fake_config_module.e_bool)

    def test_torch_inductor_compile_entry_point(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x * 2)

        gm = fx.symbolic_trace(M())
        x = torch.randn(4)
        with fake_backend_routes():
            compiled = torch._inductor.compile(
                gm, [x], options={"fake_backend_bool": False, "max_fusion_size": 64}
            )
        torch.testing.assert_close(compiled(x), M()(x))

    def test_standalone_compile_to_python_entry_point(self):
        # compile_to_python merges ``options`` into the config-patch block,
        # which must route vendor names like every other entry point
        from torch._inductor import compile_to_python
        from torch._inductor.decomposition import select_decomp_table
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(t):
            return [torch.relu(t * 2.0 + 1.0)]

        x = torch.randn(4)
        with torch.enable_grad():
            gm = make_fx(
                fn, decomposition_table=select_decomp_table(), tracing_mode="fake"
            )(x)
        with fake_backend_routes():
            src, _cache = compile_to_python(
                gm, [x], options={"fake_backend_bool": False, "max_fusion_size": 64}
            )
        self.assertTrue(fake_config_module.e_bool)
        ns = {"__name__": "_compiled"}
        exec(compile(src, "<compiled>", "exec"), ns)
        with torch.no_grad():
            torch.testing.assert_close(ns["call"]([x])[0], fn(x)[0])

    @config.patch("fx_graph_cache", True)
    @config.patch("fx_graph_remote_cache", False)
    @torch._functorch.config.patch("enable_autograd_cache", False)
    def test_fx_graph_cache_key_includes_routed_options(self):
        # compiles differing only in a routed option value must not share an
        # FX graph cache entry
        x = torch.randn(10)
        counters.clear()
        FxGraphCache.clear()

        with fake_backend_routes():
            torch._dynamo.reset()
            torch.compile(dummy_fn, options={"fake_backend_bool": False})(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

            torch._dynamo.reset()
            torch.compile(dummy_fn, options={"fake_backend_bool": True})(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            torch._dynamo.reset()
            torch.compile(dummy_fn, options={"fake_backend_bool": False})(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)


if __name__ == "__main__":
    run_tests()
