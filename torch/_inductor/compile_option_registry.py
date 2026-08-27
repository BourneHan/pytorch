"""
Registry routing user-facing ``torch.compile(..., options={...})`` names to the
backend-owned ConfigModule that holds them.

Out-of-tree Inductor backends keep their config in their own ConfigModule (see
``register_backend_for_device(device_custom_config=...)``).  The registry maps a
user-visible option name to such a module so the name is accepted by
``_TorchCompileInductorWrapper.apply_options`` (which runs before any device is
known) and patched onto the owning module instead of
``torch._inductor.config``:

    from torch._inductor.compile_option_registry import register_compile_option

    register_compile_option(
        "npu_backend", module="torch_npu._inductor.compile_config"
    )
    torch.compile(fn, options={"npu_backend": "mlir"})

Unregistered names keep falling back to ``torch._inductor.config``, and
``patch_compile_options`` resolves routes by name, so every entry point that
applies config patches (``compile_fx`` and through it ``torch.compile``,
``torch._inductor.compile``, AOTI, ``standalone_compile``) routes vendor
options identically.  Routes must be registered in every participating
process; backends typically do this where they call
``register_backend_for_device`` (e.g. from their torch.backends entry point,
so subprocess workers pick it up too).

Routed owner modules participate in the FX graph cache key through
:func:`active_routed_configs` (values patched during the compile are visible
when the key is computed), and ``compile_fx_ext`` subprocess workers replay
the routed values of the current compile through
:func:`snapshot_routed_configs` / :func:`patch_routed_configs`, mirroring how
they replay ``torch._inductor.config``.  Minifier repros still only carry the
backend module's global state, not per-compile patches.
"""

from __future__ import annotations

import contextlib
import importlib
import keyword
import sys
import threading
import unicodedata
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from torch.utils._config_module import ConfigModule
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from types import ModuleType


@dataclass(frozen=True)
class CompileOptionRoute:
    # Dotted path of the ConfigModule owning the option, stored as a string so
    # registering a route never imports the backend.
    module: str
    key: str


_routes: dict[str, CompileOptionRoute] = {}
_routes_lock = threading.Lock()
# Owner modules with routed patches active on this thread, one tuple of module
# names per _CompileOptionsPatch entry; read by snapshot_routed_configs().
_active_routed: ContextVar[tuple[tuple[str, ...], ...]] = ContextVar(
    "compile_option_registry.active_routed", default=()
)


def _normalize_option_name(name: str) -> str:
    normalized = name.replace("-", "_")
    if (
        unicodedata.normalize("NFKC", normalized) != normalized
        or not normalized.isidentifier()
        or keyword.iskeyword(normalized)
    ):
        raise AssertionError(f"compile option name {name!r} is not a valid identifier")
    return normalized


def register_compile_option(
    name: str,
    *,
    module: str | ModuleType,
    key: str | None = None,
) -> None:
    """
    Map a user-visible ``torch.compile(options=...)`` name to ``module.key``.

    ``module`` may be a ConfigModule object (validated immediately) or its
    dotted module path (validated the first time the option is used).
    Registering the same name to the same target again is a no-op, so backends
    may call this unconditionally on (re)load.
    """
    normalized = _normalize_option_name(name)
    module_name = module if isinstance(module, str) else module.__name__
    route = CompileOptionRoute(
        module=module_name, key=key if key is not None else normalized
    )
    if not isinstance(module, str):
        resolve_config_module(route)

    from torch._inductor import config

    if normalized in config.get_config_copy():
        raise AssertionError(
            f"compile option {normalized!r} shadows torch._inductor.config; "
            "routed option names must not collide with inductor's own config"
        )

    with _routes_lock:
        existing = _routes.get(normalized)
        if existing is not None and existing != route:
            raise RuntimeError(
                f"compile option {normalized!r} is already registered to "
                f"{existing.module}.{existing.key}"
            )
        _routes[normalized] = route


def get_compile_option_route(name: str) -> CompileOptionRoute | None:
    return _routes.get(name.replace("-", "_"))


def import_config_module(module_name: str) -> ConfigModule:
    module = importlib.import_module(module_name)
    if not isinstance(module, ConfigModule):
        raise RuntimeError(
            f"compile option owner {module_name!r} is not a ConfigModule"
        )
    return module


def resolve_config_module(route: CompileOptionRoute) -> ConfigModule:
    config_module = import_config_module(route.module)
    # flat _config membership; get_config_copy() would deepcopy every value
    if route.key not in config_module._config:
        raise RuntimeError(
            f"compile option target {route.module}.{route.key} does not exist"
        )
    return config_module


class _CompileOptionsPatch(contextlib.ContextDecorator):
    # Prior stacks live in ContextVars (like ConfigPatch's) so the same
    # instance is safe to re-enter while already active -- e.g. the decorated
    # inner_compile compiling a nested graph, or backward compiling on another
    # thread.
    def __init__(
        self, patches: list[Any], routed_modules: tuple[str, ...] = ()
    ) -> None:
        self._patches = patches
        self._routed_modules = routed_modules
        self._stacks: ContextVar[tuple[contextlib.ExitStack, ...]] = ContextVar(
            f"_CompileOptionsPatch[{id(self)}].stacks", default=()
        )

    def __enter__(self) -> None:
        stack = contextlib.ExitStack()
        try:
            for patch in self._patches:
                stack.enter_context(patch)
        except BaseException:
            # match ConfigPatch's rollback of a partially-applied patch set
            stack.close()
            raise
        self._stacks.set((*self._stacks.get(), stack))
        _active_routed.set((*_active_routed.get(), self._routed_modules))

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
        stacks = self._stacks.get()
        if not stacks:
            raise AssertionError("__exit__ called without matching __enter__")
        self._stacks.set(stacks[:-1])
        _active_routed.set(_active_routed.get()[:-1])
        return stacks[-1].__exit__(exc_type, exc_val, exc_tb)


def patch_compile_options(
    config_patches: dict[str, Any] | None,
) -> _CompileOptionsPatch:
    """
    Patch ``torch._inductor.config`` and every routed owner ConfigModule with
    the matching entries of ``config_patches``, resolving routes by name from
    the registry; unregistered entries are patched onto the core config.

    Usable as a context manager or a decorator; like ``config.patch``, the
    decorator form re-enters the patches on each call, which is how backward
    compilation (out of scope of the forward compile) keeps seeing them.
    """
    core_patches: dict[str, Any] = {}
    module_patches: dict[str, dict[str, Any]] = {}
    for name, value in (config_patches or {}).items():
        route = get_compile_option_route(name)
        if route is None:
            core_patches[name] = value
        else:
            module_patches.setdefault(route.module, {})[route.key] = value
    patches = []
    if core_patches:
        from torch._inductor import config

        patches.append(config.patch(core_patches))
    for module_name, owner_patches in module_patches.items():
        patches.append(import_config_module(module_name).patch(owner_patches))
    return _CompileOptionsPatch(patches, tuple(module_patches))


def active_routed_configs() -> dict[str, dict[str, Any]]:
    """
    Portable config of the owner modules with routed patches active on this
    thread.  ``FxGraphHashDetails`` hashes this so compiles differing in the
    routed options they set do not collide on the FX graph cache; owners also
    registered as ``device_custom_config`` are hashed through that mechanism
    regardless of patches.  Only the modules this compile actually patches
    participate, so an imported-but-unused backend never perturbs cache keys.
    """
    snapshots: dict[str, dict[str, Any]] = {}
    for module_name in {name for entry in _active_routed.get() for name in entry}:
        module = sys.modules.get(module_name)
        if isinstance(module, ConfigModule):
            snapshots[module_name] = module.save_config_portable(
                ignore_private_configs=False, readonly_values=True
            )
    return snapshots


def snapshot_routed_configs() -> dict[str, dict[str, Any]]:
    """
    Portable config of the owner modules with routed patches active on this
    thread -- the counterpart of ``config.save_config_portable()`` for routed
    options.  ``compile_fx_ext`` serializes this alongside the core config so
    subprocess workers replay the routed values of the current compile instead
    of silently compiling with the backend's defaults; compiles that use no
    routed options serialize nothing extra.  Owner modules whose values cannot
    be pickled make the serialization fail, which falls back to in-process
    compilation.
    """
    active = OrderedSet([name for entry in _active_routed.get() for name in entry])
    snapshots: dict[str, dict[str, Any]] = {}
    for module_name in active:
        module = sys.modules.get(module_name)
        if isinstance(module, ConfigModule):
            snapshots[module_name] = module.save_config_portable()
    return snapshots


def patch_routed_configs(
    snapshots: dict[str, dict[str, Any]],
) -> _CompileOptionsPatch:
    """
    Apply a :func:`snapshot_routed_configs` result; imports the owner modules,
    so a worker that has not loaded the backend yet picks up both its
    registration side effects and the parent's values.
    """
    return _CompileOptionsPatch(
        [
            import_config_module(module_name).patch(values)
            for module_name, values in snapshots.items()
        ]
    )
