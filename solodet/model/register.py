"""Register custom modules with Ultralytics so YAML parser can resolve them."""

_registered = False


def register_custom_modules() -> None:
    """Register CBAM and ECA modules in the Ultralytics module namespace.

    Must be called before loading a model YAML that references custom modules.
    Safe to call multiple times.
    """
    global _registered
    if _registered:
        return

    import ultralytics.nn.modules as modules

    from solodet.model.attention import CBAM, ECA

    # Register in the modules namespace
    modules.CBAM = CBAM
    modules.ECA = ECA

    # Also register in the __all__ list if it exists
    if hasattr(modules, "__all__"):
        for name in ("CBAM", "ECA"):
            if name not in modules.__all__:
                modules.__all__.append(name)

    _registered = True
