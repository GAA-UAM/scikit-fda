import lazy_loader as lazy  # noqa: D104

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "anova",
        "hotelling",
    ],
)
