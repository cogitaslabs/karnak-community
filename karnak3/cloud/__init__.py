import karnak3.core.engine_registry as kcer

_cloud_registry = kcer.KarnakRegistry('cloud')


class CloudEngine(kcer.KarnakEngine):
    def __init__(self, key: str):
        super().__init__(key)
        _cloud_registry.register(self)
