import karnak3.core.engine_registry as kcer

_store_registry = kcer.KarnakRegistry('store')


class StoreResource(kcer.KarnakUriResource):
    pass


class StoreEngine(kcer.KarnakUriEngine):
    def __init__(self, scheme: str):
        super().__init__(scheme)
        _store_registry.register(self)
