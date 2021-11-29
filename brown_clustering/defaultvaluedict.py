from typing import Generic, TypeVar

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class DefaultValueDict(dict, Generic[_KT, _VT]):
    def __init__(self, default_value: _VT, *args, **kwargs):
        self.default_value = default_value
        super().__init__(*args, **kwargs)

    def __getitem__(self, item: _KT):
        return self.get(item, self.default_value)
