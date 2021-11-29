class DefaultValueDict(dict):
    def __init__(self, default_value, *args, **kwargs):
        self.default_value = default_value
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        return self.get(item, self.default_value)
