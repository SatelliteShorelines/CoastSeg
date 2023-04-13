class Observable:
    def __init__(self, initial_value=None, name=""):
        self._value = initial_value
        self._observers = []
        self.name = name

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def set(self, value):
        self._value = value
        self._notify_observers()

    def add(self, value):
        if isinstance(self._value, list):
            if isinstance(value, list):
                self._value.extend(value)
            else:
                self._value.append(value)
        self._notify_observers()

    def get(self):
        return self._value

    def _notify_observers(self):
        for observer in self._observers:
            observer.update(self)
