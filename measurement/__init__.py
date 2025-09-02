from multiprocessing.sharedctypes import Synchronized

if not hasattr(Synchronized, "__class_getitem__"):
    Synchronized.__class_getitem__ = lambda *_, **__: Synchronized
