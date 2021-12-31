import json
import pickle
import marshal
import dataclasses
import pydantic

from lavka import Benchmark


DATA = dict(
    my_int=3.28 * 10 ** 80,
    my_str="a" * 1024,
    my_list=list(range(1024)),
    my_dict={n: n ** 2 for n in range(1024)},
)


def seqgen(cls):
    cls._SEQ_ID = 0

    def _next():
        while 1:
            yield cls._SEQ_ID
            cls._SEQ_ID += 1

    return _next()


class PlainClass:
    def __init__(self, **kwargs):
        self.__id = kwargs.pop("__id", None) or next(PlainClass.SEQ)
        self.my_int = kwargs.pop("my_int", DATA["my_int"])
        self.my_str = kwargs.pop("my_str", DATA["my_str"])
        self.my_list = kwargs.pop("my_list", DATA["my_list"])
        self.my_dict = kwargs.pop("my_dict", DATA["my_dict"])


PlainClass.SEQ = seqgen(PlainClass)


@dataclasses.dataclass
class DataClass:
    __id: int = dataclasses.field(default_factory=lambda: next(DataClass.SEQ))
    my_int: int = dataclasses.field(default_factory=lambda: DATA["my_int"])
    my_str: str = dataclasses.field(default_factory=lambda: DATA["my_str"])
    my_list: list = dataclasses.field(default_factory=lambda: DATA["my_list"])
    my_dict: dict = dataclasses.field(default_factory=lambda: DATA["my_dict"])


DataClass.SEQ = seqgen(DataClass)


class PydanticClass(pydantic.BaseModel):
    __id: int = pydantic.Field(default_factory=lambda: next(PydanticClass.SEQ))
    my_int: int = pydantic.Field(default_factory=lambda: DATA["my_int"])
    my_str: str = pydantic.Field(default_factory=lambda: DATA["my_str"])
    my_list: list = pydantic.Field(default_factory=lambda: DATA["my_list"])
    my_dict: dict = pydantic.Field(default_factory=lambda: DATA["my_dict"])


PydanticClass.SEQ = seqgen(PydanticClass)


if __name__ == "__main__":
    types = [PlainClass, DataClass, PydanticClass]
    serializers = [json, pickle, marshal]

    b = Benchmark()
    grp_init = b.add_group("initialization")
    [
        grp_init.add_case(f, kwargs=DATA, identifier=f.__qualname__)
        for f in types
    ]

    grp_init_defaults = b.add_group("init (defaults)")
    [
        grp_init_defaults.add_case(f, kwargs=DATA, identifier=f.__qualname__)
        for f in types
    ]

    grp_serializers = b.add_group("serializers")
    [
        grp_serializers.add_case(f.dumps, args=(DATA,), identifier=f.__name__)
        for f in serializers
    ]

    grp_deserializers = b.add_group("deserializers")
    [
        grp_deserializers.add_case(
            f.loads,
            args=(f.dumps(DATA),),
            identifier=f.__name__
        )
        for f in serializers
    ]

    parser = b.create_parser()
    args = parser.parse_args()
    b(args)

