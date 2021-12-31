import argparse
import dataclasses
import os
import sys
import time

import matplotlib.pyplot as plt
import typing as T


__SEQ_IDS = {}


def _mkseq(cls):
    if cls not in __SEQ_IDS:
        def _nextval_gen(cls):
            while 1:
                yield cls._SEQ_ID
                cls._SEQ_ID += 1
        __SEQ_IDS[cls] = _nextval_gen(cls)
    return __SEQ_IDS[cls]


@dataclasses.dataclass
class Base(object):
    identifier: T.Optional[str] = None

    def __new__(cls, *args, **kwargs):
        cls._SEQ_ID = 0
        cls._seqgen = _mkseq(cls)
        obj = super(Base, cls).__new__(cls)
        return obj

    def __post_init__(self, *args, **kwargs):
        self.identifier = self.identifier or (
            "%s:%s" % (self.__class__.__name__, next(self._seqgen))
        )

    def __str__(self):
        return self.identifier


@dataclasses.dataclass
class Result(Base):
    n_times: int = 0,
    total_time: float = 0.0
    dt_points: list = dataclasses.field(default_factory=list)

    def mean(self):
        return self.n_times / self.total_time


@dataclasses.dataclass
class Case(Base):
    func: T.Callable = lambda: None
    positional_args: tuple = dataclasses.field(default_factory=tuple)
    keyword_args: dict = dataclasses.field(default_factory=dict)
    result: Result = dataclasses.field(default_factory=Result)

    def __str__(self):
        return self.identifier or self.func.__qualname__

    def __call__(self, po: argparse.Namespace):
        return self.func(*self.positional_args, **self.keyword_args)


@dataclasses.dataclass
class CaseGroup(Base):
    cases: list[Case] = dataclasses.field(default_factory=list)

    def add_case(self,
                 func: T.Callable,
                 args: T.Optional[tuple] = None,
                 kwargs: T.Optional[dict] = None,
                 identifier: T.Optional[str] = None):
        assert callable(func), "Callable"
        case = func
        if not isinstance(func, Case):
            case = Case(
                func=func,
                positional_args=(args or tuple()),
                keyword_args=(kwargs or dict()),
                identifier=identifier,
            )
        self.cases.append(case)
        return case

    def __len__(self):
        return len(self.cases)

    def __bool__(self):
        return bool(self.cases)

    def __iter__(self):
        return iter(self.cases)


class Benchmark:
    def __init__(self):
        self._platform_info = {
            "python": {
                "version": sys.version_info
            }
        }

        self.groups = {
            "default": CaseGroup("default"),
        }

        self.results = {}

    def on_setup(self, po: argparse.Namespace): ...
    def on_teardown(self, po: argparse.Namespace): ...

    def add_group(self, name: T.Optional[str] = None) -> CaseGroup:
        assert not name or name not in self.groups, "Group: " + (name or "")
        group = CaseGroup(name)
        self.groups[group.identifier] = group
        return group

    def add_case(self,
                 func: T.Callable,
                 positional_args: tuple,
                 keyword_args: dict,
                 /,
                 identifier: T.Optional[str] = None):
        return self.groups["default"].add_case(
            func,
            positional_args,
            keyword_args,
            identifier=identifier,
        )

    def __call__(self, po: argparse.Namespace):
        for group_name, group in self.groups.items():
            if group:
                self.results[group_name] = []
                print("-" * 80)
                print("→", str(group))
                for case in group:
                    self.results[group_name].append(
                        self._run_case(po, case)
                    )
        print_results(po, self.results)
        self.create_plots(po)

    def _run_case(self, po, case):
        case.result.n_times = n_times = po.n_times
        n_ticks = po.n_ticks
        print("   → %24s %8d" % (case, n_times), end=" ")
        sys.stdout.flush()
        case.result.dt_points = {
            k: 0 for k in range(0, n_times, int(n_times / n_ticks))
        }

        self.on_setup(po)
        t0 = time.monotonic()
        tp = time.monotonic()
        try:
            for i in range(n_times):
                case(po)
                if n_ticks and i in case.result.dt_points:
                    tmp_t = time.monotonic()
                    case.result.dt_points[i] = tmp_t - tp
                    tp = tmp_t
            case.result.total_time = time.monotonic() - t0
        finally:
            self.on_teardown(po)
        print(round(case.result.total_time, 8))
        return case

    def create_parser(self, **kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument("-N", "--normalize", action="store_true")
        parser.add_argument("-t", "--n-times", type=int,
                            default=kwargs.pop("n_times", 2 ** 10))
        parser.add_argument("-x", "--n-ticks", type=int,
                            default=kwargs.pop("n_ticks", 64))
        parser.add_argument("-o", "--plots-dir", type=str,
                            default="/tmp/benchmark-plots")
        return parser

    def create_plots(self, po):
        os.makedirs(po.plots_dir, exist_ok=True)
        for group_name, cases in self.results.items():
            if not self.groups[group_name]:
                continue
            fig = plt.Figure(figsize=(16, 9), tight_layout=True)
            for case in cases:
                xs = case.result.dt_points.keys()
                ys = case.result.dt_points.values()
                if po.normalize:
                    miny, maxy = (min(ys), max(ys))
                    ys = [(y - miny) / (maxy - miny) for y in ys]  # normalized
                fig.gca().plot(xs, ys, label=case.identifier)

            title = "%s: %s" % (
                group_name,
                " / ".join([str(c) for c in cases])
            )
            title += "\n(normalized)" if po.normalize else ""
            fig.suptitle(title, fontsize=32, fontweight="bold")

            fig.gca().set_ylabel("Δt", fontsize="x-large")
            fig.gca().set_xlabel("loops", fontsize="x-large")
            fig.legend(fontsize=16)
            fig.gca().grid(True)

            filename = group_name.replace(" ", "-").replace("/", "-")
            filename = "".join([
                filename,
                "-normalized" if po.normalize else "",
                ".png"
            ])
            filename = os.path.join(po.plots_dir, filename)
            fig.savefig(filename)
            print("PLOT", group_name, filename)


def print_results(po, results):
    _str = (lambda w: lambda v: str(v)[:w])
    _flt = (lambda p: lambda v: str(round(float(v), p)))

    fields = "group_name identifier n_times total_time mean     "  # noqa: E202
    h_spec = "<20s       <20s       <8s      <12s       <12s    "  # noqa: E202
    h_func = [_str(20),  _str(20),  _str(8), _str(12),  _str(12)]  # noqa: E202
    r_func = [_str(20),  _str(20),  _str(8), _flt(8),   _flt(2) ]  # noqa: E202
    fields = list(filter(bool, fields.split()))
    h_spec = list(filter(bool, h_spec.split()))

    fmt = "| ".join(["{%s:%s}" % (f, v) for f, v in zip(fields, h_spec)])

    h_map = dict(zip(fields, h_func))
    r_map = dict(zip(fields, r_func))

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    headers = dict(
        group_name="GROUP",
        identifier="CASE",
        n_times="N_TIMES",
        total_time="TOTAL TIME",
        mean="N/SEC"
    )
    print(fmt.format(**{f: h_map[f](v) for f, v in headers.items()}))
    for group_name, cases in results.items():
        print("-" * 80)
        for case in sorted(cases, key=(lambda c: c.result.total_time)):
            data = dict(
                group_name=group_name,
                mean=case.result.mean(),
                total_time=case.result.total_time,
                n_times=case.result.n_times,
                identifier=case.identifier,
            )
            print(fmt.format(**{f: r_map[f](v) for f, v in data.items()}))
    print("-" * 80)


if __name__ == "__main__":
    foo = (lambda *args, **kwargs: (args, kwargs))

    b = Benchmark()
    group_a = b.add_group("testing")
    group_a.add_case(foo, (None,), {}, "A")
    group_a.add_case(foo, (None,), {}, "B")

    group_b = b.add_group("another")
    group_b.add_case(foo, (1,), {"example_b": 1}, "A")
    group_b.add_case(foo, (2,), {"example_b": 2}, "B")

    parser = b.create_parser()
    args = parser.parse_args()
    b(args)
