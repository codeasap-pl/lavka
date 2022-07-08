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
class Config:
    normalize: bool = False
    n_times: int = 1
    plots_dir: str = "/tmp/benchmark-plots"


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

    def __call__(self, *args, **kwargs):
        return self.func(
            *(self.positional_args or []),
            **(self.keyword_args or {})
        )


@dataclasses.dataclass
class CaseGroup(Base):
    cases: list[Case] = dataclasses.field(default_factory=list)

    def add_case(self,
                 func: T.Callable,
                 /,
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
                identifier=(identifier or func.__qualname__),
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
        self.cfg = Config()

    def on_setup(self, *args, **kwargs): ...
    def on_teardown(self, *args, **kwargs): ...

    def add_group(self, name: T.Optional[str] = None) -> CaseGroup:
        assert not name or name not in self.groups, "Group: " + (name or "")
        group = CaseGroup(name)
        self.groups[group.identifier] = group
        return group

    def add_case(self,
                 func: T.Callable,
                 /,
                 args: T.Optional[tuple] = None,
                 kwargs: T.Optional[dict] = None,
                 identifier: T.Optional[str] = None):
        return self.groups["default"].add_case(
            Case(
                func=func,
                positional_args=args,
                keyword_args=kwargs,
                identifier=(identifier or func.__qualname__),
            )
        )

    def __call__(self, *args, **kwargs):
        return self.run(**kwargs)

    def run(self, **kwargs):
        for attr in self.cfg.__annotations__.keys():
            print(attr, kwargs.get(attr, getattr(self.cfg, attr)))

        # unhacking...
        [
            setattr(self.cfg, attr, kwargs.pop(attr, getattr(self.cfg, attr)))
            for attr in self.cfg.__annotations__.keys()
        ]

        for group_name, group in self.groups.items():
            if group:
                self.results[group_name] = []
                print("-" * 80)
                print("→", str(group))
                for case in group:
                    self.results[group_name].append(
                        self._run_case(case)
                    )
        print_results(self.results)
        self.create_plots()

    def _run_case(self, case):
        case.result.n_times = n_times = self.cfg.n_times
        n_ticks = n_times
        print("   → %24s %8d" % (case, n_times), end=" ")
        sys.stdout.flush()

        case.result.dt_points = {k: 0 for k in range(0, n_times, n_ticks)}

        self.on_setup()
        t0 = time.monotonic()
        tp = time.monotonic()

        try:
            for i in range(n_times):
                case(cfg=self.cfg)
                tmp_t = time.monotonic()
                case.result.dt_points[i] = tmp_t - tp
                tp = tmp_t
            case.result.total_time = time.monotonic() - t0
        finally:
            self.on_teardown()
        print(round(case.result.total_time, 8))
        return case

    def create_parser(self, **kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument("-N", "--normalize", action="store_true")
        parser.add_argument("-t", "--n-times", type=int,
                            default=kwargs.pop("n_times", 2 ** 10))
        parser.add_argument("-o", "--plots-dir", type=str,
                            default="/tmp/benchmark-plots")
        return parser

    def create_plots(self):
        plots_dir = self.cfg.plots_dir
        normalize = self.cfg.normalize
        os.makedirs(plots_dir, exist_ok=True)

        n_times = self.cfg.n_times
        for group_name, cases in self.results.items():
            if not self.groups[group_name]:
                continue

            if n_times == 1:
                results = []
                labels = []
                fig = plt.Figure(figsize=(16, 9), tight_layout=True)
                for case in cases:
                    _ = case.result.dt_points.keys()
                    ys = case.result.dt_points[0]
                    results.append(ys)
                    labels.append(str(case))

                fig.gca().bar(labels, results, width=0.5)
            else:
                fig = plt.Figure(figsize=(16, 9), tight_layout=True)
                for case in cases:
                    xs = case.result.dt_points.keys()
                    ys = case.result.dt_points.values()
                    if normalize:
                        miny, maxy = (min(ys), max(ys))
                        ys = [(y - miny) / (maxy - miny) for y in ys]
                    fig.gca().plot(xs, ys, label=str(case))

                fig.gca().set_ylabel("Δt", fontsize="x-large")
                fig.gca().set_xlabel("loops", fontsize="x-large")
                fig.legend(fontsize=12)

            title = "%s: %s" % (
                group_name,
                " / ".join([str(c) for c in cases])
            )
            title += "\n(normalized)" if normalize else ""
            fig.suptitle(title, fontsize=24, fontweight="bold")

            fig.gca().grid(True)

            filename = group_name.replace(" ", "-").replace("/", "-")
            filename = "".join([
                filename,
                "-normalized" if normalize else "",
                ".png"
            ])
            filename = os.path.join(plots_dir, filename)
            fig.savefig(filename)
            print("PLOT", group_name, filename)


def print_results(results, **kwargs):
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
    b(**args.__dict__)
