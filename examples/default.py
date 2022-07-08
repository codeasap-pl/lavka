from lavka import Benchmark
import time


def func_a(*args, **kwargs):
    time.sleep(0.01)


def func_b(*args, **kwargs):
    time.sleep(0.02)


if __name__ == "__main__":
    b = Benchmark()
    b.add_case(func_a)
    b.add_case(func_b)

    parser = b.create_parser()
    args = parser.parse_args()
    b(**args.__dict__)
