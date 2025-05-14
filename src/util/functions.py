# taken from chatGPT
class Logger:
    def __init__(self, headers, widths=None):
        self.headers = headers
        if widths is None:
            self.widths = [max(len(str(h)), 10) for h in headers]
        else:
            self.widths = widths
        header_str = "\t".join(f"{h:<{w}}" for h, w in zip(self.headers, self.widths))
        print(header_str)

    def log(self, values):
        assert len(values) == len(self.headers)
        value_str = "\t".join(
            f"{v:<{w}.4f}" if isinstance(v, float) else f"{str(v):<{w}}"
            for v, w in zip(values, self.widths)
        )
        print(value_str)
