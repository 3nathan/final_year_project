# taken from chatGPT
def print_rl_log(headers, values, widths=None):
    """
    Pretty-print RL stats in tabular form using tabs.

    Args:
        headers (list of str): Column headers.
        values (list): Corresponding values (can be float, int, str).
        widths (list of int, optional): Widths for each column (for alignment).
    """
    assert len(headers) == len(values), "Headers and values must match in length"

    if widths is None:
        widths = [max(len(str(h)), 10) for h in headers]

    # Format header
    header_str = "\t".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    value_str = "\t".join(
        f"{v:<{w}.4f}" if isinstance(v, float) else f"{str(v):<{w}}"
        for v, w in zip(values, widths)
    )

    print(header_str)
    print(value_str)

