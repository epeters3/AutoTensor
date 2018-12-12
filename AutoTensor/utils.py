import sys


def print_no_nl(string):
    """
    Prints with no new line, and flushes so doesn't wait
    for the rest of the print statements on the line to
    evaluate before printing.
    """
    sys.stdout.write(string)
    sys.stdout.flush()