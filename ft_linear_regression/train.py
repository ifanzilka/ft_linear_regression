import matplotlib.pyplot as plt
import argparse
import signal
import os
import sys

def prRed(skk): return "\033[91m{}\033[00m".format(skk)
def prGreen(skk): return "\033[92m{}\033[00m".format(skk)
def prYellow(skk): return "\033[33m{}\033[0m".format(skk)
def prLightPurple(skk): return "\033[94m{}\033[00m".format(skk)
def prPurple(skk): return "\033[95m{}\033[00m".format(skk)
def prCyan(skk): return "\033[96m{}\033[00m".format(skk)
def prLightGray(skk): return "\033[97m{}\033[00m".format(skk)
def prBlack(skk): return "\033[98m {}\033[00m".format(skk)

def optparse():
    """
        Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-in', action="store", dest="input", type=str, default='../data/data.csv',
                        help='source of data file')

    parser.add_argument('--output', '-o', action="store", dest="output", type=str, default='thetas.txt',
                        help='source of data file')

    parser.add_argument('--iteration', '-it', action="store", dest="iter", type=int, default=0,
                        help='Change number of iteration. (default is Uncapped)')

    parser.add_argument('--history', '-hs', action="store_true", dest="history", default=False,
                        help='save history to futur display')

    parser.add_argument('--plotOriginal', '-po', action="store_true", dest="plot_original", default=False,
                        help="Enable to plot the original data sets")

    parser.add_argument('--plotNormalized', '-pn', action="store_true", dest="plot_normalized", default=False,
                        help="Enable to plot the normalized data sets")

    parser.add_argument('--learningRate', '-l', action="store", dest="rate", type=float, default=0.1,
                        help='Change learning coeficient. (default is 0.1)')

    parser.add_argument('--live', '-lv', action="store_true", dest="live", default=False,
                        help='Store live chnaged on gif graph')
    return parser.parse_args()


def signal_handler(sig, frame):
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    welcome = """
    ████████ ████████         ██       ████ ██    ██ ████████    ███    ████████          ████████  ████████  ██████   ████████  ████████  ██████   ██████  ████  ███████  ██    ██ 
    ██          ██            ██        ██  ███   ██ ██         ██ ██   ██     ██         ██     ██ ██       ██    ██  ██     ██ ██       ██    ██ ██    ██  ██  ██     ██ ███   ██ 
    ██          ██            ██        ██  ████  ██ ██        ██   ██  ██     ██         ██     ██ ██       ██        ██     ██ ██       ██       ██        ██  ██     ██ ████  ██ 
    ██████      ██            ██        ██  ██ ██ ██ ██████   ██     ██ ████████          ████████  ██████   ██   ████ ████████  ██████    ██████   ██████   ██  ██     ██ ██ ██ ██ 
    ██          ██            ██        ██  ██  ████ ██       █████████ ██   ██           ██   ██   ██       ██    ██  ██   ██   ██             ██       ██  ██  ██     ██ ██  ████ 
    ██          ██            ██        ██  ██   ███ ██       ██     ██ ██    ██          ██    ██  ██       ██    ██  ██    ██  ██       ██    ██ ██    ██  ██  ██     ██ ██   ███ 
    ██          ██            ████████ ████ ██    ██ ████████ ██     ██ ██     ██         ██     ██ ████████  ██████   ██     ██ ████████  ██████   ██████  ████  ███████  ██    ██ 

    """
    print(welcome)

    #if not os.path.exists('./gif'):
    #    os.makedirs('./gif')

    options = optparse()
    if (options.rate < 0.0000001 or options.rate > 1):
        options.rate = 0.1
    print("\033[33m{:s}\033[0m".format('Initial Params for training model:'))
    print(prCyan('    Learning Rate    : ') + str(options.rate))
    print(prCyan('    Max iterations   : ') + "Uncapped" if str(options.iter) == "0" else "0")
    print(prCyan('    Plot Original    : ') + ('Enabled' if options.plot_original else 'Disabled'))
    print(prCyan('    Plot Normalized  : ') + ('Enabled' if options.plot_normalized else 'Disabled'))
    print(prCyan('    Plot History     : ') + ('Enabled' if options.history else 'Disabled'))
    print(prCyan('    DataSets File    : ') + options.input)
    print(prCyan('    Output File      : ') + options.output)