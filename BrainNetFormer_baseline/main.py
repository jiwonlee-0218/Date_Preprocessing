import util
from experiment import train, tt_function


if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()

    # run and analyze experiment
    # if not argv.no_train: train(argv)
    if not argv.no_test: tt_function(argv)
    exit(0)
