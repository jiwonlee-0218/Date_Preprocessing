import util
from experiment import train, tt_function
from analysis import analyze


if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()

    # run and analyze experiment
    if not any([argv.train, argv.test, argv.analyze]): argv.train = argv.test = argv.analyze = True
    # if argv.train: train(argv)
    if argv.test: tt_function(argv)
    # if argv.analyze: analyze(argv)
    exit(0)
