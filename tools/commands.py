import argparse


def parse_commands():
    parser = argparse.ArgumentParser(prog='BERT Trainning',
                                     usage='%(prog)s task',
                                     description='Run BERT trainning')

    parser.add_argument('--model',
                        action='store',
                        default='neuralmind/bert-base-portuguese-cased',
                        help='Base model to train')
    parser.add_argument('--epochs',
                        action='store',
                        default='1',
                        help='Define epochs number')
    args = vars(parser.parse_args())
    return args['model'], args['epochs']
