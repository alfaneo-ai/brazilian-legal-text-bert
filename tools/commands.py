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
                        default=1,
                        type=int,
                        help='Define epochs number')
    parser.add_argument('--batch_size',
                        action='store',
                        default=8,
                        type=int,
                        help='Define number of batch size')
    parser.add_argument('--max_seq',
                        action='store',
                        default=384,
                        type=int,
                        help='Define max sequence size')
    args = vars(parser.parse_args())
    return args['model'], args['epochs'], args['batch_size'], args['max_seq']
