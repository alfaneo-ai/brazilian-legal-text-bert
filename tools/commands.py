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
                        default=2,
                        type=int,
                        help='Define number of batch size')
    parser.add_argument('--max_seq',
                        action='store',
                        default=384,
                        type=int,
                        help='Define max sequence size')
    parser.add_argument('--train_type',
                        action='store',
                        default='binary',
                        type=str,
                        help='Define train type (binary, scale, triplet, contrastive)')
    parser.add_argument('--sample',
                        action='store_true',
                        default=False,
                        help='Define is a sample')
    parser.add_argument('--to_lowercase',
                        action='store_true',
                        default=False,
                        help='Define if data should be converted to lowerase')
    args = vars(parser.parse_args())
    return args['model'], args['epochs'], args['batch_size'], args['max_seq'], args['train_type'], args['sample'], args['to_lowercase']
