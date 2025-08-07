import argparse

def arg_conf():
    parser = argparse.ArgumentParser()
    
    # arguments of environment
    parser.add_argument('--random_seed', type=int, default=11)

    # arguments of model
    parser.add_argument('--input_shape', nargs='+', type=int, default=[1, 150, 220])
    parser.add_argument('--output_dim', type=int, default=1024)
    # parguments of data
    parser.add_argument('--dataset', type=str, choices=['cedar', 'hansig', 'bengali', 'hindi'], default='cedar')
    parser.add_argument('--batch_train', type=int, choices=[18, 20], default=18)
    parser.add_argument('--batch_valid', type=int, default=100)
    parser.add_argument('--batch_test', type=int, default=1)
    parser.add_argument('--data_seed', type=int, default=11)
    # arguments of training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--loss_epsilon', type=float, choices=[0.2, 0.3], default=0.2)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--n_epoch', type=int, default=90)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--prev_loss', type=int, default=100)
    parser.add_argument('--trig_times', type=int, default=0)
    parser.add_argument('--saved_name', type=str, default='saved_model/min_loss_cedar_')

    args = parser.parse_known_args()[0]
    #print(vars(args))

    return args


if __name__ == "__main__":
    args = arg_conf()