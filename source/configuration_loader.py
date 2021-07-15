import argparse
import os


def parse_configuration_file():
    """
    loads all arguments given in a configuration file that was defined in sys.args with flag --configuration_file
    :rtype: namespace including all parsed elements

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration_file',
                        type=str,
                        default="./configuration_lab_pal.txt")

    class StoreDictKeyPair(argparse.Action):
        """
        formats dictionary arg flags of the form key:value
        """

        def __init__(self, option_strings, dest, nargs=None, value_type=str, **kwargs):
            self.value_type = value_type
            super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            my_dict = {}
            data = values.split(" ")
            for kv in data:
                a = kv.split(":")
                if len(a) > 1:
                    (k, v) = a
                    if v[0] == "[":
                        try:
                            my_dict[k] = eval(v)
                        except:
                            my_dict[k]=v
                            print("could not convert {} to float, it will be interpreted as string".format(v))
                    else:
                        v = v if v != "False" else False
                        v = v if v != "True" else True
                        try:
                            my_dict[k] = self.value_type(v)
                        except:
                                my_dict[k]=v
                                print("could not convert {} to float, it will be interpreted as string".format(v))
            setattr(namespace, self.dest, my_dict)

    config_args_parser = argparse.ArgumentParser()

    config_args_parser.add_argument('--experiment_name',
                                    type=str,
                                    default="test_experiment",
                                    help='the name of the experiment')
    config_args_parser.add_argument('--run_name',
                                    type=str,
                                    default="test_experiment_run",
                                    help=' unique name of the run ')
    config_args_parser.add_argument('--dataset_path',
                                    type=str,
                                    default='/home/mutschler/PycharmProjects/LineSearchExperiments/datasets/cifar10_dataset.npy',
                                    # default='/localhome/mutschler/PycharmProjects/LinesOfWellPerformingTrainings/Datasets/tolstoi.npy',
                                    # default= "/home/mutschler/mnt/tcml-cluster/ImageNet_TFRecords/",
                                    help='location of the dataset in numpy format', )
    config_args_parser.add_argument('--dataset_name',
                                    type=str,
                                    default='CIFAR-10',
                                    help='the dataset to use', )
    config_args_parser.add_argument('--model',
                                    type=str,
                                    default="resnet",  # 2,
                                    # choices=['resnet', "densenet"],
                                    help='network type',
                                    )
    config_args_parser.add_argument('--training_steps',
                                    type=int,
                                    default=1000,  # 2,
                                    help='training steps',
                                    )
    config_args_parser.add_argument('--batch_size',
                                    type=int,
                                    default=100,
                                    help='batch_size', )
    config_args_parser.add_argument('--train_data_size',
                                    type=int,
                                    default=2000,
                                    help='train data size,remaining elements define the  evaluation set', )
    config_args_parser.add_argument('--random_seed',
                                    type=int,
                                    default=1,
                                    help='random number seed for numpy and tensorflow to get same results for multiple runs', )
    config_args_parser.add_argument('--dataset_fraction',
                                    type=float,
                                    default=1.0,
                                    help='random chosen fraction of the train and val set to use', )
    config_args_parser.add_argument('--num_gpus',
                                    type=int,
                                    default=1,
                                    help='num gpus to train on', )
    config_args_parser.add_argument('--optimizer',
                                    type=str,
                                    default="SGD",
                                    help='the optimizer to use', )  #
    config_args_parser.add_argument("--optimizer_args",
                                    value_type=float,
                                    default={"learning rate": 0.1},
                                    action=StoreDictKeyPair,
                                    metavar="KEY:VAL")
    config_args_parser.add_argument('--schedule',
                                    type=str,
                                    default="",  # leads to None value
                                    help='the decay to use', )  #
    config_args_parser.add_argument("--schedule_args",
                                    value_type=float,
                                    default={"boundaries": [1000, 1500, 2000], "values": [1.0, 0.1, 0.01]},
                                    action=StoreDictKeyPair,
                                    metavar="KEY:VAL")
    config_args_parser.add_argument("--additional",
                                    value_type=float,
                                    default={"decay_rate": 0.1},
                                    action=StoreDictKeyPair,
                                    metavar="KEY:VAL",
                                    required=False)

    parser_FLAGS, unparsed = parser.parse_known_args()

    with open(parser_FLAGS.configuration_file) as f:
        config_args = f.read()
    config_args = os.path.expandvars(config_args)  # needed that ${SLURM_JOB_ID} is expanded
    config_args = config_args.split("\n")
    config_args = ["--" + s for s in config_args if s is not ""]

    config_args_parser_FLAGS, unparsed = config_args_parser.parse_known_args(config_args)
    config_args_parser_FLAGS.dataset_path = os.path.expanduser(config_args_parser_FLAGS.dataset_path)
    print("-" * 25)
    print("Configuration loaded:")
    print("-" * 25)
    for k, v in vars(config_args_parser_FLAGS).items():
        k, v = str(k), str(v)
        print('%s: %s' % (k, v))
    print("-" * 25)

    return config_args_parser_FLAGS
