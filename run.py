#  +-+ +-+ +-+ +-+ +-+ +-+
#  |i| |m| |p| |o| |r| |t|
#  +-+ +-+ +-+ +-+ +-+ +-+

import argparse

from main import main
from training import evaluation_summary

#  +-+ +-+ +-+
#  |r| |u| |n|
#  +-+ +-+ +-+


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("-exp", "--exp",'--experiment_name', help='experiment name', default='exp')
    parser.add_argument("-dn",'--dataset_name', help='dataset name or folder name', default='ucmerced')
    parser.add_argument("-mn",'--model_name', help='model username should be in torchvision models', default='mobilenet_v2')
    # save paths
    parser.add_argument("-lp",'--log_path', help='save log', default='./results/log/', type=str)
    parser.add_argument("-cp",'--checkpoints_path', help='save checkpoints', default='./results/checkpoints/', type=str)
    parser.add_argument("-ep",'--evals_path', help='save evaluations', default='./results/evals/', type=str)
    # running option 
    parser.add_argument("-tr","--train", help="training option (default: False)", default=False, action="store_true")
    parser.add_argument("-ev","--eval", help="evaluation option (default: False)", default=False, action="store_true")
    parser.add_argument("-evs","--eval_summary", help="evaluation summary option (default: False)", default=False, action="store_true")
    # hyperparameters
    parser.add_argument("-epc",'--epochs', help='training iteration number', default=100, type=int)
    parser.add_argument("-bs",'--batch_size', help='training/evaluation batch size', default=16, type=int)
    parser.add_argument("-is",'--image_size', help='image size to resize', default=224, type=int)
    parser.add_argument("-lr",'--learning_rate', help='training learning rate ', default=0.0001, type=float)
    # extra
    parser.add_argument("-nsv","--no-save", help="save the model and training history (default: False)", default=False, action="store_true")
    parser.add_argument("-now","--no-overwrite", help="overwrite the current model with the same dataset_name, model_name, and init_name (default: False)", default=False, action="store_true")
    parser.add_argument("-npr","--no-printing", help="print the used hyperparameters, dataset details, and model training progress (default: False)", default=False, action="store_true")
    # summary
    parser.add_argument("-an",'--avg_num', help='number of the evaluation for taking the average', default=10, type=int)
    # comet
    parser.add_argument("-uc",'--use-comet', help='use comet to store you progress (default: True)', default=False, action="store_true")
    parser.add_argument("-cpn",'--comet-project-name', help='comet project name', default=None, type=str)
    parser.add_argument("-cws",'--comet-workspace', help='comet workspace', default=None, type=str)
    parser.add_argument("-cen",'--comet-experiment-name', help='comet experiment name', default=None, type=str)
    parser.add_argument("-cak",'--comet-api-key', help='comet api key', default=None, type=str)
    
    args = parser.parse_args()

    # setup comet_ml
    if args.use_comet:
        from comet_ml import Experiment
        # connection
        api_key = args.comet_api_key if args.comet_api_key else 'APY_KEY'
        project_name = args.comet_project_name if args.comet_project_name else 'PROJECT_NAME'
        workspace = args.comet_workspace if args.comet_workspace else 'WORKSPACE'
        experiment_name = args.comet_experiment_name if args.comet_experiment_name else args.model_name
        # create experiment
        experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )
        experiment.set_name(experiment_name)
        # object
        comet = experiment
    else:
        comet = None
        
    if not (args.train or args.eval or args.eval_summary):
        print('please run the file properly. for more information: https://github.com/0aub/easy-pytorch-image-classification')
    if args.train or args.eval:
        main(args.exp, args.dataset_name, args.model_name, args.train, args.eval, not args.no_save, not args.no_overwrite, 
             args.batch_size, args.image_size, args.learning_rate, args.epochs, not args.no_printing, False, args.checkpoints_path, args.log_path, args.evals_path, comet)
    if args.eval_summary:
        evaluation_summary(args.exp, args.dataset_name, args.batch_size, args.avg_num, args.log_path, args.checkpoints_path)
