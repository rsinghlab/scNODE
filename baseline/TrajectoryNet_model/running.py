'''
Description:
    Wrapper for running TrajectoryNet model in our benchmarks
    Codes are adopted from TrajectoryNet source codes.

Reference:
    [1] https://github.com/KrishnaswamyLab/TrajectoryNet
'''
from baseline.TrajectoryNet_model.model_wrapper.train import train
from baseline.TrajectoryNet_model.model_wrapper.evaluate import eval
from baseline.TrajectoryNet_model.TrajectoryNet.parse import parser
from baseline.TrajectoryNet_model.model_wrapper.MyDataset import BenchmarkData


def TrajectoryNetTrain(data_name, split_type, dir_path, n_pcs, n_tps, leaveout_timepoint, n_iters, batch_size, top_k_reg=0.1, vecint=None):
    args = parser.parse_args()
    # Load daat
    args.dataset = data_name
    args.split_type = split_type
    args.data = BenchmarkData(
        data_name = args.dataset,
        dir_path = dir_path,
        split_type = args.split_type,
        n_pcs = n_pcs,
        use_velocity = False
    )
    pca_model = args.data.pca_model
    scalar_model = args.data.scalar_model
    # Model training
    args.weight_decay = 5e-5
    args.top_k_reg = top_k_reg
    args.vecint = vecint
    args.training_noise = 0.0
    args.niters = n_iters
    args.batch_size = batch_size
    args.use_growth = False
    args.leaveout_timepoint = leaveout_timepoint
    args.ntimes = n_tps
    args, model = train(args)
    return args, model, pca_model, scalar_model


def TrajectoryNetSimulate(args, model, n_tps, pca_model, scalar_model):
    args.ntimes = n_tps
    forward_latent_traj = eval(args, model)
    forward_recon_traj = [
        pca_model.inverse_transform(scalar_model.inverse_transform(each))
        for each in forward_latent_traj
    ]
    return forward_recon_traj, forward_latent_traj
