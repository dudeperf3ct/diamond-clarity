"""Sample main.py to run as python script"""
from train import ClassifierModel

if __name__ == '__main__':
    ################################
    # Model training and evaluation
    model_dir = "experiments/cnnlstm/resnet18"
    experiment = "cnnlstm"
    model = ClassifierModel(model_dir, experiment)
    train_df = model.prepare_df("dataset/round_all/all/diamonds_db.p")
    # perform k-fold training
    if model.folds > 1:
        model.train_k_folds(train_df)
    # perform simple training
    else:
        model.train(train_df)