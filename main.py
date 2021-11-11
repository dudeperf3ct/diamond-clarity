import os

from train import ClarityClassifier

if __name__ == "__main__":
    model_name = "cnn3d"
    root_dir = 'dataset/round_all/all'
    model_dir = f'experiments/{model_name}'
    dataset_path = f'{root_dir}/diamonds_db.p'
    classifier = ClarityClassifier(dataset_path, root_dir, model_name, model_dir)
    train_data, val_data = classifier.create_dataset()
    #  train method
    classifier.train(train_data, val_data)
    print("=" * 80)
    # checkpoint_path = os.path.join(classifier.model_dir, classifier.model_name)
    # classifier.net.load_weights(checkpoint_path)
    ## save the best model at same level with filename model.h5
    # classifier.save_model(classifier.net)
    # test evaluate method
    # classifier.evaluate(val_data)
    print("="*80)
