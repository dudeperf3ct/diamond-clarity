import os
from tensorflow import keras
from train import ClarityClassifier

if __name__ == "__main__":
    experiment = 'cnn3d_tf'
    model_name = "cnn3d"
    root_dir = 'dataset/round_all/all'
    model_dir = f'experiments/{model_name}'
    dataset_path = f'{root_dir}/diamonds_db.p'
    classifier = ClarityClassifier(experiment, dataset_path, root_dir, model_name, model_dir)
    train_data, val_data = classifier.create_dataset()
    #  train method
    classifier.train(train_data, val_data)
    print("=" * 80)
    checkpoint_path = os.path.join(classifier.model_dir, classifier.model_name)
    classifier.model = keras.models.load_model(checkpoint_path+'.h5')
    # test evaluate method
    classifier.evaluate(val_data)
    print("="*80)
