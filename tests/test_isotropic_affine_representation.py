from isotropic_affine_representation_experiment import train_one
from learned_affine_representation_experiment import Config


def test_isotropic_training_smoke():
    model,_=train_one(4,0,2,0.01,Config())
    assert model.codebook().shape==(31,4)
