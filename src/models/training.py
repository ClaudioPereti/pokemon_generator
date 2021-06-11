import sys
sys.path.append('/home/claudio/machine_learning_project/pokemon_generator/pokemon_env/lib/python3.6/site-packages')
sys.path.append('../data')
sys.path.append('../features')
sys.path.append('../../config')
import config
from config import path_raw_images
from config import generator
from config import train, train_size, vae_optimizer, encoder, decoder
from load_data import load_pokemon_image
from build_features import gen_image_augmented
from build_model import VarationalConvAE,Encoder,Decoder,Sampling
from tensorflow import saved_model
from tensorflow.keras.models import load_model


def main():

    # Call the model
    vae = VarationalConvAE()
    # Compile the model
    vae.compile(optimizer = vae_optimizer)
    # Load processed Dataset
    #pokemon_array = load_pokemon_image()
    # Slice dataset to train_size
    #pokemon_array = pokemon_array[0:train_size,:,:,:]
    # Train model
    vae.fit(gen_image_augmented(**generator),**train)
    # Save model
    #vae.save('../../model/')




if __name__ == "__main__":
    main()
