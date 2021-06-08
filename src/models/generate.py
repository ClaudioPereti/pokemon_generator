import sys
sys.path.append('/home/claudio/machine_learning_project/pokemon_generator/pokemon_env/lib/python3.6/site-packages')
sys.path.append('../data')
sys.path.append('../../config')
import config
from config import train, train_size, vae_optimizer, encoder, decoder
from load_data import load_pokemon_array
from build_model import VarationalConvAE,Encoder,Decoder,Sampling
from tensorflow import saved_model
from tensorflow.keras.models import load_model
from PIL import Image



def main():

    pokemon_array = load_pokemon_array()
    print(pokemon_array.shape)

    pokemon_array = pokemon_array[0:train_size,:,:,:]

    vae = load_model('../../model/',custom_objects={'VarationalConvAE': VarationalConvAE})
    img = Image.fromarray(vae.predict(pokemon_array)[0],'RGB')
    img.show()



if __name__ == "__main__":
    main()
