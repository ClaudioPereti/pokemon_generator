import sys
sys.path.append('/home/claudio/machine_learning_project/pokemon_generator/pokemon_env/lib/python3.6/site-packages')
sys.path.append('../data')
sys.path.append('../../config')
import config
from config import train, train_size, vae_optimizer, encoder, decoder
from load_data import load_pokemon_array
from build_model import VarationalConvAE,Encoder,Decoder


def main():
    # call encoder and decoder
    encoder_model = Encoder(encoder)
    decoder_model = Decoder(decoder)

    # call the variation autoencoder
    vae = VarationalConvAE(encoder_model,decoder_model)


    vae.compile(optimizer = vae_optimizer)
    pokemon_array = load_pokemon_array()

    pokemon_array = pokemon_array[0:train_size,:,:,:]
    vae.fit(pokemon_array,**train)

if __name__ == "__main__":
    main()
