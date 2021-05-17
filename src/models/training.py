import sys
sys.path.append('../data')
from load_data import load_pokemon_array
import importlib
importlib.reload(build_model)
from build_model import VarationalConvAE,Encoder,Decoder
from sklearn.model_selection import train_test_split

encoder = Encoder()
decoder = Decoder()


vae = VarationalConvAE(encoder,decoder)

vae.compile(optimizer = 'adam')
pokemon_array = load_pokemon_array()

X,X_test = train_test_split(pokemon_array)

vae.fit(pokemon_array,pokemon_array,epochs=50,batch_size=32)
