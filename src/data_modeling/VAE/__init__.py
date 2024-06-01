# Chargement des packages
import jax
import flax
import optax
import orbax

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Tuple, Callable
from math import sqrt
import matplotlib.pyplot as plt
from numpy import einsum # somme de Einstein


class FeedForward(nn.Module):
    dimensions: Tuple[int] = (256, 128, 64)
    activation_fn: Callable = nn.relu
    drop_last_activation: bool = False
    
    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike: # -> : annotation du type qui sera retourné par la fonction
        '''
        :Objectif: envoyer des données dans un réseau de neurones de dimensions spécifiées dans la classe FeedForward
        :param x: données soumises à la propagation en avant (feedforward)
        :return: sortie des données
        '''
        for i, d in enumerate(self.dimensions):
            # itère à travers chaque dimension (256, 128, 64)
            x = nn.Dense(d)(x) # pour toutes les dimensions, effectue une transformation linéaire
            if i != len(self.dimensions) - 1 or not self.drop_last_activation:
                # pour toutes les dimensions sauf la dernière, effectue une transformation ReLU (remplace les transformations linéaires effectuées juste précédemment)
                # par conséquent seule la dernière dimension peut subir une transformation renvoyant une valeur négative
                # car ReLU retourne des valeurs nulles ou positives
                x = self.activation_fn(x) # i.e. une transformaton ReLU
        return x

# type : image processing
class VAE(nn.Module):
    encoder_dimensions: Tuple[int] = (256, 128, 64) # 28*28 (784) pixels --> 8*8 (64) pixels
    decoder_dimensions: Tuple[int] = (128, 256, 784)
    latent_dim: int = 4
    activation_fn: Callable = nn.relu
    
    def setup(self):
        '''
        :Objectif: construire les couches du VAE du début (encodeur) à la fin (décodeur)
        
        :return : None
        '''
        self.encoder = FeedForward(self.encoder_dimensions, self.activation_fn) # couche : encodeur
        self.pre_latent_proj = nn.Dense(self.latent_dim * 2) # couche : projection dans l'espace latent (4*2 afin d'avoir 4 et 4 pour mean/logvar lors du jnp.split)
        self.post_latent_proj = nn.Dense(self.encoder_dimensions[-1]) # couche : projection hors de l'espace latent
        self.class_proj = nn.Dense(self.encoder_dimensions[-1]) # couche : projection du vecteur de classes en même dimension que la dernière couche de l'encodeur
        self.decoder = FeedForward(self.decoder_dimensions, self.activation_fn, drop_last_activation=False) # couche : décodeur
    
    def reparam(self, mean: ArrayLike, logvar: ArrayLike, key: jax.random.PRNGKey) -> ArrayLike:
        '''
        :Objectif: reparameterization trick : 
                    transformation différentiable (et donc déterministe) de l'échantillon encodé (alors non différentiable)
                    i.e. génération de données différentiables à partir de la distribution extraite après l'encodage
        
        :param mean: moyenne de la distribution des données extraite après l'encodage (cf. `encode`)
        :param logvar: idem mais logarithme de la variance
        :param key: jax.random.PRNGKey()
        
        :return: échantillon suivant une loi normale de moyenne mean et d'écart-type std
        '''
        std = jnp.exp(logvar * 0.5) # écart-type
        eps = jax.random.normal(key, mean.shape) # échantillonne une loi normale centrée-réduite en dimension mean.shape
                            # dim std/eps : (batch_size , latent_dim)
        return eps * std + mean # retourne une loi normale non centrée et non réduite (inverse de (eps - mean)/std)
    
    
    def encode(self, x: ArrayLike):
        '''
        :Objectif: encode les données dans l'espace latent d'où la distribution est extraite
        
        :param x: données à encoder
        
        :return: moyenne et log-variance de la distribution des données une fois dans l'espace latent
        '''
        x = self.encoder(x) # envoie des données dans l'encodeur, i.e. projection dans l'espace latent (taille double pour le split d'après)
                            # dim : (batch_size , decoder_dimensions[-1]) --> (batch_size , encoder_dimensions[-1])
        mean, logvar = jnp.split(self.pre_latent_proj(x), 2, axis=-1) # séparation en deux pour extraire la moyenne et la variance
                            # dim mean/logvar : (batch_size , latent_dim)
        return mean, logvar
    
    def decode(self, x: ArrayLike, c: ArrayLike):
        '''
        :Objectif: décode les données à partir de l'espace latent dans l'espace initial
        
        :param x: échantillon de données généré à partir de la distribution dans l'espace latent (données issues de `reparam`)
        :param c: vecteur de classe des données
        
        :return: données après décodage
        '''
        x = self.post_latent_proj(x) # transformation des données latentes en x
                                     # dim : (batch_size , encoder_dimensions[-1])
        x = x + self.class_proj(c) # injecte la sortie du vecteur de classes
                                   # dim c : (batch_size , n_classes)
                                   # dim self.class_proj(c) : (batch_size , encoder_dimensions[-1])
        x = self.decoder(x) # envoie des données dans le décodeur, i.e. projection dans l'espace de départ
                            # dim : (batch_size , encoder_dimensions[-1]) --> (batch_size , decoder_dimensions[-1])
        return x
    
    def __call__(self, x: ArrayLike, c: ArrayLike, key: jax.random.PRNGKey) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        '''
        :Objectif: exécute toutes les fonctions précédentes dans l'ordre (utilisée lors de l'entraînement)
        
        :param x: données à encoder dans l'espace latent où la distribution sera extraite et utilisé pour générer un nouvel échantillon de données différentiable qui sera alors décodé en l'envoyant dans l'espace de départ
        :param c: vecteur de classe des données
        :param key: jax.random.PRNGKey()
        
        :return: données après décodage, moyenne et log-variance de la distribution apprise
        '''
        mean, logvar = self.encode(x) # i.e. maxpooling/downsampling (pour un CNN)
        z = self.reparam(mean, logvar, key)
        y = self.decode(z, c) # i.e. upsampling (pour un CNN)
        return y, mean, logvar
    
    def img_gen(self, key: jax.random.PRNGKey, params, n_classes: ArrayLike, img_class, h=4, w=4) -> jnp.array:
        '''
        :Objectif: générer des images à partir de la distribution de données dans l'espace latent apprise par le VAE
        
        :param key: jax.random.PRNGKey()
        :param params: paramètres du modèle servant à effectuer les prédiction (i.e. la génération des tenseurs desciptifs des images)
        :param n_classes: nombre de classes dans le jeu de données
        :param img_class: classe à partir de laquelle on veut générer des images
        :param h: nombre de lignes d'images à générer
        :param w: nombre de colonnes d'images à générer
                    par défaut, h=4 et w=4 c'est-à-dire que 4x4 images seront générées
        
        :return: array contenant les tenseurs des images générées (le graphique n'est pas retourné, seulement imprimé)
        '''
        # num_samples = 16
        key, z_key = jax.random.split(key)
        z = jax.random.normal(z_key, (h*w, self.latent_dim))
        c = jnp.repeat(img_class, h*w, axis=-1).flatten()
        c = jax.nn.one_hot(c, n_classes)

        # normaliser ou dénormaliser ?
        sample = self.apply(params, z, c, method=self.decode)
        for i in range(sample.shape[0]): # i.e. batch_size=16            
            if jnp.max(sample[i]) > 1:
                sample = sample.at[i].set((sample[i] / jnp.max(sample[i]))*1) # normalisation
            else:
                sample = sample.at[i].set(sample[i] * 1) # normalisation
            #pass
        sample_resh = einsum('ikjl', jnp.asarray(sample).reshape(h, w, 28, 28)).reshape(28*h, 28*w) # redimensionnement
        plt.imshow(sample_resh, cmap='gray')
        plt.show()

        return sample_resh
    
    #def evaluate(self, key: jax.random.PRNGKey, params, n_classes: ArrayLike, data_loader, batch_size, grayscale=False, get_loss_distrib = True):
    def evaluate(self, key: jax.random.PRNGKey, params, n_classes: ArrayLike, data_loader, batch_size, get_loss_distrib = True):
        '''
        :Objectif: évaluer l'erreur du modèle sur un jeu de données spécifié dans un data loader
        
        :param key: jax.random.PRNGKey()
        :param params: paramètres du modèle à partir desquels les prédictions sont faites
        :param n_classes: nombre de classes dans le jeu de données
        :param data_loader: conteneur de données de type DataLoader (dans torch.utils.data)
        :param batch_size: taille des batchs de données
        :param get_loss_distrib: indique si oui ou non, la distribution de l'erreur doit être retournée
        
        :return: loss moyenne, mse moyenne, kl-divergence moyenne et la distribution de la loss si get_loss_distrib = True
        '''
        total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
        tot_mse, tot_kl = [], []
        for i, (batch, c) in enumerate(data_loader):
            key, subkey = jax.random.split(key)
            
            # Utile si images en couleurs
            #if grayscale == True:
            #    if len(batch.shape) == 3:
            #        pass
            #    else:
            #        batch = batch.numpy().mean(1)
            #        batch = batch.reshape(batch_size, batch.shape[-1]*batch.shape[-2])
            #else:
            #    batch = batch.numpy().reshape(batch_size, batch.shape[-1]*batch.shape[-2])
            batch = batch.numpy().reshape(batch_size, batch.shape[-1]*batch.shape[-2])
                
            reduce_dims = list(range(1, len(batch.shape)))
            c = c.numpy()
            c = jax.nn.one_hot(c, n_classes) # encodage one hot des classes d'image
            recon, mean, logvar = self.apply(params, batch, c, subkey)
            mse_loss = optax.l2_loss(recon, batch).sum(axis=reduce_dims).mean()
            kl_loss = jnp.mean(-0.5 * jnp.sum(1 + logvar - mean ** 2 - jnp.exp(logvar), axis=reduce_dims))
            loss = mse_loss + kl_loss

            total_loss += loss
            total_mse += mse_loss
            total_kl += kl_loss
            
            if get_loss_distrib == True:
                tot_mse.append(optax.l2_loss(recon, batch).sum(axis=reduce_dims).tolist())
                tot_kl.append((-0.5 * jnp.sum(1 + logvar - mean ** 2 - jnp.exp(logvar), axis=reduce_dims)).tolist())
        print(f'Loss totale moyenne = {total_loss / i}')
        print(f'MSE totale moyenne = {total_mse / i}')
        print(f'kl totale moyenne = {total_kl / i}')
        
        if get_loss_distrib == True:
            loss_distrib = jnp.sum((jnp.asarray(tot_mse) + jnp.asarray(tot_kl)) / batch_size, axis=1).tolist()
            return total_loss/i, total_mse/i, total_kl/i, loss_distrib
        else:
            return total_loss/i, total_mse/i, total_kl/i
    
    #def det_anom(self, key: jax.random.PRNGKey, params, img_array, n_classes, tested_class, threshold, grayscale=False):
    def det_anom(self, key: jax.random.PRNGKey, params, img_array, n_classes, tested_class, threshold):
        '''
        :Objectif: détecter les anomalies au-delà d'un seuil d'erreur dans un jeu de données en considérant une classe comme la norme
                    (en théorie, les images de classe différente seront correctement classifiées comme anomalie,
                    bien que certaines images de la classe de référence puissent aussi être incorrectement classifiées ainsi)
        
        :param key: jax.random.PRNGKey()
        :param params: paramètres du modèle à partir desquels les prédictions sont faites
        :param img_array: tenseur contenant les tenseurs de chaque image
        :param n_classes: nombre de classes dans le jeu de données
        :param tested_class: classe de référence; en théorie les images de classes autres que `tested_class` seront des anomalies
        :param threshold: seuil de loss au-delà duquel une image sera considérée comme une anomalie (i.e. une grande loss indique une reconstruction difficile à partir de la distribution des images de classe `tested_class`, donc potentiellement une anomalie)
        
        :return: dict de données contenant : tenseur, loss, mse, kl loss et le statut d'anomalie True/False
        '''
        key, subkey = jax.random.split(key)
        anomalies = {"image": [], "estAnomalie": [], "loss": [], "mse": [], "kl": []}
        for i, img in enumerate(img_array):
            
            # Utile si images en couleurs
            #if grayscale == True:
            #    if len(img.shape) == 3:
            #        pass
            #    else:
            #        img = img.numpy().mean(1)
            #        img = img.reshape(img.shape[-1]*img.shape[-2], -1)
            #else:
            #    img = img.numpy().reshape(img.shape[-1]*img.shape[-2], -1)
            img = img.numpy().reshape(img.shape[-1]*img.shape[-2], -1)
            
                         
            if jnp.max(img) > 1:
                img = (img / jnp.max(img)) * 1 # normalisation
            else:
                img = img * 1 # normalisation
                                            
            img = img.flatten()
            c = jax.nn.one_hot(tested_class, n_classes) # encodage one hot des classes d'image
            recon, mean, logvar = self.apply(params, img, c, subkey)
            mse_loss = optax.l2_loss(recon, img).sum(axis=0).mean()
            kl_loss = jnp.mean(-0.5 * jnp.sum(1 + logvar - mean ** 2 - jnp.exp(logvar), axis=0))
            loss = mse_loss + kl_loss
            print(f"Image {i} | loss = {loss} ~ mse = {mse_loss}. kl = {kl_loss}")
            
            anomalies["image"].append(img.reshape(28,28))
            anomalies["loss"].append(loss)
            anomalies["mse"].append(mse_loss)
            anomalies["kl"].append(kl_loss)
            if loss > threshold:
                anomalies["estAnomalie"].append(True)
            else:
                anomalies["estAnomalie"].append(False)    
        anomalies["image"] = jnp.asarray(anomalies["image"])
        anomalies["loss"] = jnp.asarray(anomalies["loss"])
        anomalies["mse"] = jnp.asarray(anomalies["mse"])
        anomalies["kl"] = jnp.asarray(anomalies["kl"])
        anomalies["estAnomalie"] = jnp.asarray(anomalies["estAnomalie"])
        return anomalies
    
    # def train
    
# class VAE_colorisation

# class VAE_denoising

# type : numerical data processing
