# Chargement des packages
import jax
import flax
import optax
import orbax

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import ArrayLike
import matplotlib.pyplot as plt

from typing import Tuple, Callable
from math import sqrt

# Initialisation de l'entraînement

def generate_train_step(key, model, optimizer, batch_size, num_classes, dim_params):
    '''
    :Objectif: initialiser les paramètres du modèle et l'optimiseur, ainsi que les fonctions qui seront utilisées lors de l'entraînement
    
    :param key: jax.random.PRNGKey()
    :param model: modèle dont on veut optimiser les paramètres
    :param optimizer: algorithme d'optimisation des paramètres du modèle
    :param batch_size: les données sont réparties dans des échantillons de taille `batch_size`
    :param num_classes: nombre de classes dans le jeu de données
    :param dim_params: nombre de paramètres de la première couche d'encodage du VAE
    
    :return: les fonctions train_step, train ainsi que les paramètres et l'état de l'optimiseur
    '''
    params = model.init(key, jnp.zeros((batch_size, dim_params)), jnp.zeros((batch_size, num_classes)), jax.random.PRNGKey(0)) # peu importe le nombre de la clé
    opt_state = optimizer.init(params)
    
    def loss_fun(params, x, c, key):
        '''
        :Objectif: calculer la loss globale, i.e. la somme de la MSE et de la kl-divergence
        
        :param params: paramètres du modèle permettant de faire les prédictions dont les écarts serviront à estimer la loss
        :param x: batch de données dont la loss moyenne sera calculée
        :param c: matrice de vecteurs one_hot indiquant la classe respective à chaque donnée du batch
        :param key: jax.random.PRNGKey()
        
        :return: loss du batch, (score MSE du batch, score de kl_divergence du batch)
        '''
        reduce_dims = list(range(1, len(x.shape)))
        c = jax.nn.one_hot(c, num_classes) # encodage one hot des classes d'image
        recon, mean, logvar = model.apply(params, x, c, key)
        mse_loss = optax.l2_loss(recon, x).sum(axis=reduce_dims).mean()
        kl_loss = jnp.mean(-0.5 * jnp.sum(1 + logvar - mean ** 2 - jnp.exp(logvar), axis=reduce_dims)) # kl loss : permet de garder les sorties de l'encodeur proches de la distribution normale centrée réduite
        
        # loss = mse_loss + kl_weight * kl_loss # kl_weight = 0.5 par défaut (initialisation un peu plus haut)
        loss = mse_loss + kl_loss
        return loss, (mse_loss, kl_loss)
        
    @jax.jit # JIT = compilation Just In Time pour une exécution XLA (Accelerated Linear Algebra), permet un usage direct sans la fonction mère
    def train_step(params, opt_state, x, c, key):
        '''
        :Objectif: calcul du gradient des paramètres, mise à jour et calcul de la loss
        
        :param params: paramètres du modèle avant mise à jour
        :param opt_state: état de l'optimiseur avant mise à jour
        :param x: batch de données
        :param c: matrice de vecteurs one_hot indiquant la classe respective à chaque donnée du batch
        :param key: jax.random.PRNGKey()
        
        :return: paramètres du modèle mis à jour, état de l'optimiseur mise à jour, score de loss du batch, mse loss, kl_divergence loss
        '''
        losses, grads = jax.value_and_grad(loss_fun, has_aux=True)(params, x, c, key) # 1ères parenthèses : initialisation des params de la fonction, 2ndes parenthèses : application de la fonction à ces input
        loss, (mse_loss, kl_loss) = losses
        
        updates, opt_state = optimizer.update(grads, opt_state, params) # updates = grads transformés, opt_state = nouvel état de l'optimiseur
        params = optax.apply_updates(params, updates) # application des grads transformés aux paramètres afin d'obtenir les nouveaux params
        
        return params, opt_state, loss, mse_loss, kl_loss
    
    # Exécution de l'entraînement
    #def train(key, params, freq, epochs, opt_state, train_loader, batch_size, train_step, grayscale=False):
    def train(key, params, freq, epochs, opt_state, train_loader, batch_size, train_step):
        '''
        :Objectif: entraînement du modèle et optimisation de ses paramètres sur le jeu de données spécifié
        
        :param key: jax.random.PRNGKey()
        :param freq: fréquence d'affichage de l'information relative à la progression de l'entraînement
        :param epochs: nombre de tours complets du jeu de données effectués pour entraîner le modèle
        :param opt_state: état de l'optimiseur
        :param train_loader: conteneur des données et des labels
        :param batch_size: taille du batch
        :param train_step: fonction train_step "jit compiled"
        
        :return: paramètres du modèles optimisés, et état final de l'optimiseur
        '''
        for epoch in range(epochs):
            total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
            for i, (batch, c) in enumerate(train_loader): # batch (16, 1, 28, 28) : tenseur de 16 images en 2D de dim 28x28 pixels
                                                          # c (16) : vecteur de classes des 16 images du batch
                                                          # classes possibles : 0 à 9 (tous les chiffres)
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
                    
                c = c.numpy()
                params, opt_state, loss, mse_loss, kl_loss = train_step(params, opt_state, batch, c, subkey) # train_step en paramètres ?

                total_loss += loss
                total_mse += mse_loss
                total_kl += kl_loss

                if i > 0 and not i % freq:
                    print(f"Epoch {epoch} | étape {i} | loss = {total_loss / freq} ~ mse = {total_mse / freq}. kl = {total_kl / freq}")
                    total_loss = 0.
                    total_mse, total_kl = 0.0, 0.0
        return params, opt_state

    return train_step, train, params, opt_state