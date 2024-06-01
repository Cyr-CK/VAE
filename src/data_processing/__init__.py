def prepare_test_set(dataset, class_of_ref, mixed_classes=True):
    '''
    :Objectif: construire un tenseur d'images de mêmes classes (analyse des anomalies intra-classes, i.e. des faux positifs) ou de classes différentes (analyse des anomalies inter-classes en plus, i.e. des faux négatifs)
    
    :dataset: dictionnaire de données de sous branches `targets` (classe par image) et `data` (tenseur d'images) (en. should be a dict with the sub-branches `targets` (class index per image) and `data` (image tensor))
    :class_of_ref: classe de référence de laquelle les anomalies potentielles sont comparées (en. class of reference to which potential anomalies are compared)
    :param mixed_classes: indique si le jeu de données contient seulement des images de classe `class_of_ref` (False) ou non (True) (en. tells whether the dataset contains only images in the category class_of_ref (False) or not (True))
    
    :return: tenseur d'images
    '''
    if mixed_classes == False:
        idx = []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == class_of_ref:
                idx.append(i)
        return dataset.data[idx]
    else:
        return dataset.data

def get__y_true(dataset, class_of_ref):
    '''
    :Objectif: construire le vecteur de classifications réelles auxquelles les prédictions d'anomalie seront comparées
    
    :dataset: dictionnaire de données de sous branches `targets` (classe par image) et `data` (tenseur d'images) (en. should be a dict with the sub-branches `targets` (class index per image) and `data` (image tensor))
    :class_of_ref: classe de référence de laquelle les anomalies potentielles sont comparées (en. class of reference to which potential anomalies are compared)
    
    :return: vecteur indiquant par True si l'image de même indice est en réalité une anomalie, et inversement par False
    '''
    return dataset.targets != class_of_ref