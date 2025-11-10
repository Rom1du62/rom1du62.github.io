# Librairie du projet 11

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns
from matplotlib.colors import to_rgba
palette = sns.color_palette("bright", 10)


def addAlpha(color, alpha):
    return to_rgba(color, alpha)
def display_parallel_coordinates(df, num_clusters, palette=None):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Palette par défaut
    if palette is None:
        palette = plt.cm.tab10.colors  # 10 couleurs standards

    # Sélection des points de chaque cluster
    cluster_points = [df[df.cluster == i] for i in range(num_clusters)]
    
    # Création de la figure
    fig = plt.figure(figsize=(12, 3 * num_clusters))
    fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Tracé
    for i in range(num_clusters):
        ax = fig.add_subplot(num_clusters, 1, i + 1)

        # Autres clusters en transparence
        for j, c in enumerate(cluster_points):
            if i != j:
                parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j % len(palette)], 0.2)], ax=ax)

        # Cluster principal en plus opaque
        parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i % len(palette)], 0.6)], ax=ax)

        # Décalage visuel des ticks
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)

    plt.show()     

def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)    

def detect_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Détecte les outliers dans un DataFrame en utilisant la méthode IQR.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame à analyser
    columns : list, optional
        Liste des colonnes à analyser. Si None, analyse toutes les colonnes numériques
    threshold : float, default=1.5
        Multiplicateur de l'IQR (1.5 = outliers modérés, 3.0 = outliers extrêmes)
    
    Returns:
    --------
    dict : Dictionnaire contenant les informations sur les outliers
        - 'outliers_mask': DataFrame booléen indiquant les outliers par colonne
        - 'outliers_indices': Indices des lignes contenant au moins un outlier
        - 'outliers_by_column': Dict avec le nombre d'outliers par colonne
        - 'stats': Statistiques (Q1, Q3, IQR, bornes) par colonne
    """
    
    # Sélectionner les colonnes numériques si non spécifiées
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers_mask = pd.DataFrame(index=df.index)
    stats_dict = {}
    outliers_by_column = {}
    
    for col in columns:
        # Calculer Q1, Q3 et IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculer les bornes
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Identifier les outliers
        outliers_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers_by_column[col] = outliers_mask[col].sum()
        
        # Sauvegarder les stats
        stats_dict[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    # Identifier les lignes contenant au moins un outlier
    outliers_indices = outliers_mask.any(axis=1)
    
    return {
        'outliers_mask': outliers_mask,
        'outliers_indices': outliers_indices,
        'outliers_by_column': outliers_by_column,
        'stats': stats_dict
    }

def remove_outliers(df, columns=None, threshold=1.5, verbose=True):
    """
    Supprime les lignes contenant des outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame à nettoyer
    columns : list, optional
        Colonnes à analyser pour les outliers
    threshold : float, default=1.5
        Multiplicateur de l'IQR
    verbose : bool, default=True
        Afficher les informations sur les outliers supprimés
    
    Returns:
    --------
    pandas.DataFrame : DataFrame nettoyé sans outliers
    """
    
    result = detect_outliers_iqr(df, columns, threshold)
    
    if verbose:
        print(f"Nombre total de lignes : {len(df)}")
        print(f"Nombre de lignes avec outliers : {result['outliers_indices'].sum()}")
        print(f"\nOutliers par colonne :")
        for col, count in result['outliers_by_column'].items():
            if count > 0:
                print(f"  {col}: {count}")
    
    # Retourner le DataFrame sans les lignes contenant des outliers
    df_clean = df[~result['outliers_indices']].copy()
    
    if verbose:
        print(f"\nLignes restantes après nettoyage : {len(df_clean)}")
    
    return df_clean

def get_outliers_details(df, columns=None, threshold=1.5):
    """
    Retourne un DataFrame avec les détails des outliers détectés.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame à analyser
    columns : list, optional
        Colonnes à analyser
    threshold : float, default=1.5
        Multiplicateur de l'IQR
    
    Returns:
    --------
    pandas.DataFrame : DataFrame contenant uniquement les lignes avec outliers
                       et une colonne indiquant quelles variables sont outliers
    """
    
    result = detect_outliers_iqr(df, columns, threshold)
    
    # Récupérer les lignes avec outliers
    outliers_df = df[result['outliers_indices']].copy()
    
    # Ajouter une colonne indiquant quelles variables sont outliers
    outliers_list = []
    for idx in outliers_df.index:
        outlier_cols = result['outliers_mask'].loc[idx]
        outlier_cols = outlier_cols[outlier_cols].index.tolist()
        outliers_list.append(', '.join(outlier_cols))
    
    outliers_df['outlier_variables'] = outliers_list
    
    return outliers_df

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # Initialise the matplotlib figure      
            fig = plt.figure(figsize=(7,6))
        
            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
         
def display_factorial_planes(X_projected, x_y, pca=None, labels = None, clusters=None, alpha=1, figsize=[10,8], marker="." ):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # on définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters
 
    # Les points    
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha, 
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) : 
        # j'ai copié collé la fonction sans la lire
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()

def plot_dendrogram(Z, names, figsize=(10,25)):
    '''Plot a dendrogram to illustrate hierarchical clustering'''

    plt.figure(figsize=figsize)
    plt.title('Classification Hierarchique Ascendente -- Dendrogramme --')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()

def correlation_graph(pca, x_y, features) : 
    """Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.04,
                pca.components_[y, i] + 0.04,
                features[i])
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # J'ai copié collé le code et je l'ai lu
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)

def Affich_scree_plot(pca):

    '''Affichage du scree plot pour le pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Pourcentage de variance expliquée")
    plt.title("Graphique Scree plot")
    plt.show(block=False)
